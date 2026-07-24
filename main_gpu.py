"""
main.py
─────────────────────────────────────────────────────────────────────────────
Orchestrateur du pipeline de vérification des images de prostate.
 
Structure :
    Étape 1 — Chargement brut + dimensions
    Étape 2 — Suppression d'arrière-plan + cadre de matrice
    Étape 3 — Recadrage autour du contenu
    Étape 4 — Mise en matrice carrée diagonale (a × a)
 
Tous les résultats intermédiaires sont conservés dans `results`
(liste de dicts) pour l'ajout de nouvelles étapes.
"""

#  PARAMÈTRES images TIFF
PATHS = [
    r"C:\\Users\\Gustave\\Documents\\Education\\L3_EEEA\\Stage\\Pour_Arthur\\Prostate4\\A7_x1.25_z0.tif",
    r"C:\\Users\\Gustave\\Documents\\Education\\L3_EEEA\\Stage\\Pour_Arthur\\Prostate4\\A8_x1.25_z0.tif",
]

# Créer une image TIFF combinée en sortie ?
MAKE_TIFF = False
# Image combinée en sortie TIF
TIFF_OUT_PATH = r"C:\Users\Gustave\Documents\Education\L3_EEEA\Stage\Pour_Arthur\Results\combined_tiff.tif"

# Dossier avec les cartes de déplacement
DISP_FOLDER = r"C:\Users\Gustave\Documents\Education\L3_EEEA\Stage\python\cartes_dep"

# Images d'entrée NIfTI
NIFTI_PATH_GAUCHE = r"C:\Users\Gustave\Documents\Education\L3_EEEA\Stage\Pour_Arthur\Images\A7\tumor_map_P4_A7.nii\tumor_map_P4_A7.nii"
NIFTI_PATH_DROITE = r"C:\Users\Gustave\Documents\Education\L3_EEEA\Stage\Pour_Arthur\Images\A8\tumor_map_P4_A8.nii\tumor_map_P4_A8.nii"

# Image de sortie NIfTI
NIFTI_OUT_COMBINED = r"C:\Users\Gustave\Documents\Education\L3_EEEA\Stage\Pour_Arthur\Results\warped_P4_combined.nii"

# Overlay TIFF et NIfTI pour vérification
OVERLAY_OUT_PATH = r"C:\Users\Gustave\Documents\Education\L3_EEEA\Stage\Pour_Arthur\Results\overlay_tiff_nii.tif"

# Nombre d'éléments a récupérer dans chaque image de base
TOP_N_LIST = [2, 1] # Pour Prostate 4 A10 -> FLIP = 1

# Inverser sur l'axe horizontal une image [Gauche, Droite] (0 = non, 1 = oui)
FLIP = [0, 0] # Pour Prostate 4 A7 -> TOP_N_LIST = 2

# Facteur de réduction de la taille des images (1 = original, 2 = divisé par 2, 4 = par 4, etc.)
DOWNSCALE_FACTOR = 16

# ── Activer / désactiver les étapes de vérification intermédiaires ──────────
SHOW_STEPS = False   # False = passe directement à l'étape face-à-face
# ────────────────────────────────────────────────────────────────────────────

######################################################################################################################################################################""

# Bibliothèque globales
import nibabel as nib
import os
import napari
import numpy as np
import math
import matplotlib.pyplot as plt
import psutil
import time

# Bibliothèques internes du projet
from processing import (
    load_image,
    remove_background,
    pad_to_diagonal_square,
    orient_and_face,
    flip_image_vertically,
    filtrer_doublons_y,
    get_center_of_mass
    )

from napari_display import (
    show_raw_images,
    show_masked_images,
    show_square_matrices,
    show_facing_images
    )

from nii_remap_pipeline import run_nii_remap_pipeline

# Test de GPU Nvidia
try :
    import cupy as cp
    print("Cupy installé")
    print("Mode avec GPU Nvidia")
    GPU = True
except ModuleNotFoundError:
    print("Cupy non présent.")
    print("Mode sans GPU Nvidia")
    GPU = False

if GPU:
    from f_tps_warp_gpu import auto_warp_image
elif not GPU:
    from f_tps_warp_2 import auto_warp_image

# OpenCV avec CUDA
print("Tentative d'import de OpenCV CUDA")
try :
    dossier_dlls = os.path.join(os.path.dirname(__file__), "dlls")
    os.add_dll_directory(dossier_dlls)
    import cv2
    print(cv2.__version__)
    print("OpenCV CUDA activé")
except FileNotFoundError:
    print("OpenCV sans CUDA")
    import cv2
    print(f"Version OpenCV : {cv2.__version__}")

mem = psutil.virtual_memory()
total_ram_gb = mem.total / (1024 ** 3)
available_ram_gb = mem.available / (1024 ** 3)

print(f"Quantité totale de RAM : {total_ram_gb:.2f} Go")
print(f"Quantité disponible de RAM : {available_ram_gb:.2f} Go")

color_liste = [
    "red", "blue", "green", "yellow", "cyan", "magenta", "orange", "purple",
    "pink", "brown", "lime", "teal", "navy", "maroon", "olive", "coral",
    "turquoise", "violet", "gold", "indigo", "salmon", "plum", "khaki",
    "orchid", "sienna", "tomato", "aquamarine", "crimson", "chartreuse",
    "cornflowerblue", "darkorange", "deeppink", "dodgerblue", "firebrick",
    "forestgreen", "fuchsia", "hotpink", "indianred", "lawngreen", "mediumpurple"
]

def _open_viewer(title: str, step: int, total: int) -> napari.Viewer:
    full_title = f"[{step}/{total}] {title}"
    print(f"\n  → Fenêtre Napari : « {full_title} »  (fermer pour continuer)")
    return napari.Viewer(title=full_title)


def snap_au_contour_vectorise(points_cliques, liste_contours):
    # points_cliques : array de shape (N, 2)
    # liste_contours : array de shape (M, 2)
    # Calcule les distances entre chaque point_clique et chaque point du contour
    distances = np.linalg.norm(liste_contours[:, np.newaxis, :] - points_cliques, axis=2)
    # Trouve l'index du contour le plus proche pour chaque point_clique
    indices_plus_proches = np.argmin(distances, axis=0)
    # Retourne les points du contour les plus proches
    return liste_contours[indices_plus_proches]


def cs_sort(l): # Tri par ordre croissant suivant y, pour trier les points de singularité placés dans l'ordre.
    sorted_indices = np.argsort(l[:, 0])
    l_sorted = l[sorted_indices]
    return l_sorted

def segmenter_ligne(ligne_complete, points_cles):
    index_cles = []
    
    # On extrait uniquement la colonne des 'y' (première colonne)
    y_ligne = ligne_complete[:, 0]
    
    for point in points_cles:
        y_cible = point[0]
        
        # Comparaison uniquement sur les valeurs de y
        correspondance = np.where(np.isclose(y_ligne, y_cible))[0]
        
        if len(correspondance) > 0:
            # S'il y a plusieurs points avec le même 'y', on prend la première occurrence
            index_cles.append(correspondance[0])
        #else:
            #print(f"Attention : La coordonnée y={y_cible} n'a pas été trouvée dans la ligne.")
            
    # S'assurer que les index sont dans l'ordre croissant
    index_cles.sort()
    
    segments = []
    
    for i in range(len(index_cles) - 1):
        debut = index_cles[i] + 1
        fin = index_cles[i+1]
        
        segment = ligne_complete[debut:fin]
        segments.append(segment)
        
    return segments


def frac_n(x):
    a = math.floor(x) # Entier en dessous
    b = math.ceil(x) # Entier au dessus
    c = x - a # Partie décimale
    return a, b, c


def lerp_coordinates(p0: tuple, p1: tuple, t: float)-> tuple:
    """
    Calcule l'interpolation linéaire entre deux points 2D.
    p0: tuple ou liste (y, x) du point de départ
    p1: tuple ou liste (y, x) du point d'arrivée
    t: float, la proportion de 0.0 à 1.0
    """
    y = p0[0] + t * (p1[0] - p0[0])
    x = p0[1] + t * (p1[1] - p0[1])
    
    return (y, x)


def arr2tup(arr: np.ndarray):
    return (arr[0], arr[1])

def get_anchor_points(img: np.ndarray, left: bool = True, n_points: int = 20) -> np.ndarray:
    """
    Retourne N points régulièrement espacés sur le côté opposé à la déformation,
    à utiliser comme points de contrôle fixes (src == dst) pour ancrer la TPS.

    Args:
        img      : image RGBA (H, W, 4)
        left     : True = ancres à gauche, False = ancres à droite
                   (mettre le côté OPPOSÉ à votre déformation)
        n_points : nombre de points d'ancrage

    Returns:
        (N, 2) np.ndarray en [y, x]
    """
    coords = np.array(get_half_shape_coordinates(img, left=left))

    if len(coords) == 0:
        return np.empty((0, 2))

    # Sous-échantillonnage régulier pour n_points bien répartis en Y
    indices = np.linspace(0, len(coords) - 1, n_points, dtype=int)
    sampled = coords[indices]

    # Ajouter toute la colonne (x = 0 ou x = w-1) + la colonne décalée de 10 px
    h, w = img.shape[0], img.shape[1]
    if left:
        x0 = 0
        x1 = min(x0 + 10, w - 1)
    else:
        x0 = w - 1
        x1 = max(x0 - 10, 0)

    col0 = np.column_stack((np.arange(h), np.full(h, x0)))  # (y, x0)
    col1 = np.column_stack((np.arange(h), np.full(h, x1)))  # (y, x1)

    # Concaténer les points échantillonnés et les deux colonnes (éviter doublons)
    #if sampled.size == 0:
    combined = np.vstack([col0, col1])
    #else:
        #combined = np.vstack([sampled, col0, col1])

    # Assurer le type et retourner
    return combined.astype(np.float64)



def get_center_column_coordinates(img: np.ndarray, left: bool = True) -> list[list[int]]:
    """
    Retourne des points autour de la colonne du centre de masse.
    Pour chaque image, on ajoute aussi les points des lignes supérieure et
    inférieure de l'image, depuis le coin gauche (pour l'image gauche) ou le
    coin droit (pour l'image droite) jusqu'à la colonne du centre de masse.
    """
    if img.ndim != 3 or img.shape[2] != 4:
        raise ValueError("L'image doit être au format RGBA (4 canaux).")

    alpha_channel = img[:, :, 3]
    coords = np.argwhere(alpha_channel > 0)

    if len(coords) == 0:
        return []

    M = cv2.moments(alpha_channel)
    if M["m00"] != 0:
        center_y = M["m01"] / M["m00"]
        center_x = M["m10"] / M["m00"]
    else:
        center_y = np.mean(coords[:, 0])
        center_x = np.mean(coords[:, 1])

    center_row = int(np.clip(int(round(center_y)), 0, img.shape[0] - 1))
    center_col = int(np.clip(int(round(center_x)), 0, img.shape[1] - 1))

    row_coords = coords[coords[:, 0] == center_row]
    if row_coords.size == 0:
        y_values = np.unique(coords[:, 0])
        nearest_y = y_values[np.argmin(np.abs(y_values - center_y))]
        center_row = int(nearest_y)
        row_coords = coords[coords[:, 0] == center_row]

    if row_coords.size == 0:
        return []

    if left:
        x_edge = int(np.min(row_coords[:, 1]))
        x_start = 0
    else:
        x_edge = int(np.max(row_coords[:, 1]))
        x_start = img.shape[1] - 1

    center_point = [center_row, x_edge]
    h, w = img.shape[:2]
    top_y = 0
    bottom_y = h - 1

    if x_start <= center_col:
        x_range = list(range(x_start, center_col + 1))
    else:
        x_range = list(range(x_start, center_col - 1, -1))

    top_points = [[top_y, x] for x in x_range]
    bottom_points = [[bottom_y, x] for x in x_range]

    points = [center_point] + top_points + bottom_points
    unique_points = []
    seen = set()
    for y, x in points:
        coord = (int(y), int(x))
        if coord not in seen:
            seen.add(coord)
            unique_points.append([coord[0], coord[1]])

    return unique_points



def get_half_shape_coordinates(img: np.ndarray, left: bool = True) -> list[list[int]]:
    """
    Parcourt une image RGBA et retourne les coordonnées des pixels de la forme
    situés à gauche ou à droite de son centre de masse.
    
    Args:
        img (np.ndarray): Image source au format (H, W, 4) (RGB + Alpha).
        left (bool): Si True, retourne les points à gauche du centre.
                     Si False, retourne les points à droite.
                     
    Returns:
        list[list[int]]: Liste de coordonnées sous la forme [[y, x], [y, x], ...]
    """
    # Vérification des dimensions : on s'assure qu'on a bien le canal alpha
    if img.ndim != 3 or img.shape[2] != 4:
        raise ValueError("L'image doit être au format RGBA (4 canaux).")
        
    # Extraction du masque de la forme via le canal Alpha (valeurs > 0)
    alpha_channel = img[:, :, 3]
    
    # Trouver toutes les coordonnées (y, x) où la forme est présente
    coords = np.argwhere(alpha_channel > 0)
    
    if len(coords) == 0:
        return []  # Retourne une liste vide si l'image est entièrement transparente
        
    # Calcul du centre de masse de la forme (axe X)
    # L'utilisation de cv2.moments correspond à la fonction get_center_of_mass de ton code
    M = cv2.moments(alpha_channel)
    if M["m00"] != 0:
        center_x = M["m10"] / M["m00"]
    else:
        # Solution de repli si le moment d'ordre 0 est nul : moyenne géométrique des X
        center_x = np.mean(coords[:, 1])
        
    # Filtrer les coordonnées selon leur position par rapport à center_x
    if left:
        filtered_coords = coords[coords[:, 1] < center_x]
    else:
        filtered_coords = coords[coords[:, 1] > center_x]
        
    # Convertir l'array NumPy en une liste de listes native Python
    return filtered_coords.tolist()


def find_max_in_list_of_lists(lst):
    """
    Trouve la valeur maximale dans une liste de listes.
    
    Args:
        lst (list): Liste de listes contenant des valeurs numériques.
        
    Returns:
        float: La valeur maximale trouvée dans la liste de listes.
    """
    max_value = float('-inf')
    for sublist in lst:
        for value in sublist:
            if value > max_value:
                max_value = value
    return max_value


def downscale_image(image_data: dict, factor: int) -> dict:
    """Réduit la taille de l'image d'un facteur donné."""
    if factor <= 1:
        return image_data
    
    orig_h, orig_w = image_data["h"], image_data["w"]
    new_h = orig_h // factor
    new_w = orig_w // factor
    
    # Utilisation de cv2.INTER_AREA, idéal pour la réduction (downsampling) d'images
    resized_arr = cv2.resize(image_data["raw"], (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    print(f"  ↳ Image réduite (facteur {factor}) : {orig_h} × {orig_w} px → {new_h} × {new_w} px")
    return {**image_data, "raw": resized_arr, "h": new_h, "w": new_w}

def run_pipeline(paths: list[str], top_n_list: list[int]) -> list[dict]:
    """
    Pipeline complet.
    Avec SHOW_STEPS=True, chaque étape intermédiaire ouvre sa propre fenêtre
    Napari (à fermer pour continuer). Utile pour déboguer les dimensions.
    """
    TOTAL = 6 if SHOW_STEPS else 1

    # ── Étape 1 : Chargement brut + redimensionnement ────────────────────────
    print("\n" + "═" * 60)
    print("  Étape 1 — Chargement brut")
    print("═" * 60)
    results = [load_image(p) for p in paths]

    #TODO
    # Afficher images pour placement points en full res :
    results_full = results
    
    # Réduction de la taille de l'image selon le facteur défini
    if DOWNSCALE_FACTOR > 1:
        for i in range(len(results)):
            results[i] = downscale_image(results[i], DOWNSCALE_FACTOR)

    #TODO dimmensions de l'image downscale
    nifti_gauche = nib.load(NIFTI_PATH_GAUCHE)
    nifti_droite = nib.load(NIFTI_PATH_DROITE)
    print("Dimensions de NIfTI gauche :", nifti_gauche.shape)
    print("Dimensions de NIfTI droite :", nifti_droite.shape)

    # Apply vertical flip based on FLIP list
    for i in range(len(results)):
        if FLIP[i] == 1:
            results[i] = flip_image_vertically(results[i])
            results_full[i] = flip_image_vertically(results_full[i])
            print(f"  ↳ Image {i + 1} flipped vertically")
    
    """
    results = resize_to_reference_width(results)
    results_full = resize_to_reference_width(results_full)
    """

    if SHOW_STEPS:
        viewer = _open_viewer("Images brutes", 1, TOTAL)
        show_raw_images(viewer, results)
        napari.run()

    # ── Étape 2 : Suppression d'arrière-plan ─────────────────────────────────
    print("\n" + "═" * 60)
    print("  Étape 2 — Suppression d'arrière-plan")
    print("═" * 60)
    for i in range(len(results)):
        print(f"\n  Image {i + 1} :")
        results[i] = remove_background(results[i], top_n=top_n_list[i])
        
    if SHOW_STEPS:
        viewer = _open_viewer("Masques + cadres", 2, TOTAL)
        show_masked_images(viewer, results)
        napari.run() 

    # ── Étape 4 : Matrices carrées diagonales ────────────────────────────────
    print("\n" + "═" * 60)
    print("  Étape 4 — Matrices carrées diagonales")
    print("═" * 60)
    for i in range(len(results)):
        print(f"\n  Image {i + 1} :")
        results[i] = pad_to_diagonal_square(results[i])

    if SHOW_STEPS:
        viewer = _open_viewer("Matrices carrées", 5, TOTAL)
        show_square_matrices(viewer, results)
        napari.run()

    # ── Étape 6 : Orientation face à face ────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Étape 6 — Orientation et alignement face à face")
    print("═" * 60)
    sides = ["left_piece", "right_piece"]
    for i in range(len(results)):
        print(f"\n  Image {i + 1} :")
        results[i] = orient_and_face(results[i], side=sides[i])

    viewer = _open_viewer("Alignement Face à Face", TOTAL, TOTAL)
    space = results[0]["diagonal"]
    show_facing_images(viewer, results, ecart=space//10)
    napari.run()

    print("\n" + "═" * 60)
    print("  Pipeline terminé.")
    print("═" * 60 + "\n")

    # Récupérer les données
    img_gauche = results[0]
    img_droite = results[1]

    if "ligne_mediane" in img_gauche:
        ligne_med = img_gauche["ligne_mediane"]
        trans_d = img_droite["translation_droite"]
        seg_g = img_gauche["segment_coupe"]
        seg_d = img_droite["segment_coupe"]

        print(f"  ↳ Ligne médiane générée : {len(ligne_med)} points.")
        print(f"  ↳ Translation du morceau droit : Y = {trans_d[0]:.2f}, X = {trans_d[1]:.2f}")
        print(f"  ↳ Segment gauche : {len(seg_g)} points | Segment droit : {len(seg_d)} points")

    return results

if __name__ == "__main__":
    r = run_pipeline(PATHS, top_n_list=TOP_N_LIST)


img_gauche = r[0]
print(img_gauche.keys())

img_droite = r[1]
print(img_droite.keys())

if "ligne_mediane" not in img_gauche or "segment_coupe" not in img_gauche:
    raise RuntimeError(
        "Appuie sur 'f' dans la fenêtre Napari avant de la fermer "
        "pour générer la ligne médiane et les segments de coupe."
    )

seg_g = img_gauche["segment_coupe"]
seg_d = img_droite["segment_coupe"]
p1g = img_gauche["p1g"]
p2g = img_gauche["p2g"]

# Axe Y converti en entier et filtré
seg_g2 = filtrer_doublons_y(seg_g, select_max=True)
seg_d2 = filtrer_doublons_y(seg_d, select_max=False)

print(f"Segment gauche : {len(seg_g2)} points | Segment droit : {len(seg_d2)} points")

y_g = seg_g2[:, 0]
x_g = seg_g2[:, 1]

y_d = seg_d2[:, 0]
x_d = seg_d2[:, 1]

# Préparation des images
img_g = img_gauche["rotated"]
img_d_raw = img_droite["rotated"]
trans_d = img_droite["translation_droite"]
translation_y, translation_x = trans_d[0], trans_d[1]

new_width_d = img_d_raw.shape[1] + int(max(0, translation_x))
new_height_d = img_d_raw.shape[0] + int(max(0, translation_y))
img_d = cv2.warpAffine(img_d_raw, np.float32([[1, 0, translation_x], [0, 1, translation_y]]), (new_width_d, new_height_d))

height_g, width_g = img_g.shape[:2]
height_d, width_d = img_d.shape[:2]

y_all_g, y_all_d = np.arange(height_g), np.arange(height_d)
center_g, center_d = get_center_of_mass(img_g), get_center_of_mass(img_d)

class FusionApp:
    def __init__(self):
        self.pts_g = None
        self.pts_d = None
        self.viewer = napari.Viewer(title="Fusion Strict 2D (Zéro superposition)")
        self.viewer.add_image(img_g, name="Morceau Gauche", blending="additive")
        self.viewer.add_image(img_d, name="Morceau Droit", blending="additive")
        #self.viewer.add_points(seg_g2, name="Segment Gauche", face_color="green", size=1)
        #self.viewer.add_points(seg_d2, name="Segment Droit", face_color="blue", size=1)
        #self.viewer.add_points(np.vstack((p1g, p2g)), name="Points Extremes", face_color="magenta", size=2, symbol="cross")
        self.viewer.add_points(np.empty((0, 2)), name="Repères Gauche", face_color="red", size=10, symbol="cross")
        self.viewer.add_points(np.empty((0, 2)), name="Repères Droite", face_color="blue", size=10, symbol="cross")
        print("\n" + "═" * 60)
        print("  Mode d'alignement SANS superposition")
        print("═" * 60)
        print("  1. Cliquez des points ROUGES sur le GAUCHE (proches de la coupe)")
        print("  2. Cliquez des points BLEUS sur le DROIT (même ordre)")
        print("  3. Appuyez sur 'e' pour fermer le tissu 2D exactement sur la ligne médiane")
        print("═" * 60)
        self.viewer.bind_key('e', self.apply_fusion)

    def apply_fusion(self, v):
        self.pts_g = v.layers["Repères Gauche"].data
        self.pts_d = v.layers["Repères Droite"].data

app = FusionApp()
napari.run()

pts_g = app.pts_g
pts_d = app.pts_d

pts_g_seg = snap_au_contour_vectorise(pts_g, seg_g2)
pts_d_seg = snap_au_contour_vectorise(pts_d, seg_d2)

if SHOW_STEPS:
    print("\n"+"--"*30)
    print("Points placés à gauche :")
    print(pts_g)
    print(f"Nombre de points : {len(pts_g)}")
    print("--"*30)
    print("Points placés à droite :")
    print(pts_d)
    print(f"Nombre de points : {len(pts_d)}")

if len(pts_g) != len(pts_d):
    print("Erreur : nombre de points placés différents.")
    print("--"*30)

if SHOW_STEPS:
    viewer = napari.Viewer(title="Affichage des points")
    viewer.add_points(seg_g2, name="Segment Gauche", face_color="green", size=1)
    viewer.add_points(seg_d2, name="Segment Droit", face_color="blue", size=1)
    viewer.add_points(np.vstack((p1g, p2g)), name="Points Extremes", face_color="magenta", size=1)
    viewer.add_points(pts_g, name="Repères Gauche", face_color="red", size=1)
    viewer.add_points(pts_d, name="Repères Droite", face_color="yellow", size=1)
    viewer.add_points(pts_g_seg, name="Repères gauche placés", face_color="red", size=1)
    viewer.add_points(pts_d_seg, name="Repères droite placés", face_color="yellow", size=1)
    napari.run()

extremes = np.vstack((p1g, p2g))
# Conserver uniquement la partie entière sur la première colonne (axe y)
extremes[:, 0] = np.trunc(extremes[:, 0])

if SHOW_STEPS:
    print("Extremes :")
    print(extremes)
    print("--"*30)

# Ajout des points extremes à la liste de points de référence
pts_g_seg = np.vstack((pts_g_seg, extremes))
pts_d_seg = np.vstack((pts_d_seg, extremes))

# Tri par ordre croissant suivant y.
pts_g_seg = cs_sort(pts_g_seg)
pts_d_seg = cs_sort(pts_d_seg)


########################################
########################################
if SHOW_STEPS:
    for ele in pts_g_seg:
        if ele in seg_g2:
            print(ele)
        else:
            print("Erreur")
    print(f"Premier : {seg_g2[0]}")
    print(f"Dernier : {seg_g2[-1]}")

    print("--"*30)
    for ele in pts_d_seg:
        if ele in seg_d2:
            print(ele)
        else:
            print("Erreur")
    print(f"Premier : {seg_d2[0]}")
    print(f"Dernier : {seg_d2[-1]}")
########################################
########################################


part_g = segmenter_ligne(seg_g2, pts_g_seg)
part_d = segmenter_ligne(seg_d2, pts_d_seg)


print("--"*30)
print("Nombre de segments :")
print(f"Gauche : {len(part_g)}")
print(f"Droite : {len(part_d)}")

if SHOW_STEPS:
    viewer = napari.Viewer(title="Affichage des segments")
    for x in range(len(part_g)):
        viewer.add_points(part_g[x], name=f"Segment Gauche {x+1}", face_color=color_liste[x], size=1)
    for x in range(len(part_d)):
        viewer.add_points(part_d[x], name=f"Segment Droit {x+1}", face_color=color_liste[x], size=1)
    viewer.add_points(pts_g_seg, name="Repères Gauche", face_color="white", size=3)
    viewer.add_points(pts_d_seg, name="Repères Droite", face_color="white", size=3)
    napari.run()
print("--"*30)

# part_g = liste des coordonnées par petits segments de gauche


g_dest = []
d_dest = []

# Boucle pour calcul de position cible
for x in range(len(part_g)):
    g_inter_s = int(pts_g_seg[x+1][0] - pts_g_seg[x][0])
    d_inter_s = int(pts_d_seg[x+1][0] - pts_d_seg[x][0])

    if SHOW_STEPS:
        print("Intermédiaires :")
        print(g_inter_s)
        print(d_inter_s)
        print("--"*5)

    ratio_g = d_inter_s/g_inter_s
    ratio_d = g_inter_s/d_inter_s

    delta_base_g = int(pts_d_seg[x][0] - pts_d_seg[0][0])
    delta_base_d = int(pts_g_seg[x][0] - pts_g_seg[0][0])

    g_dest.append([int(pts_d_seg[x][0]-pts_d_seg[0][0]), int(pts_d_seg[x][0]-pts_d_seg[0][0]), 0])
    d_dest.append([int(pts_g_seg[x][0]-pts_g_seg[0][0]), int(pts_g_seg[x][0]-pts_g_seg[0][0]), 0])

    for v in range(g_inter_s - 1):
        k = v*ratio_g + delta_base_g
        und, up, dec = frac_n(k)

        g_dest.append([und, up, dec])
    ######################################
    for s in range(d_inter_s-1):
        k = s*ratio_d + delta_base_d
        und, up, dec = frac_n(k)

        d_dest.append([und, up, dec])

g_dest.append([int(pts_d_seg[-1][0]-pts_d_seg[0][0]), int(pts_d_seg[-1][0]-pts_d_seg[0][0]), 0])
d_dest.append([int(pts_g_seg[-1][0]-pts_g_seg[0][0]), int(pts_g_seg[-1][0]-pts_g_seg[0][0]), 0])

if len(seg_d2) < len(seg_g2):
    for i in range(len(seg_g2) - len(seg_d2)):
        seg_d2 = np.vstack((seg_d2,seg_d2[-1]))

if len(seg_g2) < len(seg_d2):
    for i in range(len(seg_d2) - len(seg_g2)):
        seg_g2 = np.vstack((seg_g2,seg_g2[-1]))

if SHOW_STEPS:
    print(f"longueur g_dest : {len(g_dest)} | Max : {find_max_in_list_of_lists(g_dest)}")
    print(f"longueur d_dest : {len(d_dest)} | Max : {find_max_in_list_of_lists(d_dest)}")
    print(f"longueur seg_g2 : {len(seg_g2)}")
    print(f"longueur seg_d2 : {len(seg_d2)}")
    print("\n")
    for i in range(10):
        print(f"g_dest[{i}] : {g_dest[i]}")
        print(f"d_dest[{i}] : {d_dest[i]}")
        print("-")

    print("--"*30)


if SHOW_STEPS:
    print("Segments de part_d :")
    for els in part_d:
        print(len(els))
    print("--"*30)



g_base = []
g_displace = []
#g_base_len = len(g_base)

d_base = []
d_displace = []
#d_base_len = len(d_base)

g_mov = []

for f in range(len(g_dest)):
    ng1 = int(g_dest[f][0])
    ng2 = int(g_dest[f][1])
    t = g_dest[f][2]
    if SHOW_STEPS:
        print(f"ng1 : {ng1}")
        print(f"ng2 : {ng2}")
        #print("--"*5)

    pixel_a = arr2tup(seg_d2[ng1])
    try :
        pixel_b = arr2tup(seg_d2[ng2])
    except IndexError:
        print(f"IndexError: ng2 = {ng2}, Longueur de seg_d2 = {len(seg_d2)}, Longueur de seg_g2 = {len(seg_g2)}, Longueur de g_dest = {len(g_dest)}")
        raise SystemExit("Arrêt du programme en raison d'une erreur d'index.")

    res = lerp_coordinates(pixel_a, pixel_b, t)
    try :
        coords_rel = (res + seg_g2[f])/2
    except IndexError:
        print(f"IndexError: f = {f}, Longueur de seg_g2 = {len(seg_g2)}, Longueur de g_dest = {len(g_dest)}")
        raise SystemExit("Arrêt du programme en raison d'une erreur d'index.")

    g_displace.append([coords_rel[0], coords_rel[1]])
    
    g_base.append(seg_g2[f])
    #print(f"g_displace[{f}] : {g_displace[f]}")
    #print(f"g_base[{f}] : {g_base[f]}")
    #print("##########")
    g_mov.append(g_displace[f][1]-g_base[f][1])

for ele in g_displace:
    ele[1] = ele[1] - max(g_mov)

# Trouver celui qui a le plus gros déplacement dans l'axe x (valeur de combien)
print("g_mov")
print("Longueur : ", len(g_mov))
print("Min : ", min(g_mov))
print("Max : ",max(g_mov))    

d_mov = []

for f in range(len(d_dest)):
    nd1 = int(d_dest[f][0])
    nd2 = int(d_dest[f][1])
    t = d_dest[f][2]
    if SHOW_STEPS:
        print(f"nd1 : {nd1}")
        print(f"nd2 : {nd2}")
        print(f"t : {t}")
        print("--"*5)
    
    pixel_a = arr2tup(seg_g2[nd1])
    pixel_b = arr2tup(seg_g2[nd2])

    res = lerp_coordinates(pixel_a, pixel_b, t)
    coords_rel = (res + seg_d2[f])/2
    d_displace.append([coords_rel[0], coords_rel[1]])

    d_base.append(seg_d2[f])
    d_mov.append(d_displace[f][1]-d_base[f][1])

for ele in d_displace:
    ele[1] = ele[1] - min(d_mov)

print("d_mov")
print("Longueur : ", len(d_mov))
print("Min : ", min(d_mov))
print("Max : ",max(d_mov))

g_base.extend(get_center_column_coordinates(img_g, left=True))
g_displace.extend(get_center_column_coordinates(img_g, left=True))
d_base.extend(get_center_column_coordinates(img_d, left=False))
d_displace.extend(get_center_column_coordinates(img_d, left=False))


print(f"Nombre de points dans g_base : {len(g_base)}")
print(f"Longueur de g_displace : {len(g_displace)}")
print("--"*30)




# 1. Normaliser les images en float32 [0, 1]
img_g_float = img_g.astype(np.float32) / 255.0
img_d_float = img_d.astype(np.float32) / 255.0

# 2. Convertir les points [y, x] → [x, y] pour le solveur TPS
anchors_g = get_anchor_points(img_g, left=True, n_points=100)
src_g = np.vstack([np.array(g_base,     dtype=np.float64), anchors_g])
#src_g = np.array(g_base, dtype=np.float64)
dst_g = np.vstack([np.array(g_displace, dtype=np.float64), anchors_g])
#dst_g = np.array(g_displace, dtype=np.float64)

anchors_d = get_anchor_points(img_d, left=False, n_points=100)
src_d = np.vstack([np.array(d_base,     dtype=np.float64), anchors_d])
#src_d = np.array(d_base, dtype=np.float64)
dst_d = np.vstack([np.array(d_displace, dtype=np.float64), anchors_d])
#dst_d = np.array(d_displace, dtype=np.float64)

if SHOW_STEPS:
    viewer = napari.Viewer(title="Points de base")
    viewer.add_points(src_g, name="base gauche", face_color="white", size=1)
    viewer.add_points(dst_g, name=f"g_displace", face_color=color_liste[9], size=1)
    viewer.add_points(src_d, name="base droite", face_color="purple", size=1)
    viewer.add_points(dst_d, name=f"d_displace", face_color=color_liste[11], size=1)
    if SHOW_STEPS:
        for x in range(len(part_g)):
            viewer.add_points(part_g[x], name=f"Segment Gauche {x+1}", face_color=color_liste[x], size=1)
        for x in range(len(part_d)):
            viewer.add_points(part_d[x], name=f"Segment Droit {x+1}", face_color=color_liste[x], size=1)
    napari.run()



# 3. Appliquer le warp (déclenche les affichages Matplotlib successifs)
print("\n" + "═" * 60)
print("  Application du TPS - Morceau Gauche")
print("═" * 60)
print(img_g_float.shape)
if GPU:
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()
warped_g, disp_y_g, disp_x_g = auto_warp_image(img_g_float, src_g, dst_g)


print("\n" + "═" * 60)
print("  Application du TPS - Morceau Droit")
print("═" * 60)
print(img_d_float.shape)
if GPU:
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()
warped_d, disp_y_d, disp_x_d = auto_warp_image(img_d_float, src_d, dst_d)




# 4. Affichage combiné final dans Napari
print("\n" + "═" * 60)
print("  Affichage du résultat combiné")
print("═" * 60)

viewer_final = napari.Viewer(title="Résultat Final Combiné (TPS)")

# Utilisation du mode "translucent" pour respecter la transparence (canal alpha)
viewer_final.add_image(warped_g, name="TPS Gauche", blending="translucent")
viewer_final.add_image(warped_d, name="TPS Droite", blending="translucent", translate=(0, -(max(g_mov) - min(d_mov))))

viewer_final.add_points(src_g, name="base gauche", face_color="white", size=1)
viewer_final.add_points(dst_g, name=f"g_displace", face_color=color_liste[9], size=1)
viewer_final.add_points(src_d, name="base droite", face_color="purple", size=1)
viewer_final.add_points(dst_d, name=f"d_displace", face_color=color_liste[11], size=1)
napari.run()






# ─────────────────────────────────────────────────────────────
#  EXPORT DES CARTES DE DÉPLACEMENT
# ─────────────────────────────────────────────────────────────
print("\n" + "═" * 60)
print("  Exportation des cartes de déplacement (format .npy)")
print("═" * 60)

now = time.time()
os.makedirs(DISP_FOLDER, exist_ok=True)
np.save(os.path.join(DISP_FOLDER, f"disp_x_gauche_{now}.npy"), disp_x_g)
np.save(os.path.join(DISP_FOLDER, f"disp_y_gauche_{now}.npy"), disp_y_g)
np.save(os.path.join(DISP_FOLDER, f"disp_x_droite_{now}.npy"), disp_x_d)
np.save(os.path.join(DISP_FOLDER, f"disp_y_droite_{now}.npy"), disp_y_d)
print("✓  Fichiers exportés avec succès.")



# ═════════════════════════════════════════════════════════════════════════════
#  ÉTAPE 1 : RECOMPOSITION ET EXPORT DU TIFF FINAL
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  Recomposition du TIFF final")
print("═" * 60)

def combine_tiff_rgba(img_g, img_d, shift_x):
    """Recombine les deux images avec la même logique de superposition Alpha que Napari."""
    h_g, w_g = img_g.shape[:2]
    h_d, w_d = img_d.shape[:2]
    
    min_x = min(0, shift_x)
    max_x = max(w_g, shift_x + w_d)
    
    out_h = max(h_g, h_d)
    out_w = int(max_x - min_x)
    
    # Canevas initialisé à 0 (transparence totale)
    canvas = np.zeros((out_h, out_w, 4), dtype=img_g.dtype)
    
    # 1. Pose du morceau gauche
    xg1 = int(0 - min_x)
    canvas[:h_g, xg1:xg1+w_g] = img_g
    
    # 2. Composition Alpha pour le morceau droit (comme dans Napari)
    xd1 = int(shift_x - min_x)
    
    # Normalisation des alphas entre 0.0 et 1.0 si nécessaire
    src_alpha = img_d[:, :, 3]
    if src_alpha.max() > 1.0:
        src_alpha = src_alpha / 255.0
    src_alpha = src_alpha[:, :, np.newaxis]  # Dimension (H, W, 1) pour le broadcast
    
    dst_target = canvas[:h_d, xd1:xd1+w_d]
    dst_alpha = dst_target[:, :, 3]
    if dst_alpha.max() > 1.0:
        dst_alpha = dst_alpha / 255.0
    dst_alpha = dst_alpha[:, :, np.newaxis]

    # Formule standard "Source Over" (Morceau droit par-dessus le morceau gauche)
    out_alpha = src_alpha + dst_alpha * (1.0 - src_alpha)
    
    # Évite les divisions par zéro dans les zones 100% transparentes
    mask = out_alpha > 0
    
    composite_rgb = np.zeros_like(dst_target[:, :, :3])
    composite_rgb[mask[:, :, 0]] = (
        img_d[:, :, :3][mask[:, :, 0]] * src_alpha[mask[:, :, 0]] + 
        dst_target[:, :, :3][mask[:, :, 0]] * dst_alpha[mask[:, :, 0]] * (1.0 - src_alpha[mask[:, :, 0]])
    ) / out_alpha[mask[:, :, 0]]
    
    # Mise à jour de la zone de superposition sur le canvas
    canvas[:h_d, xd1:xd1+w_d, :3] = composite_rgb
    canvas[:h_d, xd1:xd1+w_d, 3] = (out_alpha[:, :, 0] * (255 if img_g.max() > 1.0 else 1.0)).astype(img_g.dtype)
    
    return canvas

tiff_shift_x = int(-(max(g_mov) - min(d_mov)))
combined_tiff = combine_tiff_rgba(warped_g, warped_d, tiff_shift_x)

# Affichage dans Napari sans le fond opaque
viewer_final = napari.Viewer(title="Résultat Final Combiné (TPS)")
viewer_final.add_image(combined_tiff, blending="translucent")
napari.run()

if MAKE_TIFF:
    os.makedirs(os.path.dirname(TIFF_OUT_PATH), exist_ok=True)

    img_to_save = combined_tiff.copy()
    if img_to_save.max() <= 1.0:
        img_to_save = (img_to_save * 255).astype(np.uint8)

    # Re-conversion RGBA vers BGRA obligatoire pour cv2
    img_bgra = cv2.cvtColor(img_to_save, cv2.COLOR_RGBA2BGRA)

    # cv2 écrit en TIFF si l'extension est .tif ou .tiff
    cv2.imwrite(TIFF_OUT_PATH, img_bgra)

    print(f"  ✓ TIFF combiné sauvegardé avec succès : {TIFF_OUT_PATH}")











# ═════════════════════════════════════════════════════════════════════════════
#  ÉTAPE 2 : PIPELINE NIfTI
# ═════════════════════════════════════════════════════════════════════════════
# On récupère l'image NIfTI combinée (normalisée 0-1)
combined_nii = run_nii_remap_pipeline(
    nii_path_g=NIFTI_PATH_GAUCHE,
    nii_path_d=NIFTI_PATH_DROITE,
    tiff_g=r[0],
    tiff_d=r[1],
    out_path_combined=NIFTI_OUT_COMBINED,
    disp_folder=DISP_FOLDER,
    tiff_shift_x=tiff_shift_x
)







# ═════════════════════════════════════════════════════════════════════════════
#  ÉTAPE 3 : OVERLAY TIFF + NIfTI
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  Création de l'Overlay TIFF / NIfTI")
print("═" * 60)

# 1. Redimensionner le NIfTI pour qu'il matche exactement la taille du TIFF
h_t, w_t = combined_tiff.shape[:2]
nii_resized = cv2.resize(combined_nii, (w_t, h_t), interpolation=cv2.INTER_NEAREST)

# 2. Préparer les canaux de couleur
# Histologie en niveaux de gris
tiff_gray = cv2.cvtColor(combined_tiff, cv2.COLOR_RGB2GRAY)

# NIfTI en rouge vif
nii_rgb = np.zeros((h_t, w_t, 3), dtype=np.float32)
nii_rgb[:, :, 0] = nii_resized  # Canal Rouge

# 3. Fusionner avec transparence
# L'histologie est assombrie là où la tumeur est présente, et la tumeur devient rouge
alpha = 0.65  # Opacité de la carte tumorale
overlay = np.zeros_like(tiff_gray)
overlay = tiff_gray * (1 - nii_resized * alpha) + nii_rgb[:, :, 0] * (nii_resized * alpha)

# Reconvertir en RGB pour l'affichage et la sauvegarde
overlay_rgb = np.stack([overlay, tiff_gray * (1 - nii_resized * alpha), tiff_gray * (1 - nii_resized * alpha)], axis=-1)
overlay_rgb = np.clip(overlay_rgb, 0, 1)

# 4. Sauvegarde et affichage
cv2.imwrite(OVERLAY_OUT_PATH, (overlay_rgb[:, :, ::-1] * 255).astype(np.uint8))
print(f"  ✓  Overlay sauvegardé : {OVERLAY_OUT_PATH}")

plt.figure(figsize=(12, 10))
plt.imshow(overlay_rgb)
plt.title("Overlay : Histologie (Gris) + Carte Tumorale NIfTI (Rouge)")
plt.axis('off')
plt.tight_layout()
plt.show()

print("\n" + "═" * 60)
print("  Pipeline complet terminé avec succès !")
print("═" * 60)