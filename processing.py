"""
processing.py
─────────────────────────────────────────────────────────────────────────────
Fonctions de traitement d'image : chargement, masquage, recadrage, mise en
matrice carrée diagonale.

Chaque image est traitée une seule fois (masque calculé une seule fois)
et toutes les données intermédiaires sont retournées dans un dict,
ce qui facilite l'ajout de nouvelles étapes.
"""

import math
import numpy as np
import cv2
from PIL import Image

from tif_mask import mask as compute_mask
from mask_image import apply_mask


# ─────────────────────────────────────────────────────────────────────────────
# Chargement brut
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path: str) -> dict:
    """
    Charge une image TIF sans aucun pré-traitement.

    Retourne un dict avec :
        'path'  : chemin d'accès
        'raw'   : array numpy brut (H, W) ou (H, W, C)
        'h', 'w': dimensions originales
    """
    img = Image.open(path)
    raw = np.array(img)
    h, w = raw.shape[:2]
    print(f"  ↳ Chargé  : {path}")
    print(f"    Dimensions brutes : {h} × {w} px  (H × W)")
    return {"path": path, "raw": raw, "h": h, "w": w}


def flip_image_vertically(image_data: dict) -> dict:
    """
    Flip the image vertically (upside down).
    """
    flipped_raw = np.flipud(image_data["raw"])
    return {**image_data, "raw": flipped_raw, "is_flipped_v": True}


# ─────────────────────────────────────────────────────────────────────────────
# Suppression d'arrière-plan
# ─────────────────────────────────────────────────────────────────────────────

def remove_background(image_data: dict, top_n: int = 1) -> dict:
    """
    Calcule le masque et applique la transparence en respectant 
    le redimensionnement effectué à l'étape 1.
    """
    path = image_data["path"]
    new_h = image_data["h"]
    new_w = image_data["w"]

    # 1. Calcul du masque (tif_mask.mask utilise le path, il génère la taille originale)
    mask_arr = compute_mask(
        path,
        channel="L",
        threshold=None,
        plot=False,
        show_img=False,
        min_peak_distance=20,
        op_clo=False,
        gaussian_blur=True,
        top_n=top_n,
    )
    mask_arr = np.array(mask_arr, dtype=np.uint8)

    # 2. Redimensionnement du masque pour qu'il corresponde à l'étape 1
    if mask_arr.shape[:2] != (new_h, new_w):
        mask_pil = Image.fromarray(mask_arr)
        # NEAREST est important pour ne garder que des 0 et des 255 sur le masque binaire
        mask_arr = np.array(mask_pil.resize((new_w, new_h), Image.NEAREST))

    # Retournement du masque si l'image brute a été inversée
    if image_data.get("is_flipped_v", False):
        mask_arr = np.flipud(mask_arr)
        print("  ↳ Masque inversé verticalement pour correspondre à l'image")

    # 3. Création de l'image RGB à partir de l'image 'raw' DÉJÀ redimensionnée
    raw = image_data["raw"]
    if raw.ndim == 2:
        rgb = np.stack((raw,) * 3, axis=-1)
    elif raw.ndim == 3 and raw.shape[2] == 4:
        rgb = raw[..., :3]
    else:
        rgb = raw

    # 4. Application du masque → RGBA
    rgba = apply_mask(rgb, mask_arr)

    print(f"  ↳ Masque redimensionné : {mask_arr.shape}, top_n={top_n}")

    image_data = {**image_data, "mask": mask_arr, "rgba": rgba}
    return image_data

def pad_to_diagonal_square(image_data: dict) -> dict:
    """
    Place l'image RGBA dans une matrice carrée a × a où
        a = ceil( sqrt(W² + H²) )
    avec W et H les dimensions de l'image RGBA.
    L'image est centrée ; le reste est transparent (alpha = 0).

    Nécessite image_data["rgba"] et image_data["mask"].

    Ajoute au dict :
        'square'      : array RGBA carré (a, a, 4)
        'square_mask' : masque carré (a, a)
        'diagonal'    : valeur entière a
        'bbox'        : (0, h-1, 0, w-1) pour simuler un recadrage sur l'image entière
    """
    # On prend directement l'image RGBA et le masque non recadrés
    img = image_data["rgba"]
    mask = image_data["mask"]
    h_src, w_src = img.shape[:2]

    a = math.ceil(math.sqrt(w_src ** 2 + h_src ** 2))

    square = np.zeros((a, a, 4), dtype=np.uint8)
    square_mask = np.zeros((a, a), dtype=np.uint8)

    row_off = (a - h_src) // 2
    col_off = (a - w_src) // 2

    square[row_off : row_off + h_src, col_off : col_off + w_src] = img
    square_mask[row_off : row_off + h_src, col_off : col_off + w_src] = mask

    print(f"  ↳ Carré   : {a} × {a} px"
          f"  (a = ceil(√({w_src}²+{h_src}²)) = {a})"
          f"  [image : {h_src} × {w_src}]")

    # On fournit un 'bbox' couvrant toute l'image pour que le pipeline NIfTI 
    # s'aligne sur l'image non recadrée.
    bbox = (0, h_src - 1, 0, w_src - 1)

    image_data = {**image_data, 
                  "square": square, 
                  "square_mask": square_mask, 
                  "diagonal": a,
                  "bbox": bbox}
    return image_data

# ─────────────────────────────────────────────────────────────────────────────
# Détection de côté de prostate
# ─────────────────────────────────────────────────────────────────────────────

def orient_and_face(image_data: dict, side: str) -> dict:
    """
    Détecte le côté de coupe avec Douglas-Peucker et tourne la matrice carrée
    pour que la coupe soit verticale et face au centre.
    """
    import cv2
    mask = image_data["square_mask"]
    square_img = image_data["square"]

    # 1. Extraction des contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("Aucun contour trouvé dans le masque carré.")
    
    dots = np.vstack([c.reshape(-1, 2) for c in contours])
    
    # 2. Douglas-Peucker pour trouver la plus longue droite
    contour_cv2 = dots.reshape((-1, 1, 2)).astype(np.int32)
    perimetre = cv2.arcLength(contour_cv2, True)
    epsilon = 0.03 * perimetre
    approx = cv2.approxPolyDP(contour_cv2, epsilon, True)

    longueur_max = 0
    p1_xy, p2_xy = None, None
    n_points = len(approx)
    
    for i in range(n_points):
        pt1 = approx[i][0]
        pt2 = approx[(i + 1) % n_points][0]
        dist = np.linalg.norm(pt1 - pt2)
        if dist > longueur_max:
            longueur_max = dist
            p1_xy, p2_xy = pt1, pt2

    # Passage en format (y,x) pour correspondre exactement à ta logique d'origine
    p1_yx = np.array([p1_xy[1], p1_xy[0]])
    p2_yx = np.array([p2_xy[1], p2_xy[0]])

    # 3. Calcul de l'angle et détermination du haut/bas
    den = (p1_yx[1] - p2_yx[1])
    if den == 0: den = 1e-5
    angle = np.degrees(np.arctan(((-p1_yx[0]) - (-p2_yx[0])) / den))

    mid_yx = (p1_yx + p2_yx) // 2
    M_mom = cv2.moments(mask)
    cY = int(M_mom["m01"] / M_mom["m00"]) if M_mom["m00"] != 0 else mask.shape[0]//2
    
    up = (cY < mid_yx[0])

    # 4. Déduction de la rotation selon le rôle du morceau
    if side == "left_piece":
        rotation = (-angle + 90) if up else (-angle - 90)
    else: # right_piece
        rotation = (-angle - 90) if up else (-angle + 90)

    # 5. Application de la rotation via OpenCV
    center = (float(mask.shape[1] / 2), float(mask.shape[0] / 2))
    M_rot = cv2.getRotationMatrix2D(center, float(rotation), 1.0)
    
    rotated_img = cv2.warpAffine(square_img, M_rot, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_LINEAR)

    # 6. Suivi du point médian tourné pour le recalage final
    mid_arr_xy = np.array([[[mid_yx[1], mid_yx[0]]]], dtype=np.float32)
    mid_rot_xy = cv2.transform(mid_arr_xy, M_rot)[0][0]
    turn_mid_yx = np.array([mid_rot_xy[1], mid_rot_xy[0]])

    # 7. Extraire le contour de la version tournée pour le magnétisme
    rotated_mask = cv2.warpAffine(mask, M_rot, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST)
    contours_rot, _ = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rotated_contour_xy = np.vstack([c.reshape(-1, 2) for c in contours_rot])
    rotated_contour_yx = np.flip(rotated_contour_xy, axis=1)

    print(f"  ↳ Angle de coupe détecté : {angle:.1f}° | Rotation appliquée : {rotation:.1f}°")

    return {
        **image_data, 
        "rotated": rotated_img, 
        "turn_mid_yx": turn_mid_yx,
        "rotation_angle": rotation,            # Nécessaire pour la rotation finale
        "rotated_contour_yx": rotated_contour_yx # Nécessaire pour le snap
    }


# ─────────────────────────────────────────────────────────────────────────────
# Filtre des doublons, pour garder uniquement le max ou le min
# ─────────────────────────────────────────────────────────────────────────────

def filtrer_doublons_y(segment, select_max=True):
    """
    Filtre les doublons de Y dans un segment.
    - Pour seg_g : conserve le X maximum pour chaque Y.
    - Pour seg_d : conserve le X minimum pour chaque Y.
    """
    filtre = {}
    for y, x in segment:
        y_int = int(y)  # Convertir Y en entier
        if y_int not in filtre:
            filtre[y_int] = x
        else:
            if select_max and x > filtre[y_int]:
                filtre[y_int] = x
            elif not select_max and x < filtre[y_int]:
                filtre[y_int] = x
    # Retourner un tableau trié par Y
    return np.array(sorted(filtre.items()))


# ─────────────────────────────────────────────────────────────────────────────
# Centre de masse
# ─────────────────────────────────────────────────────────────────────────────

def get_center_of_mass(img):
    """
    Calcule le centre de masse (centre de gravité) d'une image à partir de son canal alpha ou en niveaux de gris.

    Retourne un tuple (y, x) représentant les coordonnées du centre de masse.
    """
    if isinstance(img, dict) and "mask" in img:
        mask = img["mask"]
    elif img.ndim == 3 and img.shape[2] == 4:
        mask = img[:, :, 3]
    else:
        mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        
    moments = cv2.moments(mask)

    if moments["m00"] != 0:
        cY = int(moments["m01"] / moments["m00"])
        cX = int(moments["m10"] / moments["m00"])
    else:
        # Si le masque est vide, retourner le centre de l'image
        h, w = mask.shape[:2]
        cY, cX = h // 2, w // 2

    return (cY, cX)