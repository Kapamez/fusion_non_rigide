"""
napari_display.py
─────────────────────────────────────────────────────────────────────────────
Fonctions d'affichage Napari.
Chaque fonction reçoit un viewer et une liste de dicts (sortie de processing.py).

Conventions de couleur des cadres :
    Image 1 → cyan
    Image 2 → yellow
    Image 3 → magenta
"""

import numpy as np
import napari
import cv2
import traceback


# ─────────────────────────────────────────────────────────────────────────────
# Constantes d'affichage
# ─────────────────────────────────────────────────────────────────────────────

BORDER_COLORS = ["cyan", "yellow", "magenta", "green", "red"]
OFFSET = 8000   # Décalage horizontal entre images (pixels)


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires cadres
# ─────────────────────────────────────────────────────────────────────────────

def _is_rgb(arr: np.ndarray) -> bool:
    return arr.ndim == 3 and arr.shape[2] in (3, 4)

def _border_rect(h: int, w: int, offset: int = 0) -> np.ndarray:
    return np.array(
        [[0, offset], [0, offset + w - 1],
         [h - 1, offset + w - 1], [h - 1, offset]],
        dtype=float,
    )

def _add_border(viewer: napari.Viewer, h: int, w: int,
                color: str, name: str, edge_width: int = 12, offset: int = 0):
    rect = _border_rect(h, w, offset=offset)
    viewer.add_shapes(
        [rect],
        shape_type="polygon",
        edge_color=color,
        face_color="transparent",
        edge_width=edge_width,
        name=name,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Étape 1 — Images brutes
# ─────────────────────────────────────────────────────────────────────────────

def show_raw_images(viewer: napari.Viewer, images: list[dict]):
    print("\n── Napari : affichage brut ──")
    for i, data in enumerate(images):
        label = f"Brut {i + 1}"
        color = BORDER_COLORS[i % len(BORDER_COLORS)]
        viewer.add_image(data["raw"], name=label, rgb=_is_rgb(data["raw"]),
                         translate=[0, i * OFFSET])
        _add_border(viewer, data["h"], data["w"],
                    color=color, name=f"Cadre {label}", offset=i * OFFSET)
        print(f"  [{label}]  {data['h']} × {data['w']} px  ({data['path']})")


# ─────────────────────────────────────────────────────────────────────────────
# Étape 2 — Images masquées + cadre de la matrice d'origine
# ─────────────────────────────────────────────────────────────────────────────

def show_masked_images(viewer: napari.Viewer, images: list[dict]):
    print("\n── Napari : affichage masqué + cadres ──")
    for i, data in enumerate(images):
        color = BORDER_COLORS[i % len(BORDER_COLORS)]
        label = f"Masqué {i + 1}"
        viewer.add_image(data["rgba"], name=label, rgb=True,
                         translate=[0, i * OFFSET])
        _add_border(viewer, data["h"], data["w"],
                    color=color, name=f"Cadre {label}", offset=i * OFFSET)
        print(f"  [{label}]  cadre {data['h']} × {data['w']} px  (couleur : {color})")


# ─────────────────────────────────────────────────────────────────────────────
# Étape 4 — Matrices carrées diagonales
# ─────────────────────────────────────────────────────────────────────────────

def show_square_matrices(viewer: napari.Viewer, images: list[dict]):
    print("\n── Napari : affichage matrices carrées ──")
    for i, data in enumerate(images):
        a = data["diagonal"]
        color = BORDER_COLORS[i % len(BORDER_COLORS)]
        label = f"Carré {i + 1}"
        viewer.add_image(data["square"], name=label, rgb=True,
                         translate=[0, i * OFFSET])
        _add_border(viewer, a, a,
                    color=color, name=f"Cadre {label}", offset=i * OFFSET)
        print(f"  [{label}]  {a} × {a} px  "
              f"(original : {data['h']} × {data['w']} px)")


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires recalage
# ─────────────────────────────────────────────────────────────────────────────

def angle_de_correction(p1, p2, limite_degres=20):
    dy = p2[0] - p1[0]
    dx = p2[1] - p1[1]
    angle_actuel = np.degrees(np.arctan2(dy, dx))
    if angle_actuel < 0:
        angle_actuel += 180
    correction = 90 - angle_actuel
    if correction > 90:
        correction -= 180
    elif correction < -90:
        correction += 180
    correction = -correction
    if abs(correction) > limite_degres:
        print(f"⚠️ Alerte : Correction de {correction:.1f}° annulée (dépasse {limite_degres}°).")
        return 0.0
    return correction


def get_contour_segment(contour, p1, p2):
    idx1 = np.argmin(np.linalg.norm(contour - p1, axis=1))
    idx2 = np.argmin(np.linalg.norm(contour - p2, axis=1))
    i_min, i_max = min(idx1, idx2), max(idx1, idx2)
    path1 = contour[i_min:i_max+1]
    path2 = np.vstack((contour[i_max:], contour[:i_min+1]))
    len1 = np.sum(np.linalg.norm(np.diff(path1, axis=0), axis=1))
    len2 = np.sum(np.linalg.norm(np.diff(path2, axis=0), axis=1))
    return path1 if len1 < len2 else path2


def update_point(pt_yx, angle, center):
    center_float = (float(center[0]), float(center[1]))
    M = cv2.getRotationMatrix2D(center_float, float(angle), 1.0)
    pt_xy = np.array([[[pt_yx[1], pt_yx[0]]]], dtype=np.float32)
    new_xy = cv2.transform(pt_xy, M)[0, 0]
    return np.array([new_xy[1], new_xy[0]])


def transform_contour(contour_yx, angle, center, translation=np.array([0, 0])):
    center_float = (float(center[0]), float(center[1]))
    M = cv2.getRotationMatrix2D(center_float, float(angle), 1.0)
    contour_xy = np.flip(contour_yx, axis=1).reshape(-1, 1, 2).astype(np.float32)
    new_xy = cv2.transform(contour_xy, M).reshape(-1, 2)
    new_yx = np.flip(new_xy, axis=1)
    return new_yx + translation


def filtrer_flanc_coupe(segment, unique_axis=0, target_axis=1, select_max=True):
    filtre = {}
    for pt in segment:
        val_idx = int(round(pt[unique_axis]))
        val_target = pt[target_axis]
        if val_idx not in filtre:
            filtre[val_idx] = val_target
        else:
            if select_max and val_target > filtre[val_idx]:
                filtre[val_idx] = val_target
            elif not select_max and val_target < filtre[val_idx]:
                filtre[val_idx] = val_target
    return np.array(sorted([[val_idx if unique_axis == 0 else val_target, 
                             val_target if unique_axis == 0 else val_idx] 
                            for val_idx, val_target in filtre.items()]))


def calculer_ligne_mediane_filtree(seg_gauche, seg_droite):
    seg_g = filtrer_flanc_coupe(seg_gauche, unique_axis=0, target_axis=1, select_max=True)
    seg_d = filtrer_flanc_coupe(seg_droite, unique_axis=0, target_axis=1, select_max=False)
    if len(seg_g) == 0 or len(seg_d) == 0:
        return np.empty((0, 2))
    y_min = max(seg_g[0, 0], seg_d[0, 0])
    y_max = min(seg_g[-1, 0], seg_d[-1, 0])
    if y_min >= y_max:
        return np.empty((0, 2))
    y_commun = np.arange(y_min, y_max)
    x_g = np.interp(y_commun, seg_g[:, 0], seg_g[:, 1])
    x_d = np.interp(y_commun, seg_d[:, 0], seg_d[:, 1])
    x_moyen = (x_g + x_d) / 2.0
    return np.column_stack((y_commun, x_moyen))


def rotate_center(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), float(angle), 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)


def snap_au_contour(point_clique, liste_contours):
    distances = np.linalg.norm(liste_contours - point_clique, axis=1)
    return liste_contours[np.argmin(distances)]


# ─────────────────────────────────────────────────────────────────────────────
# Étape 5 — Affichage interactif et recalage manuel
# ─────────────────────────────────────────────────────────────────────────────

def show_facing_images(viewer: napari.Viewer, images: list[dict], ecart: int = 1000):
    """
    Affiche les images orientées face à face. L'utilisateur place directement 
    4 points rouges et appuie sur 'f' pour tout exécuter (magnétisme, rapprochement, 
    et ligne médiane).
    """

    img_g = images[0]
    img_d = images[1]

    # Initialisation des positions initiales
    mid_g = img_g["turn_mid_yx"]
    mid_d = img_d["turn_mid_yx"]
    y_translate = mid_g[0] - mid_d[0]
    x_translate = mid_g[1] - mid_d[1] + ecart

    viewer.add_image(img_g["rotated"], name="Morceau Gauche", blending='additive')
    viewer.add_image(img_d["rotated"], name="Morceau Droit", 
                     translate=[y_translate, x_translate], blending='additive')

    # Calque pour que l'utilisateur place ses 4 points rouges
    viewer.add_points(np.empty((0, 2)), name='Limites de coupe', size=15, face_color='red')

    contour_0 = img_g["rotated_contour_yx"]
    contour_1 = img_d["rotated_contour_yx"]

    print("\n── Instructions Napari ──")
    print("  1. Placez 4 points rouges (2 sur la coupe gauche, 2 sur la coupe droite).")
    print("  2. Appuyez directement sur 'f' pour lancer le rapprochement et la ligne médiane.")

    # Dictionnaire pour stocker les données entre les étapes (évite les erreurs d'attributs Napari)
    etat_pipeline = {}

    @viewer.bind_key('f')
    def rapprocher_morceaux(v):
        try:
            points_cliques = v.layers['Limites de coupe'].data

            print("--- MAGNÉTISME ET RAPPROCHEMENT AUTOMATIQUE ---")

            # 1. Tri des points cliqués selon l'axe X pour séparer gauche et droite
            points_tries = points_cliques[points_cliques[:, 1].argsort()]

            pts_gauche = points_tries[:2]
            pts_droite = points_tries[2:]

            # 2. Magnétisme effectué silencieusement en mémoire
            p1_g = snap_au_contour(pts_gauche[0], contour_0)
            p2_g = snap_au_contour(pts_gauche[1], contour_0)

            y_trans, x_trans = v.layers['Morceau Droit'].translate
            contour_1_global = contour_1 + np.array([y_trans, x_trans])

            p1_d = snap_au_contour(pts_droite[0], contour_1_global)
            p2_d = snap_au_contour(pts_droite[1], contour_1_global)

            # --- MISE À L'ÉCHELLE DU MORCEAU DROIT ---
            dist_g = np.linalg.norm(p1_g - p2_g)
            dist_d = np.linalg.norm(p1_d - p2_d)
            scale_d = dist_g / dist_d if dist_d > 0 else 1.0

            print(f"  ↳ Longueur de coupe (Gauche) : {dist_g:.2f} px")
            print(f"  ↳ Longueur de coupe (Droite) : {dist_d:.2f} px")
            print(f"  ↳ Mise à l'échelle du morceau droit : x{scale_d:.4f}")

            h_sq, w_sq = img_d["square"].shape[:2]
            new_h_sq, new_w_sq = max(1, int(h_sq * scale_d)), max(1, int(w_sq * scale_d))
            img_d["square"] = cv2.resize(img_d["square"], (new_w_sq, new_h_sq), interpolation=cv2.INTER_LANCZOS4)

            p1_d_local = (p1_d - np.array([y_trans, x_trans])) * scale_d
            p2_d_local = (p2_d - np.array([y_trans, x_trans])) * scale_d
            contour_1_scaled = contour_1 * scale_d



            # 3. Calcul des angles de correction
            corr_g = angle_de_correction(p1_g, p2_g)
            corr_d = angle_de_correction(p1_d_local, p2_d_local)
            angle_final_g = img_g["rotation_angle"] + corr_g
            angle_final_d = img_d["rotation_angle"] + corr_d

            # 4. Application des rotations aux matrices carrées d'origine
            nouvelle_img_g = rotate_center(img_g["square"], angle_final_g)
            nouvelle_img_d = rotate_center(img_d["square"], angle_final_d)

            center_g = (nouvelle_img_g.shape[1] / 2, nouvelle_img_g.shape[0] / 2)
            center_d = (nouvelle_img_d.shape[1] / 2, nouvelle_img_d.shape[0] / 2)

            # 5. Mise à jour des coordonnées des points d'ancrage après rotation
            new_p1_g = update_point(p1_g, corr_g, center_g)
            new_p2_g = update_point(p2_g, corr_g, center_g)
            
            new_p1_d_local = update_point(p1_d_local, corr_d, center_d)
            new_p2_d_local = update_point(p2_d_local, corr_d, center_d)

            # 6. Calcul de la nouvelle translation globale pour mettre face à face
            nouvelle_translation = ((new_p1_g + new_p2_g) / 2) - ((new_p1_d_local + new_p2_d_local) / 2)

            # 7. Mise à jour des calques d'images dans Napari
            v.layers['Morceau Gauche'].data = nouvelle_img_g
            v.layers['Morceau Droit'].data = nouvelle_img_d
            v.layers['Morceau Droit'].translate = nouvelle_translation.tolist()

            # Masquage des points rouges initiaux saisis par l'utilisateur
            v.layers['Limites de coupe'].visible = False

            # Affichage des 4 points réels finaux (magnétisés et repositionnés) pour vérification
            ligne_droite_finale_p1 = new_p1_d_local + nouvelle_translation
            ligne_droite_finale_p2 = new_p2_d_local + nouvelle_translation
            points_finaux = np.array([new_p1_g, new_p2_g, ligne_droite_finale_p1, ligne_droite_finale_p2])
            
            if 'Points de verification' in v.layers:
                v.layers['Points de verification'].data = points_finaux
            else:
                v.add_points(points_finaux, name='Points de verification', size=15, face_color='red')

            # 8. Calcul et tracé de la ligne médiane géométrique
            contour_g_final = transform_contour(contour_0, corr_g, center_g)
            contour_d_final = transform_contour(contour_1_scaled, corr_d, center_d, translation=nouvelle_translation)
            
            segment_g = get_contour_segment(contour_g_final, new_p1_g, new_p2_g)
            segment_d = get_contour_segment(contour_d_final, ligne_droite_finale_p1, ligne_droite_finale_p2)
            
            ligne_mediane = calculer_ligne_mediane_filtree(segment_g, segment_d)

            if len(ligne_mediane) > 0:
                if 'Ligne Mediane Idéale' in v.layers:
                    v.layers.remove(v.layers['Ligne Mediane Idéale'])
                v.add_points(ligne_mediane, size=2, face_color='green', border_width=0, name='Ligne Mediane Idéale')





            # --- SAUVEGARDE POUR L'ÉTAPE SUIVANTE ---
            etat_pipeline['segment_g'] = segment_g
            etat_pipeline['segment_d'] = segment_d
            etat_pipeline['ligne_mediane'] = ligne_mediane
            etat_pipeline['translation_droite'] = nouvelle_translation

            img_g["segment_coupe"] = segment_g
            img_d["segment_coupe"] = segment_d
            img_g["ligne_mediane"] = ligne_mediane
            img_g["p1g"] = new_p1_g
            img_g["p2g"] = new_p2_g

            # --- NOUVEAU : SAUVEGARDE DES PARAMÈTRES RIGIDES ---
            img_g["final_angle"] = angle_final_g
            img_g["final_scale"] = 1.0
            img_g["final_translation"] = np.array([0.0, 0.0])
            
            img_d["final_angle"] = angle_final_d
            img_d["final_scale"] = scale_d
            img_d["translation_droite"] = nouvelle_translation
            img_d["final_translation"] = nouvelle_translation

            img_d["translation_droite"] = nouvelle_translation
            img_g["rotated"] = nouvelle_img_g
            img_d["rotated"] = nouvelle_img_d

            print("\n[TRANSFORMATIONS RIGIDES FINALES]")
            print(f"  Rotation G : {angle_final_g:.2f}° | Rotation D : {angle_final_d:.2f}°")
            print(f"  Translation Droite -> Y: {nouvelle_translation[0]:.2f}px, X: {nouvelle_translation[1]:.2f}px")
            print("="*45 + "\n")

            return etat_pipeline
            
        except Exception:
            traceback.print_exc()