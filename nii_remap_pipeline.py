"""
nii_remap_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Rejoue le pipeline géométrique (rigide) sur les images NIfTI, puis applique
les cartes de déplacement TPS (.npy) mises à l'échelle via cv2.remap.
"""

import numpy as np
import nibabel as nib
import cv2
import math
import glob
import os

def load_nifti(path):
    """Charge un NIfTI 2D/3D et retourne les données 2D + métadonnées."""
    nii = nib.load(path)
    data = nii.get_fdata().astype(np.float32)
    if data.ndim == 3:
        data = data[:, :, 0]  # Prend la première slice si 3D
    return data, nii.affine, nii.header

def apply_rigid_transforms(nii_data, tiff_meta):
    """
    Applique le recadrage, la matrice carrée, la rotation et la translation 
    calculées sur le TIFF.
    """
    ratio = nii_data.shape[1] / tiff_meta["w"]
    
    # 1. Crop (BBox mis à l'échelle)
    rmin, rmax, cmin, cmax = tiff_meta["bbox"]
    rmin_n, rmax_n = int(rmin * ratio), int(rmax * ratio)
    cmin_n, cmax_n = int(cmin * ratio), int(cmax * ratio)
    nii_crop = nii_data[rmin_n:rmax_n+1, cmin_n:cmax_n+1]

    # 2. Pad to diagonal square (Alignement parfait sur le TIFF)
    h, w = nii_crop.shape[:2]
    
    # Utiliser exactement la diagonale du TIFF mise à l'échelle
    a_tiff = tiff_meta.get("diagonal", math.ceil(math.sqrt(tiff_meta["w"]**2 + tiff_meta["h"]**2)))
    a = int(round(a_tiff * ratio))
    
    # Utiliser exactement les mêmes offsets que le TIFF, mis à l'échelle
    row_off_tiff = (a_tiff - tiff_meta["h"]) // 2
    col_off_tiff = (a_tiff - tiff_meta["w"]) // 2
    row_off = int(round(row_off_tiff * ratio))
    col_off = int(round(col_off_tiff * ratio))
    
    # Sécurité pour éviter de déborder
    h = min(h, a - row_off)
    w = min(w, a - col_off)
    
    nii_square = np.zeros((a, a), dtype=np.float32)
    nii_square[row_off:row_off+h, col_off:col_off+w] = nii_crop[:h, :w]

    # 3. Scale (uniquement pour la pièce droite)
    scale = tiff_meta.get("final_scale", 1.0)
    if scale != 1.0:
        h_sq, w_sq = nii_square.shape[:2]
        new_h_sq, new_w_sq = max(1, int(round(h_sq * scale))), max(1, int(round(w_sq * scale)))
        nii_square = cv2.resize(nii_square, (new_w_sq, new_h_sq), interpolation=cv2.INTER_LANCZOS4)

    # 4. Rotate
    # Utilisation de rotation_angle (clé réelle) avec fallback
    angle = tiff_meta.get("final_angle", tiff_meta.get("rotation_angle", 0))
    center = (nii_square.shape[1] / 2, nii_square.shape[0] / 2)
    M_rot = cv2.getRotationMatrix2D(center, float(angle), 1.0)
    nii_rotated = cv2.warpAffine(nii_square, M_rot, (nii_square.shape[1], nii_square.shape[0]), flags=cv2.INTER_LINEAR)

    # 5. Translate
    # Correction critique : utiliser "translation_droite" au lieu de "final_translation"
    trans_y, trans_x = tiff_meta.get("translation_droite", tiff_meta.get("final_translation", [0, 0]))
    
    if trans_y != 0 or trans_x != 0:
        tiff_rotated_h = tiff_meta["rotated"].shape[0]
        tiff_rotated_w = tiff_meta["rotated"].shape[1]
        ratio_y = nii_rotated.shape[0] / tiff_rotated_h
        ratio_x = nii_rotated.shape[1] / tiff_rotated_w
        
        t_y_nii = int(round(trans_y * ratio_y))
        t_x_nii = int(round(trans_x * ratio_x))
        
        # La taille du canevas augmente si le décalage est positif
        new_h = nii_rotated.shape[0] + max(0, t_y_nii)
        new_w = nii_rotated.shape[1] + max(0, t_x_nii)
        
        # Correction critique : ne pas utiliser max(0, ...) dans la matrice M_trans
        # Il faut utiliser la vraie valeur (négative ou positive) pour décaler l'image
        M_trans = np.float32([[1, 0, t_x_nii], [0, 1, t_y_nii]])
        nii_final = cv2.warpAffine(nii_rotated, M_trans, (new_w, new_h), flags=cv2.INTER_LINEAR)
    else:
        nii_final = nii_rotated

    return nii_final

def apply_tps_remap(nii_img, disp_x_path, disp_y_path):
    """Applique les cartes de déplacement TPS sur l'image NIfTI."""
    disp_x = np.load(disp_x_path)
    disp_y = np.load(disp_y_path)

    H, W = nii_img.shape[:2]
    
    # 1. Calculer le ratio d'échelle
    scale_x = W / disp_x.shape[1]
    scale_y = H / disp_x.shape[0]
    
    # 2. Redimensionner la grille spatiale
    disp_x_resized = cv2.resize(disp_x, (W, H), interpolation=cv2.INTER_LINEAR)
    disp_y_resized = cv2.resize(disp_y, (W, H), interpolation=cv2.INTER_LINEAR)
    
    # 3. MULTIPLIER LES VALEURS DE DÉPLACEMENT PAR LE RATIO !
    disp_x_resized *= scale_x
    disp_y_resized *= scale_y

    map_x, map_y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    map_x += disp_x_resized
    map_y += disp_y_resized

    warped = cv2.remap(nii_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped

def combine_images(img_g, img_d, shift_xy):
    """
    Fusionne img_g et img_d dans un grand canevas.
    shift_xy : (y, x) décalage de img_d par rapport à img_g.
    """
    h_g, w_g = img_g.shape
    h_d, w_d = img_d.shape
    sy, sx = shift_xy
    
    min_y = min(0, sy)
    max_y = max(h_g, sy + h_d)
    min_x = min(0, sx)
    max_x = max(w_g, sx + w_d)
    
    out_h = int(max_y - min_y)
    out_w = int(max_x - min_x)
    
    canvas = np.zeros((out_h, out_w), dtype=np.float32)
    
    # Placement Gauche
    yg1, xg1 = int(0 - min_y), int(0 - min_x)
    canvas[yg1:yg1+h_g, xg1:xg1+w_g] = img_g
    
    # Placement Droit
    yd1, xd1 = int(sy - min_y), int(sx - min_x)
    target = canvas[yd1:yd1+h_d, xd1:xd1+w_d]
    canvas[yd1:yd1+h_d, xd1:xd1+w_d] = np.maximum(target, img_d)
    
    return canvas

def run_nii_remap_pipeline(nii_path_g, nii_path_d, tiff_g, tiff_d, out_path_combined, disp_folder, tiff_shift_x=0):
    print("\n" + "═" * 60)
    print("  PIPELINE NIfTI — Replay & Remap")
    print("═" * 60)

    # 1. Charger NIfTI
    nii_g_data, affine_g, header_g = load_nifti(nii_path_g)
    nii_d_data, affine_d, header_d = load_nifti(nii_path_d)
    
    # Normalisation [0, 1]
    max_g, max_d = np.max(nii_g_data), np.max(nii_d_data)
    if max_g > 0: nii_g_data /= max_g
    if max_d > 0: nii_d_data /= max_d

    # 2. Appliquer les transformations rigides
    print("  ⚙  Application des transformations rigides (Crop, Rot, Trans)...")
    nii_g_prepared = apply_rigid_transforms(nii_g_data, tiff_g)
    nii_d_prepared = apply_rigid_transforms(nii_d_data, tiff_d)

    # 3. Trouver les dernières cartes .npy sauvegardées
    files_x_g = glob.glob(os.path.join(disp_folder, "disp_x_gauche_*.npy"))
    files_y_g = glob.glob(os.path.join(disp_folder, "disp_y_gauche_*.npy"))
    files_x_d = glob.glob(os.path.join(disp_folder, "disp_x_droite_*.npy"))
    files_y_d = glob.glob(os.path.join(disp_folder, "disp_y_droite_*.npy"))
    
    if not files_x_g or not files_x_d:
        raise FileNotFoundError("Cartes de déplacement .npy introuvables !")

    disp_x_g_path = max(files_x_g, key=os.path.getmtime)
    disp_y_g_path = max(files_y_g, key=os.path.getmtime)
    disp_x_d_path = max(files_x_d, key=os.path.getmtime)
    disp_y_d_path = max(files_y_d, key=os.path.getmtime)

    # 4. Appliquer TPS Remap
    print("  ⚙  Application des cartes de déplacement TPS (cv2.remap)...")
    warped_g_nii = apply_tps_remap(nii_g_prepared, disp_x_g_path, disp_y_g_path)
    warped_d_nii = apply_tps_remap(nii_d_prepared, disp_x_d_path, disp_y_d_path)

    # 5. Combiner les deux moitiés
    print("  ⚙  Fusion des deux moitiés NIfTI...")
    ratio_tiff_to_nii = nii_g_prepared.shape[1] / tiff_g["rotated"].shape[1]
    nii_shift_x = int(tiff_shift_x * ratio_tiff_to_nii)
    
    combined_nii = combine_images(warped_g_nii, warped_d_nii, (0, nii_shift_x))

    # 6. Sauvegarder
    os.makedirs(os.path.dirname(out_path_combined), exist_ok=True)
    out_combined = (combined_nii * max_g).astype(np.float32)
    nib.save(nib.Nifti1Image(out_combined, affine_g, header_g), out_path_combined)
    
    print(f"\n  ✓  NIfTI combiné sauvegardé :\n     {out_path_combined}")
    
    return combined_nii