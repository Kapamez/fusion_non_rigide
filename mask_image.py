import numpy as np

def apply_mask(img_source, img_mask):
    """
    Applique un masque binaire à une image pour créer une image avec transparence.
    
    Paramètres:
        img_source : array numpy de l'image (H, W, 3) ou (H, W, 4)
        img_mask : array numpy du masque (H, W) avec valeurs 0 ou 255
                   Blanc (255) = garder, Noir (0) = transparent
    
    Retourne:
        array numpy de l'image avec canal alpha (H, W, 4), valeurs 0-255
    """
    
    # S'assurer que l'image est en uint8
    image_base = np.array(img_source, dtype=np.uint8)
    masque = np.array(img_mask, dtype=np.uint8)
    
    h, w = masque.shape[:2]
    
    # Normaliser le masque de 0-255 à 0-1 (float)
    alpha = masque.astype(np.float32) / 255.0
    
    # Si l'image de base a déjà un canal alpha, on le retire
    if image_base.ndim == 3 and image_base.shape[2] == 4:
        rgb = image_base[:, :, :3]
        
    elif image_base.ndim == 3 and image_base.shape[2] == 3:
        rgb = image_base
    else:
        # Si image en niveaux de gris, convertir en RGB
        rgb = np.stack([image_base] * 3, axis=-1) if image_base.ndim == 2 else image_base
    
    # Recadrer au cas où les dimensions ne correspondent pas exactement
    rgb = rgb[:h, :w]
    
    # Créer l'image RGBA : RGB + canal alpha
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = masque  # Le masque devient le canal alpha
    
    return rgba




