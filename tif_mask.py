import numpy as np
from PIL import Image
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt
import time

# Fonction de seuillage : transforme une image en binaire selon un seuil.
# Les pixels strictement supérieurs au seuil deviennent 255, les autres deviennent 0.
def threshold_image(image_array, threshold):
    binary_image = (image_array > threshold).astype(np.uint8) * 255
    return binary_image


# Calcul de l'histogramme des niveaux de gris de l'image.
# On utilise 256 bins pour couvrir 0 -> 255.
def compute_histogram(image_array):
    """
    image_array : image 2D en niveaux de gris
    """
    hist, _ = np.histogram(image_array.flatten(), bins=256, range=(0, 255))
    return hist


def smooth_histogram(hist, sigma=1.5):
    """
    hist : histogramme brut à lisser
    sigma : paramètre de lissage (écart-type de la gaussienne)
    """
    # On lisse l'histogramme pour atténuer les petits maxima locaux.
    # Cela facilite la détection des deux pics principaux.
    return ndimage.gaussian_filter1d(hist.astype(float), sigma=sigma)


def find_peak_groups(hist_smooth, min_distance=20):
    """
    hist_smooth : histogramme lissé
    min_distance : distance minimale entre deux pics distincts
    """
    # Recherche des indices de maximum local dans l'histogramme lissé.
    candidate_indices = []

    # On considère un point comme un pic s'il est supérieur ou égal à ses voisins immédiats.
    for idx in range(1, len(hist_smooth) - 1):
        if hist_smooth[idx] >= hist_smooth[idx - 1] and hist_smooth[idx] >= hist_smooth[idx + 1]: # Si idx plus grand ou égal aux deux côtés
            candidate_indices.append(idx)

    # Pas de pics -> liste vide
    if not candidate_indices:
        return []
    
    # ---------------------------------------------------------------------
    # Regroupement des pics voisins dans des plages de largeur min_distance.
    # Cela évite de considérer plusieurs maxima adjacents comme des pics séparés.
    groups = [] # Liste finale qui contiendra tous les groupes
    current_group = [candidate_indices[0]] # On commence le premier groupe avec le premier pic candidat

    for idx in candidate_indices[1:]: # Parcours des pics candidats à partir du deuxième 
        if idx - current_group[-1] <= min_distance: # Si le pic actuel (idx) est à moins de (min_distance) du dernier pic du groupe en cours (current_group[-1])
            current_group.append(idx) # On ajoute le pic actuel au groupe en cours
        
        else:
            groups.append(current_group) # On sauvegarde le groupe terminé (current_group) dans le panier principal (groups)
            current_group = [idx] # On démarre un NOUVEAU groupe avec l'élément actuel (idx)

    groups.append(current_group) # Pour le dernier groupe à ajouter après la boucle
    # ---------------------------------------------------------------------

    # Pour chaque groupe, on choisit l'indice du pic le plus élevé.
    peak_centers = []
    for group in groups:
        group = np.array(group)
        center = group[np.argmax(hist_smooth[group])]
        peak_centers.append(int(center))
    

    # On trie les pics par hauteur décroissante.
    peak_centers.sort(key=lambda idx: hist_smooth[idx], reverse=True)
    return peak_centers


def auto_threshold_from_histogram(image_array, min_distance=20, smoothing_sigma=1.5):
    """
    Calcule un seuil de binarisation automatique à partir de l'histogramme de l'image.
    On détecte les deux pics les plus élevés de l'histogramme lissé, puis on cherche le minimum (le creux) entre ces deux pics pour définir le seuil optimal.
    img_array : image 2D en niveaux de gris
    min_distance : distance minimale entre deux pics distincts dans l'histogramme
    smoothing_sigma : paramètre de lissage de l'histogramme
    """

    # Calcul de l'histogramme, lissage et détection des pics
    hist = compute_histogram(image_array)
    hist_smooth = smooth_histogram(hist, sigma=smoothing_sigma)
    peaks = find_peak_groups(hist_smooth, min_distance=min_distance)

    # Vérification qu'on a bien détecté au moins deux pics distincts.
    if len(peaks) < 2:
        raise ValueError(
            "Impossible de détecter deux pics suffisamment distincts dans l'histogramme."
        )

    # On prend les deux pics les plus élevés.
    peak1 = peaks[0]
    peak2 = peaks[1]

    # On définit une région entre les deux pics pour chercher le minimum (le creux) qui correspond au seuil optimal.
    left = min(peak1, peak2)
    right = max(peak1, peak2)

    # On cherche le minimum dans la région entre les deux pics pour trouver le seuil optimal.
    valley_region = hist_smooth[left : right + 1]
    min_valley_index = int(np.argmin(valley_region))
    threshold = left + min_valley_index
    return threshold, hist, hist_smooth, peaks, threshold



def plot_histogram_process(image_array, hist, hist_smooth, peaks, threshold):
    # Affichage de l'histogramme brut, du lissage, des pics détectés et du seuil choisi.
    plt.figure(figsize=(12, 5))
    plt.bar(np.arange(len(hist)), hist, color='lightgray', label='Hist. brut', width=1.0)
    plt.plot(hist_smooth, color='blue', linewidth=2, label='Hist. lissé')
    plt.scatter(peaks, hist_smooth[peaks], color='red', s=80, zorder=3, label='Pics détectés')
    plt.axvline(threshold, color='green', linestyle='--', linewidth=2, label=f'Seuil auto = {threshold}')
    plt.title('Histogramme de niveaux de gris et seuil automatique')
    plt.xlabel('Valeur de niveau de gris')
    plt.ylabel('Nombre de pixels')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_image_process(image_array, binary_img):
    # Affichage de l'image source et du masque obtenu après seuillage.
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_array, cmap='gray')
    plt.title('Image utilisée pour l’histogramme')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(binary_img, cmap='gray')
    plt.title('Masque binaire obtenu')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def biggest_component_1(binary_image):
    """
    fonction qui prend une image binaire et ne garde que la plus grande composante.
    """
    # Analyse des composantes connexes pour filtrer le masque binaire.
    nb_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image)

    # Création d'une image vide
    largest_component = np.zeros_like(binary_image)

    min_size = 0.3 * binary_image.size  # Seuil de taille (30% de la taille totale de l'image)

    # Filtre par taille (on prend uniquement le plus grand composant)
    for i in range(1, nb_labels):  # On ignore le label 0 qui correspond au fond
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            largest_component[labels == i] = 255

    return largest_component

def biggest_component_2(binary_image, top_n=1, verbose=False):
    """
    Prend une image binaire et ne garde que les 'top_n' plus grandes composantes.
    
    top_n : nombre de plus grandes composantes à conserver (défaut: 1)
    verbose : si True, affiche la taille des composantes trouvées
    """
    # Analyse des composantes connexes
    nb_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image)

    # S'il n'y a que le fond (label 0), on retourne une image vide
    if nb_labels <= 1:
        return np.zeros_like(binary_image)

    # Récupération des tailles de chaque composante (en ignorant le fond, index 0)
    # On crée une liste de tuples : (label_id, surface)
    components = []
    for i in range(1, nb_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        components.append((i, area))

    # Tri de la liste par surface décroissante
    components.sort(key=lambda x: x[1], reverse=True)

    if verbose:
        print(f"Composantes trouvées (label, taille en pixels) : {components}")

    # Création de l'image vide pour le résultat
    result_component = np.zeros_like(binary_image)

    # On garde seulement les 'top_n' premiers éléments (ou le maximum disponible)
    for i in range(min(top_n, len(components))):
        label_to_keep = components[i][0]
        result_component[labels == label_to_keep] = 255

    return result_component
    

def fill_large_defects(binary_image, min_depth=150, max_width = 400):
    """
    Ferme les grandes failles sur les bords de l'objet en utilisant les défauts de convexité.
    min_depth : profondeur minimale de la faille.
    max_width : largeur maximale de l'embouchure (pour ne pas fermer les courbes naturelles).
    """
    # 1. Extraction du contour externe
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return binary_image

    # On prend le plus grand contour (la prostate)
    cnt = max(contours, key=cv2.contourArea)

    # 2. Calcul de l'enveloppe convexe et des défauts
    # returnPoints=False est obligatoire pour convexityDefects (il a besoin des index, pas des coordonnées)
    hull = cv2.convexHull(cnt, returnPoints=False)
    
    try:
        defects = cv2.convexityDefects(cnt, hull)
    except cv2.error:
        # Arrive si la forme est déjà parfaitement convexe
        return binary_image

    if defects is None:
        return binary_image

    result_img = binary_image.copy()

    # 3. Parcours des défauts
    for i in range(defects.shape[0]):
        # s: start (début de la faille), e: end (fin), f: farthest (fond du trou), d: distance
        s, e, f, d = defects[i, 0]
        
        # La distance renvoyée par OpenCV est multipliée par 256 (précision sub-pixel)
        depth = d / 256.0

        # Si le gouffre est plus profond que notre seuil
        if depth > min_depth:
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            
            # Calcul de la largeur de l'ouverture (distance euclidienne entre start et end)
            width = np.linalg.norm(np.array(start) - np.array(end))
            
            # On ne ferme que si l'ouverture est plus petite que la limite max
            if width < max_width:
                cv2.line(result_img, start, end, 255, thickness=10)

    # 4. Remplissage
    # Maintenant que l'embouchure est fermée, la faille est devenue un "trou" interne.
    # ndimage.binary_fill_holes va le combler parfaitement.
    result_img = ndimage.binary_fill_holes(result_img).astype(np.uint8) * 255
    
    return result_img



def mask(path, channel='L',
         threshold=None,
         plot=False,
         show_img=False,
         min_peak_distance=20,
         op_clo=True,
         gaussian_blur=True,
         fix_tears=True,
        top_n=1,
         ):
    """
    Fonction masque
    path : chemin de l'image TIFF
    channel : 'L' pour niveaux de gris ou 'R','G','B' pour un canal RGB
    threshold : valeur de seuil manuelle, ou None pour seuil automatique
    plot : si True, affiche les graphiques de diagnostic
    show_img : si True, enregistre les images résultats
    min_peak_distance : distance minimale entre deux pics distincts dans l'histogramme
    op_clo : si True, applique les opérations d'ouverture et de fermeture
    gaussian_blur : si True, applique un flou gaussien pour lisser les contours du masque
    """

    save_path = "C:\\Users\\Gustave\\Documents\\Education\\L3_EEEA\\Stage\\Pour_Arthur\\Debug\\"

    # Chargement de l'image
    img = Image.open(path)

    # Conversion du canal choisi en image 2D (niveau de gris)
    if channel == 'L':
        img = img.convert('L')
        image_array = np.array(img)
    else:
        img = img.convert('RGB')
        arr = np.array(img)
        channel_index = {'R': 0, 'G': 1, 'B': 2}
        if channel not in channel_index:
            raise ValueError("Le canal doit être 'L', 'R', 'G' ou 'B'.")
        image_array = arr[:, :, channel_index[channel]]

    # Si aucun seuil manuel n'est fourni, on le calcule automatiquement
    # à partir des deux plus grands pics de l'histogramme.
    if threshold is None:
        threshold, hist, hist_smooth, peaks, _ = auto_threshold_from_histogram(
            image_array,
            min_distance=min_peak_distance,
            smoothing_sigma=1.5,
        )
    else:
        hist = compute_histogram(image_array)
        hist_smooth = smooth_histogram(hist, sigma=1.5)
        peaks = find_peak_groups(hist_smooth, min_distance=min_peak_distance)

    # Application du seuillage pour produire une image binaire.
    binary_img = threshold_image(image_array, threshold)
    inverted_binary_img = 255 - binary_img

    # --------

    # Détection des composantes connexes sur l'image inversée. #1
    labeled_img = biggest_component_2(inverted_binary_img, top_n=top_n, verbose=False)  # On ne garde que la plus grande composante

    if show_img:
        name = "S1_" + time.strftime("%Y%m%d-%H%M%S") + ".png"
        labeled_img_pil = Image.fromarray(labeled_img)
        labeled_img_pil.save(save_path + name)

    # Opening
    if op_clo:
        kernel = np.ones((5, 5), np.uint8)
        img_op = cv2.morphologyEx(labeled_img, cv2.MORPH_OPEN, kernel)
    else:
        img_op = labeled_img

    if show_img:
        name = "S2_" + time.strftime("%Y%m%d-%H%M%S") + ".png"
        img_op_pil = Image.fromarray(img_op)
        img_op_pil.save(save_path + name)

    # Closing
    if op_clo:
        kernel = np.ones((5, 5), np.uint8)
        img_close = cv2.morphologyEx(img_op, cv2.MORPH_CLOSE, kernel)
    else:
        img_close = img_op

    if show_img:
        name = "S3_" + time.strftime("%Y%m%d-%H%M%S") + ".png"
        img_close_pil = Image.fromarray(img_close)
        img_close_pil.save(save_path + name)


    # --------
    # 1. TECHNIQUE DU BOUCHON
    # --------
    if fix_tears:
        # On crée un noyau circulaire pour sceller l'entrée
        taille_bouchon = 100
        kernel_seal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (taille_bouchon, taille_bouchon))
        
        # Le closing scelle l'entrée
        img_sealed = cv2.morphologyEx(img_close, cv2.MORPH_CLOSE, kernel_seal)
        
        if show_img:
            name = "S3b_SEALED_" + time.strftime("%Y%m%d-%H%M%S") + ".png"
            Image.fromarray(img_sealed).save(save_path + name)
    else:
        img_sealed = img_close


    # --------
    # 2. REMPLISSAGE
    # --------
    filled_img = ndimage.binary_fill_holes(img_sealed).astype(np.uint8) * 255

    if show_img:
        name = "S4_" + time.strftime("%Y%m%d-%H%M%S") + ".png"
        filled_img_pil = Image.fromarray(filled_img)
        filled_img_pil.save(save_path + name)


    # --------
    # 3. FLOU GAUSSIEN
    # --------
    if gaussian_blur:
        # Flou Gaussien pour lisser les contours, puis seuillage binaire pour obtenir un masque net.
        img_blur = cv2.GaussianBlur(filled_img, (5, 5), 0)
        img_smooth = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY)[1]

        if show_img:
            name  = "T1_" + time.strftime("%Y%m%d-%H%M%S") + ".png"
            img_smooth_pil = Image.fromarray(img_smooth)
            img_smooth_pil.save(save_path + name)
    
    else:
        img_smooth = filled_img

    # --------

    # Détection des composantes connexes sur l'image inversée. #2
    labeled_img2 = biggest_component_2(img_smooth, top_n=top_n, verbose=False)  # On ne garde que la plus grande composante

    if show_img:
        name = "N1_" + time.strftime("%Y%m%d-%H%M%S") + ".png"
        labeled_img2_pil = Image.fromarray(labeled_img2)
        labeled_img2_pil.save(save_path + name)

    # Remplissage des trous #2
    filled_img2 = ndimage.binary_fill_holes(labeled_img2).astype(np.uint8) * 255

    # --------

    # Affichage optionnel des diagnostics.
    if plot:
        plot_histogram_process(image_array, hist, hist_smooth, peaks, threshold)
        plot_image_process(image_array, inverted_binary_img)

    return filled_img2

# Exemple d'utilisation :
#path = "C:\\Users\\Gustave\\Documents\\Education\\L3_EEEA\\Stage\\Pour_Arthur\\Prostate8\\A9_x1.25_z0.tif"

#mask_result = mask(path, channel='L', threshold=None, plot=False, show_img=True)

#print("Type : ", type(mask_result))