# Fusion non rigide d'images histologiques de prostate

Pipeline Python permettant de réaliser la **fusion non rigide de deux morceaux d'images histologiques de prostate** à l'aide d'une transformation **Thin Plate Spline (TPS)**.

Le projet a été réalisé dans le cadre d'un stage de L3 EEEA au LTSI. Les images histologiques utilisées étants celles de prostate, on sait que ça fonctionne sur ça. Théoriquement, ce pipeline peut fonctionner sur n'importe quel type d'images (histologiques).

## Fonctionnalités

* Chargement d'images histologiques TIFF.
* Suppression automatique de l'arrière-plan.
* Détection et conservation des principales composantes du tissu.
* Mise en forme des images dans une matrice carrée.
* Détection automatique de l'orientation des morceaux.
* Mise en vis-à-vis des deux morceaux.
* Sélection interactive de points de contrôle avec Napari.
* Fusion non rigide par transformation Thin Plate Spline.
* Calcul de cartes de déplacement `X` et `Y`.
* Accélération GPU avec CuPy pour les cartes graphiques NVIDIA.
* Exécution CPU possible sans GPU.
* Prise en charge d'images NIfTI.
* Application des cartes de déplacement TPS aux images NIfTI.
* Visualisation des différentes étapes avec Napari.

## Structure du projet

```text
fusion_non_rigide/
│
├── main_gpu.py
├── processing.py
├── f_tps_warp_2.py
├── f_tps_warp_gpu.py
├── nii_remap_pipeline.py
├── tif_mask.py
├── mask_image.py
├── napari_display.py
├── README.md
└── requirements.txt
```

### Rôle des principaux fichiers

| Fichier                 | Description                                                 |
| ----------------------- | ----------------------------------------------------------- |
| `main_gpu.py`           | Point d'entrée et orchestration du pipeline                 |
| `processing.py`         | Chargement, masquage, préparation et orientation des images |
| `tif_mask.py`           | Génération et traitement des masques TIFF                   |
| `mask_image.py`         | Application d'un masque et création d'une image RGBA        |
| `f_tps_warp_2.py`       | Calcul de la transformation TPS sur CPU                     |
| `f_tps_warp_gpu.py`     | Calcul de la transformation TPS avec GPU NVIDIA             |
| `nii_remap_pipeline.py` | Application des transformations et cartes TPS aux NIfTI     |
| `napari_display.py`     | Fonctions d'affichage avec Napari                           |

## Installation

Python 3.10 ou 3.11 est recommandé.

Cloner le dépôt :

```bash
git clone https://github.com/Kapamez/fusion_non_rigide.git
cd fusion_non_rigide
```

Créer un environnement virtuel :

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

## Utilisation

La configuration principale se trouve au début de `main_gpu.py`.

Il faut notamment renseigner les chemins des deux images TIFF :

```python
PATHS = [
    r"chemin/vers/image_gauche.tif",
    r"chemin/vers/image_droite.tif",
]
```

Pour utiliser les données NIfTI :

```python
NIFTI_PATH_GAUCHE = r"chemin/vers/image_gauche.nii"
NIFTI_PATH_DROITE = r"chemin/vers/image_droite.nii"

NIFTI_OUT_COMBINED = r"chemin/vers/resultat.nii"
```

Le facteur de réduction peut être modifié avec :

```python
DOWNSCALE_FACTOR = 16
```

`1` conserve la résolution originale, `2` divise les dimensions par deux, `4` par quatre, etc.

Lancer ensuite :

```bash
python main_gpu.py
```

## Déroulement du pipeline

### 1. Chargement

Les deux images TIFF sont chargées avec leurs dimensions originales.

Un facteur de réduction peut être appliqué afin de limiter le temps de calcul et la consommation mémoire.

### 2. Suppression de l'arrière-plan

Un masque est automatiquement calculé à partir de l'histogramme de l'image.

Les principales composantes connexes peuvent être conservées avec `TOP_N_LIST`.

```python
TOP_N_LIST = [1, 1]
```

### 3. Préparation géométrique

Chaque morceau est placé dans une matrice carrée suffisamment grande pour contenir sa diagonale.

Les deux morceaux sont ensuite orientés automatiquement afin de placer leurs zones de coupe face à face.

### 4. Sélection des points de contrôle

Une fenêtre Napari permet de sélectionner manuellement des points de correspondance.

Les points doivent être placés sur les deux morceaux dans le même ordre.

Le nombre de points doit être identique pour les deux images.

Une fois les points placés, la touche `e` permet de lancer la fusion.

### 5. Transformation TPS

Une transformation **Thin Plate Spline** est calculée à partir des points de contrôle.

Le projet dispose de deux implémentations :

* CPU : `f_tps_warp_2.py`
* GPU NVIDIA : `f_tps_warp_gpu.py`

Lorsque CuPy est disponible, `main_gpu.py` utilise automatiquement l'implémentation GPU.

### 6. Cartes de déplacement

Le pipeline calcule deux cartes de déplacement :

```text
disp_x_*.npy
disp_y_*.npy
```

Elles décrivent le déplacement appliqué à chaque pixel selon les axes X et Y.

Ces cartes peuvent ensuite être réutilisées pour transformer les images NIfTI correspondantes.

### 7. Fusion NIfTI

Le pipeline NIfTI reprend les transformations géométriques appliquées aux TIFF, puis applique les cartes TPS avec `cv2.remap`.

Les deux images transformées sont ensuite combinées et sauvegardées au format NIfTI.

## GPU NVIDIA

La version GPU utilise **CuPy**.

Le code détecte automatiquement la présence de CuPy :

```python
try:
    import cupy as cp
    GPU = True
except ModuleNotFoundError:
    GPU = False
```

Sans CuPy, le pipeline utilise automatiquement la version CPU.

L'installation de CuPy dépend de la version CUDA installée sur la machine. Il est donc recommandé d'installer la version correspondant à son environnement CUDA plutôt que de la fixer dans `requirements.txt`.

Exemple :

```bash
pip install cupy-cuda12x
```

Adapter `cuda12x` à la version CUDA utilisée.

## Configuration principale

Les paramètres importants sont regroupés au début de `main_gpu.py` :

```python
PATHS = [
    r"",
    r"",
]

MAKE_TIFF = False
TIFF_OUT_PATH = r""

DISP_FOLDER = r""

NIFTI_PATH_GAUCHE = r""
NIFTI_PATH_DROITE = r""
NIFTI_OUT_COMBINED = r""

OVERLAY_OUT_PATH = r""

TOP_N_LIST = [1, 1]
FLIP = [0, 0]

DOWNSCALE_FACTOR = 16

SHOW_STEPS = False
```

## Mémoire et performances

Les images histologiques pouvant être très volumineuses, la consommation mémoire peut devenir importante.

Le paramètre :

```python
DOWNSCALE_FACTOR = 16
```

permet de réduire fortement la taille des images avant les calculs.

Pour travailler à pleine résolution :

```python
DOWNSCALE_FACTOR = 1
```

Cela peut cependant augmenter fortement le temps de calcul et la consommation RAM.

Pour les traitements TPS volumineux, l'utilisation d'un GPU NVIDIA compatible CUDA peut également réduire le temps de calcul.

## Visualisation

Napari est utilisé pour :

* visualiser les images intermédiaires ;
* vérifier les masques ;
* contrôler l'orientation des morceaux ;
* sélectionner les points de correspondance ;
* vérifier le résultat de la fusion.

Pour afficher davantage d'étapes intermédiaires :

```python
SHOW_STEPS = True
```

## Formats d'entrée

Le pipeline principal utilise :

* TIFF / TIF pour les images histologiques ;
* NIfTI (`.nii`, `.nii.gz`) pour les données volumétriques associées ;
* NumPy (`.npy`) pour les cartes de déplacement.

## Limites actuelles

Le pipeline nécessite actuellement une intervention manuelle pour sélectionner les points de contrôle TPS.

Les chemins d'entrée et de sortie sont également configurés directement dans `main_gpu.py`.

Les images très haute résolution peuvent nécessiter une quantité importante de RAM et, pour l'accélération GPU, suffisamment de mémoire vidéo.

## Contexte

Projet réalisé par **Arthur J.R.**

Stage de L3 EEEA au **LTSI**.

Dépôt :

https://github.com/Kapamez/fusion_non_rigide

## Licence

Aucune licence open source n'est actuellement spécifiée dans le dépôt.
