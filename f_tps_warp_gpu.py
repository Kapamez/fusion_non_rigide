import numpy as np
import cupy as cp
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import cdist # Utilisé uniquement pour solve_tps (CPU)

# Garder solve_tps sur CPU car N est petit (la résolution de système linéaire 
# de petite taille est souvent plus rapide sur CPU).
def solve_tps(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    N = src.shape[0]
    # On reste sur l'ancienne méthode de distance pour le petit système
    dists = cdist(src, src)
    
    with np.errstate(divide="ignore", invalid="ignore"):
        K = np.where(dists == 0.0, 0.0, dists**2 * np.log(dists**2))
        
    P = np.hstack([np.ones((N, 1)), src])

    top    = np.hstack([K, P])
    bottom = np.hstack([P.T, np.zeros((3, 3))])
    L = np.vstack([top, bottom])

    rhs = np.vstack([dst, np.zeros((3, 2))])

    lambda_reg = 1e-6
    L[:N, :N] += lambda_reg * np.eye(N)

    params = np.linalg.solve(L, rhs)

    W = params[:N]
    A = params[N:]
    return W, A

def tps_kernel_gpu(r_sq: cp.ndarray) -> cp.ndarray:
    """ Noyau TPS prenant directement la distance au carré """
    res = cp.zeros_like(r_sq)
    # On évite le log(0)
    mask = r_sq > 1e-10 
    r_sq_valid = r_sq[mask]
    res[mask] = r_sq_valid * cp.log(r_sq_valid)
    return res

def bilinear_interpolate_gpu(image: cp.ndarray, x: cp.ndarray, y: cp.ndarray) -> cp.ndarray:
    H, W, C = image.shape

    x0 = cp.floor(x).astype(cp.int32)
    y0 = cp.floor(y).astype(cp.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = (x - x0).astype(cp.float32)
    wy = (y - y0).astype(cp.float32)

    valid = (x0 >= 0) & (x1 < W) & (y0 >= 0) & (y1 < H)

    x0c = cp.clip(x0, 0, W - 1)
    x1c = cp.clip(x1, 0, W - 1)
    y0c = cp.clip(y0, 0, H - 1)
    y1c = cp.clip(y1, 0, H - 1)

    I00 = image[y0c, x0c]
    I10 = image[y0c, x1c]
    I01 = image[y1c, x0c]
    I11 = image[y1c, x1c]

    wx = wx[:, :, cp.newaxis]
    wy = wy[:, :, cp.newaxis]

    result = (1 - wy) * ((1 - wx) * I00 + wx * I10) \
           +      wy  * ((1 - wx) * I01 + wx * I11)

    result[~valid] = 0.0

    return result

def apply_tps_gpu(image, src, dst, chunk_size=100_000):
    st = time.time()
    H, W_img, C = image.shape

    # src et dst sont en [y, x] (convention Napari) — pas d'inversion nécessaire
    W_coef, A_coef = solve_tps(dst, src)          # dst → src (mapping inverse)

    W_coef_gpu = cp.asarray(W_coef, dtype=cp.float64)
    A_coef_gpu = cp.asarray(A_coef, dtype=cp.float64)
    dst_gpu    = cp.asarray(dst,    dtype=cp.float64)   # en [y, x]
    image_gpu  = cp.asarray(image,  dtype=cp.float32)

    x = cp.arange(W_img, dtype=cp.float64)
    y = cp.arange(H,     dtype=cp.float64)
    cols, rows = cp.meshgrid(x, y)
    grid_gpu = cp.stack([rows.ravel(), cols.ravel()], axis=1)  # [y, x] ✓

    n_pixels = H * W_img
    src_coords = cp.empty((n_pixels, 2), dtype=cp.float64)

    for start in range(0, n_pixels, chunk_size):
        end   = min(start + chunk_size, n_pixels)
        chunk = grid_gpu[start:end]

        g_sq  = cp.sum(chunk**2,   axis=1, keepdims=True)
        d_sq  = cp.sum(dst_gpu**2, axis=1)
        r_sq  = cp.clip(g_sq + d_sq - 2.0 * cp.dot(chunk, dst_gpu.T), 0.0, None)

        Kgrid = tps_kernel_gpu(r_sq)
        P     = cp.hstack([cp.ones((chunk.shape[0], 1), dtype=cp.float64), chunk])
        src_coords[start:end] = cp.dot(Kgrid, W_coef_gpu) + cp.dot(P, A_coef_gpu)

    # src_coords[:,0] = y (row),  src_coords[:,1] = x (col)
    src_row = src_coords[:, 0].reshape(H, W_img).astype(cp.float32)
    src_col = src_coords[:, 1].reshape(H, W_img).astype(cp.float32)

    # bilinear_interpolate_gpu(image, x, y) → x=col, y=row
    warped_gpu = bilinear_interpolate_gpu(image_gpu, src_col, src_row)
    warped = warped_gpu.get()

    print(f"apply_tps_gpu exécuté en {time.time() - st:.3f} s")
    return warped

def bilinear_interpolate(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    H, W, C = image.shape

    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)

    valid = (x0 >= 0) & (x1 < W) & (y0 >= 0) & (y1 < H)

    x0c = np.clip(x0, 0, W - 1)
    x1c = np.clip(x1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1)
    y1c = np.clip(y1, 0, H - 1)

    I00 = image[y0c, x0c]
    I10 = image[y0c, x1c]
    I01 = image[y1c, x0c]
    I11 = image[y1c, x1c]

    wx = wx[:, :, np.newaxis]
    wy = wy[:, :, np.newaxis]

    result = (1 - wy) * ((1 - wx) * I00 + wx * I10) \
           +      wy  * ((1 - wx) * I01 + wx * I11)

    result[~valid] = 0.0

    return result.astype(np.float32)


def tps_kernel_cpu(dists: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(dists == 0.0, 0.0, dists**2 * np.log(dists**2))


def compute_displacement_map_gpu(image: np.ndarray, src: np.ndarray, dst: np.ndarray,
                                  chunk_size: int = 80_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    st = time.time()
    H, W = image.shape[:2]

    W_coef, A_coef = solve_tps(dst, src)
    print(f"compute_displacement_map : solve_tps - {time.time()-st:.3f} s")

    W_coef_gpu = cp.asarray(W_coef, dtype=cp.float64)
    A_coef_gpu = cp.asarray(A_coef, dtype=cp.float64)
    dst_gpu    = cp.asarray(dst,    dtype=cp.float64)

    x = cp.arange(W, dtype=cp.float64)
    y = cp.arange(H, dtype=cp.float64)
    cols, rows = cp.meshgrid(x, y)
    grid_gpu = cp.stack([rows.ravel(), cols.ravel()], axis=1)  # [y, x]

    n_pixels = H * W
    src_coords = cp.empty((n_pixels, 2), dtype=cp.float64)

    t = time.time()
    for i, start in enumerate(range(0, n_pixels, chunk_size)):
        end = min(start + chunk_size, n_pixels)
        chunk = grid_gpu[start:end]

        g_sq = cp.sum(chunk**2, axis=1, keepdims=True)
        d_sq = cp.sum(dst_gpu**2, axis=1)
        r_sq = cp.clip(g_sq + d_sq - 2.0 * cp.dot(chunk, dst_gpu.T), 0.0, None)

        Kgrid = tps_kernel_gpu(r_sq)
        P = cp.hstack([cp.ones((chunk.shape[0], 1), dtype=cp.float64), chunk])
        src_coords[start:end] = cp.dot(Kgrid, W_coef_gpu) + cp.dot(P, A_coef_gpu)

    print(f"compute_displacement_map : chunks - {time.time()-t:.3f} s")

    t = time.time()
    # Le déplacement est la différence entre la position d'origine (grid_gpu) 
    # et la position source calculée (src_coords)
    delta = src_coords - grid_gpu
    
    # Pour l'affichage, on garde la norme (amplitude)
    amplitude = cp.linalg.norm(delta, axis=1)
    disp_map = amplitude.reshape(H, W).astype(cp.float32).get()
    
    # Séparation des composantes Y (axe 0) et X (axe 1)
    disp_y = delta[:, 0].reshape(H, W).astype(cp.float32).get()
    disp_x = delta[:, 1].reshape(H, W).astype(cp.float32).get()
    
    print(f"compute_displacement_map : delta+norm - {time.time()-t:.3f} s")
    print(f"compute_displacement_map total : {time.time()-st:.3f} s")
    
    return disp_map, disp_y, disp_x


# ─────────────────────────────────────────────────────────────
#  2. AFFICHAGE INTERMÉDIAIRE
# ─────────────────────────────────────────────────────────────

def _draw_checkerboard_on(ax, H, W):
    tile = 20
    checker = np.zeros((H, W))
    for i in range(0, H, tile):
        for j in range(0, W, tile):
            if (i // tile + j // tile) % 2 == 0:
                checker[i:i+tile, j:j+tile] = 0.6
            else:
                checker[i:i+tile, j:j+tile] = 0.85
    ax.imshow(checker, cmap="gray", vmin=0, vmax=1, zorder=1)

def plot_tps_comparison(image: np.ndarray, warped: np.ndarray, disp_map: np.ndarray, src: np.ndarray, dst: np.ndarray):
    H, W = image.shape[:2]
    COLORS = {"source": "#2979FF", "dest": "#FF1744", "arrow": "#FF9100"}

    fig, axes = plt.subplots(1, 3, figsize=(19, 7))
    fig.patch.set_facecolor("#1A1A2E")
    for ax in axes:
        ax.set_facecolor("#16213E")
        ax.axis("off")

    # Panneau 0 : image originale avec points et flèches
    _draw_checkerboard_on(axes[0], H, W)
    axes[0].imshow(image, zorder=2)
    axes[0].set_title("Image originale\n● source  ✕ destination  → déplacement", color="white", fontsize=10, pad=8)

    for i, (s, d) in enumerate(zip(src, dst)):
        sy, sx = s[0], s[1]
        dy, dx = d[0], d[1]
        axes[0].plot(sx, sy, "o", color=COLORS["source"], markersize=9, markeredgewidth=2, zorder=5)
        axes[0].text(sx + 5, sy - 5, str(i + 1), color=COLORS["source"], fontsize=8, fontweight="bold", zorder=6)
        axes[0].plot(dx, dy, "x", color=COLORS["dest"], markersize=9, markeredgewidth=2.5, zorder=5)
        axes[0].text(dx + 5, dy - 5, str(i + 1), color=COLORS["dest"], fontsize=8, fontweight="bold", zorder=6)
        axes[0].annotate("", xy=(dx, dy), xytext=(sx, sy),
                         arrowprops=dict(arrowstyle="->", color=COLORS["arrow"], lw=1.8), zorder=7)

    # Panneau 1 : image déformée avec repères
    _draw_checkerboard_on(axes[1], H, W)
    axes[1].imshow(warped, zorder=2)
    axes[1].set_title("Image déformée (TPS)\n● origine (src)  ✕ arrivée (dst)", color="white", fontsize=10, pad=8)

    for i, (s, d) in enumerate(zip(src, dst)):
        sy, sx = s[0], s[1]
        dy, dx = d[0], d[1]
        axes[1].plot(sx, sy, "o", color=COLORS["source"], markersize=9, markeredgewidth=2, zorder=5, markerfacecolor="none")
        axes[1].text(sx + 5, sy - 5, str(i + 1), color=COLORS["source"], fontsize=8, fontweight="bold", zorder=6)
        axes[1].plot(dx, dy, "x", color=COLORS["dest"], markersize=9, markeredgewidth=2.5, zorder=5)
        axes[1].text(dx + 5, dy - 5, str(i + 1), color=COLORS["dest"], fontsize=8, fontweight="bold", zorder=6)

    # Panneau 2 : carte de déplacement colormap jet
    # Masquage sur canal alpha si présent
    alpha_mask = image[:, :, 3] if image.shape[2] == 4 else np.ones((H, W))

    max_disp = disp_map.max()
    disp_norm = disp_map / max_disp if max_disp > 0 else disp_map

    cmap = plt.colormaps["jet"]
    disp_colored = cmap(disp_norm)[:, :, :3]
    disp_colored *= alpha_mask[:, :, np.newaxis]

    axes[2].imshow(disp_colored, zorder=2)
    axes[2].set_title(f"Carte de déplacement (jet)\nbleu=peu bougé  →  rouge=beaucoup bougé  |  max={max_disp:.1f} px", color="white", fontsize=10, pad=8)

    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=0, vmax=max_disp))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("Déplacement (pixels)", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")

    # Légende globale
    src_patch = mpatches.Patch(color=COLORS["source"], label="Point source (origine)")
    dst_patch = mpatches.Patch(color=COLORS["dest"],   label="Point destination (arrivée)")
    arr_patch = mpatches.Patch(color=COLORS["arrow"],  label="Flèche de déplacement")
    fig.legend(handles=[src_patch, dst_patch, arr_patch], loc="lower center", ncol=3, facecolor="#0F3460", labelcolor="white", framealpha=0.9, fontsize=9)

    fig.suptitle(f"TPS — {len(src)} points de contrôle", color="white", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────
#  3. FONCTION PRINCIPALE D'AUTOMATISATION
# ─────────────────────────────────────────────────────────────

def auto_warp_image(image: np.ndarray, src_points: np.ndarray, dst_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applique la déformation TPS sur une image de manière automatique.

    Paramètres :
        image      : (H, W, C) np.ndarray (float32, valeurs entre 0 et 1)
        src_points : (N, 2) np.ndarray
        dst_points : (N, 2) np.ndarray

    Retourne :
        warped     : (H, W, C) np.ndarray déformé
        disp_y     : (H, W) np.ndarray déplacement de chaque pixel sur l'axe Y
        disp_x     : (H, W) np.ndarray déplacement de chaque pixel sur l'axe X
    """
    print(f"⚙  Calcul de la déformation TPS ({len(src_points)} points de contrôle)…")
    warped = apply_tps_gpu(image, src_points, dst_points)
    print("✓  Déformation calculée.")

    print("⚙  Calcul de la carte de déplacement…")
    disp_map, disp_y, disp_x = compute_displacement_map_gpu(image, src_points, dst_points)
    print(f"✓  Déplacement max : {disp_map.max():.1f} px  |  moyen : {disp_map.mean():.1f} px")

    # Affichage intermédiaire de contrôle
    plot_tps_comparison(image, warped, disp_map, src_points, dst_points)

    return warped, disp_y, disp_x