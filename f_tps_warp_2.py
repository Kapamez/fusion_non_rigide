import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import cdist

# Résolution du système linéaire TPS
def solve_tps(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    N = src.shape[0]
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

def tps_kernel_cpu(r_sq: np.ndarray) -> np.ndarray:
    """ Noyau TPS prenant directement la distance au carré """
    res = np.zeros_like(r_sq)
    # On évite le log(0)
    mask = r_sq > 1e-10 
    r_sq_valid = r_sq[mask]
    res[mask] = r_sq_valid * np.log(r_sq_valid)
    return res

def bilinear_interpolate_cpu(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    H, W, C = image.shape

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
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

    return result

def apply_tps_cpu(image, src, dst, chunk_size=100_000):
    st = time.time()
    H, W_img, C = image.shape

    # src et dst sont en [y, x] — mapping inverse
    W_coef, A_coef = solve_tps(dst, src)

    W_coef_cpu = np.asarray(W_coef, dtype=np.float64)
    A_coef_cpu = np.asarray(A_coef, dtype=np.float64)
    dst_cpu    = np.asarray(dst,    dtype=np.float64)
    image_cpu  = np.asarray(image,  dtype=np.float32)

    x = np.arange(W_img, dtype=np.float64)
    y = np.arange(H,     dtype=np.float64)
    cols, rows = np.meshgrid(x, y)
    grid_cpu = np.stack([rows.ravel(), cols.ravel()], axis=1)

    n_pixels = H * W_img
    src_coords = np.empty((n_pixels, 2), dtype=np.float64)

    for start in range(0, n_pixels, chunk_size):
        end   = min(start + chunk_size, n_pixels)
        chunk = grid_cpu[start:end]

        g_sq  = np.sum(chunk**2,   axis=1, keepdims=True)
        d_sq  = np.sum(dst_cpu**2, axis=1)
        r_sq  = np.clip(g_sq + d_sq - 2.0 * np.dot(chunk, dst_cpu.T), 0.0, None)

        Kgrid = tps_kernel_cpu(r_sq)
        P     = np.hstack([np.ones((chunk.shape[0], 1), dtype=np.float64), chunk])
        src_coords[start:end] = np.dot(Kgrid, W_coef_cpu) + np.dot(P, A_coef_cpu)

    src_row = src_coords[:, 0].reshape(H, W_img).astype(np.float32)
    src_col = src_coords[:, 1].reshape(H, W_img).astype(np.float32)

    warped = bilinear_interpolate_cpu(image_cpu, src_col, src_row)

    print(f"apply_tps_cpu exécuté en {time.time() - st:.3f} s")
    return warped

def compute_displacement_map_cpu(image: np.ndarray, src: np.ndarray, dst: np.ndarray,
                                  chunk_size: int = 80_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    st = time.time()
    H, W = image.shape[:2]

    W_coef, A_coef = solve_tps(dst, src)
    print(f"compute_displacement_map : solve_tps - {time.time()-st:.3f} s")

    W_coef_cpu = np.asarray(W_coef, dtype=np.float64)
    A_coef_cpu = np.asarray(A_coef, dtype=np.float64)
    dst_cpu    = np.asarray(dst,    dtype=np.float64)

    x = np.arange(W, dtype=np.float64)
    y = np.arange(H, dtype=np.float64)
    cols, rows = np.meshgrid(x, y)
    grid_cpu = np.stack([rows.ravel(), cols.ravel()], axis=1)

    n_pixels = H * W
    src_coords = np.empty((n_pixels, 2), dtype=np.float64)

    t = time.time()
    for start in range(0, n_pixels, chunk_size):
        end = min(start + chunk_size, n_pixels)
        chunk = grid_cpu[start:end]

        g_sq = np.sum(chunk**2, axis=1, keepdims=True)
        d_sq = np.sum(dst_cpu**2, axis=1)
        r_sq = np.clip(g_sq + d_sq - 2.0 * np.dot(chunk, dst_cpu.T), 0.0, None)

        Kgrid = tps_kernel_cpu(r_sq)
        P = np.hstack([np.ones((chunk.shape[0], 1), dtype=np.float64), chunk])
        src_coords[start:end] = np.dot(Kgrid, W_coef_cpu) + np.dot(P, A_coef_cpu)

    print(f"compute_displacement_map : chunks - {time.time()-t:.3f} s")

    t = time.time()
    delta = src_coords - grid_cpu
    
    amplitude = np.linalg.norm(delta, axis=1)
    disp_map = amplitude.reshape(H, W).astype(np.float32)
    
    disp_y = delta[:, 0].reshape(H, W).astype(np.float32)
    disp_x = delta[:, 1].reshape(H, W).astype(np.float32)
    
    print(f"compute_displacement_map : delta+norm - {time.time()-t:.3f} s")
    print(f"compute_displacement_map total : {time.time()-st:.3f} s")
    
    return disp_map, disp_y, disp_x


# ─────────────────────────────────────────────────────────────
#  2. AFFICHAGE (IDENTIQUE)
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
    Applique la déformation TPS sur une image de manière automatique (CPU).
    """
    print(f"⚙  Calcul de la déformation TPS ({len(src_points)} points de contrôle)…")
    warped = apply_tps_cpu(image, src_points, dst_points)
    print("✓  Déformation calculée.")

    print("⚙  Calcul de la carte de déplacement…")
    disp_map, disp_y, disp_x = compute_displacement_map_cpu(image, src_points, dst_points)
    print(f"✓  Déplacement max : {disp_map.max():.1f} px  |  moyen : {disp_map.mean():.1f} px")

    # plot_tps_comparison(image, warped, disp_map, src_points, dst_points)

    return warped, disp_y, disp_x