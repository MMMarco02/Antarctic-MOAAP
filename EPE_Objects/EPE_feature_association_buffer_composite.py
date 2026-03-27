# This script does the same as EPE_feature_association_buffer_difference_2.py (meaning you first have to run 
# EPE_feature_association_buffer_difference_1.py for each model) but instead of plotting each model separately, 
# it creates a composite figure with all three models side by side for easier comparison. 
# The code is structured to avoid repetition by looping over models and features, and it uses shared color limits for all subplots to ensure comparability.

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
from scipy import stats
from matplotlib.colors import BoundaryNorm


# ── Helper: build rectangle vertices (for plotting domain boundaries) ──────────────────────────────────────────
def build_rectangle_vertices(rlon, rlat):
    rlon = np.asarray(rlon)
    rlat = np.asarray(rlat)
    if rlon.ndim == 1 and rlat.ndim == 1:
        rlon, rlat = np.meshgrid(rlon, rlat)
    top    = np.column_stack([rlon[-1, :], rlat[-1, :]])
    bottom = np.column_stack([rlon[ 0, :], rlat[ 0, :]])
    left   = np.column_stack([rlon[:, 0],  rlat[:, 0]])
    right  = np.column_stack([rlon[:, -1], rlat[:, -1]])
    return np.vstack([bottom, right, top[::-1], left[::-1]])


# ── CONFIG ─────────────────────────────────────────────────────────────────────
percentile  = 999
P_THRESHOLD = 0.05
mode        = 'absolute'    # 'absolute' or 'relative'

# Columns → models
models       = ['HCLIM', 'MetUM', 'RACMO2']
model_labels = ['HCLIM', 'MetUM', 'RACMO']   # column header text

# Rows → features
features       = ['AR', 'CY', 'ACY', 'FR', 'JET', 'OTHER']
feature_labels = ['(a) ARs', '(b) CYs', '(c) ACYs', '(d) FRs', '(e) JETs', '(f) OTHER']

# Folders obtained from the previous script (EPE_feature_association_buffer_difference_1.py)
diff_folders = {
    'HCLIM':  os.path.abspath(f'EPE_Significance_Maps_HCLIM_vs_ERA5_{percentile}'),
    'MetUM':  os.path.abspath(f'EPE_Significance_Maps_MetUM_vs_ERA5_{percentile}'),
    'RACMO2': os.path.abspath(f'EPE_Significance_Maps_RACMO2_vs_ERA5_{percentile}'),
}

out_folder = os.path.abspath(f'EPE_Composite_Maps_{percentile}')
os.makedirs(out_folder, exist_ok=True)


# ── 1. Load Grid Info ──────────────────────────────────────────────────────────
domain_file       = xr.open_dataset('PolarRES_WP3_Antarctic_domain_expanded.nc')
inner_domain_file = xr.open_dataset('PolarRES_WP3_Antarctic_domain.nc')

rlon       = domain_file['rlon'].values
rlat       = domain_file['rlat'].values
rlon_inner = inner_domain_file['rlon'].values
rlat_inner = inner_domain_file['rlat'].values
rlon2D, rlat2D = np.meshgrid(rlon, rlat)

verts_outer = build_rectangle_vertices(rlon, rlat)
verts_inner = build_rectangle_vertices(rlon_inner, rlat_inner)

rotated_pole = domain_file['rotated_pole'].attrs
rotated_crs  = ccrs.RotatedPole(
    pole_longitude=rotated_pole['grid_north_pole_longitude'],
    pole_latitude =rotated_pole['grid_north_pole_latitude'],
)

inner_domain_file.close()
domain_file.close()


# ── 2. Load all yearly data up-front ──────────────────────────────────────────
print("📂 Loading yearly results for all models ...")
model_data = {}
era5_data  = {}

for model in models:
    folder = diff_folders[model]
    model_data[model] = np.load(f'{folder}/EPE_Yearly_Results_{model}_{percentile}.npz')
    era5_data[model]  = np.load(f'{folder}/EPE_Yearly_Results_ERA5_{percentile}.npz')
    print(f"   ✓ {model}")


# ── 3. Colourmap / normalisation ───────────────────────────────────────────────
cmap = plt.cm.RdBu_r
if mode == 'absolute':
    levels = np.array([-25, -20, -15, -10, -5, -2.5, 2.5, 5, 10, 15, 20, 25])
else:
    levels = np.array([-100, -80, -60, -40, -20, -10, 10, 20, 40, 60, 80, 100])
norm = BoundaryNorm(levels, ncolors=cmap.N, extend='both')


# ── 4. Build the composite figure (6 rows × 3 columns) ────────────────────────
n_rows = len(features)  
n_cols = len(models)    

fig, axs = plt.subplots(
    n_rows, n_cols,
    figsize=(22, 36),
    subplot_kw={'projection': ccrs.Orthographic(0, -90)},
)

mesh_handle = None

for col, model in enumerate(models):
    print(f"\n🔄 Processing model: {model}")
    h_data = model_data[model]
    e_data = era5_data[model]

    for row, feat in enumerate(features):
        print(f"   ... {feat}")
        ax = axs[row, col]

        h_years = h_data[feat]   
        e_years = e_data[feat]

        # ── Pixel-wise Wilcoxon signed-rank test ──────────────────────────────
        ny, nx = h_years.shape[1:]
        p_map = np.ones((ny, nx))

        for y in range(ny):
            for x in range(nx):
                diff   = h_years[:, y, x] - e_years[:, y, x]
                d_test = diff[diff != 0]
                if len(d_test) >= 8:
                    try:
                        _, p = stats.wilcoxon(h_years[:, y, x], e_years[:, y, x], mode='approx')
                        p_map[y, x] = p
                    except Exception:
                        p_map[y, x] = 1.0

        # ── Difference maps ───────────────────────────────────────────────────
        h_mean = np.nanmean(h_years, axis=0)
        e_mean = np.nanmean(e_years, axis=0)

        if mode == 'absolute':
            diff_map = h_mean - e_mean
        else:
            diff_map = np.divide(
                h_mean - e_mean, e_mean,
                out=np.zeros_like(e_mean), where=e_mean > 1.0,
            ) * 100

        # ── Axes extent & coastlines ──────────────────────────────────────────
        ax.set_extent([-180, 180, -90, -20], crs=ccrs.PlateCarree())
        ax.coastlines(resolution='50m', linewidth=0.8)

        # ── Domain boxes ──────────────────────────────────────────────────────
        ax.plot(verts_outer[:, 0], verts_outer[:, 1],
                color='#EABB0F', linewidth=2, transform=rotated_crs)
        ax.plot(verts_inner[:, 0], verts_inner[:, 1],
                color='k', linewidth=1.5, transform=rotated_crs)

        # ── Data ──────────────────────────────────────────────────────────────
        mesh_handle = ax.pcolormesh(
            rlon2D, rlat2D, diff_map,
            transform=rotated_crs,
            cmap=cmap, norm=norm,
            shading='auto',
        )

        # ── Hatch non-significant pixels ──────────────────────────────────────
        insig_mask = p_map > P_THRESHOLD
        ax.contourf(
            rlon2D, rlat2D, insig_mask,
            levels=[0.5, 1.5],
            transform=rotated_crs,
            hatches=['////'], colors='none', alpha=0,
        )

        # ── Model name on top of each column (first row only) ─────────────────
        if row == 0:
            ax.set_title(model_labels[col], fontsize=20, fontweight='bold', pad=6)

        # ── Feature label on the left of each row (first column only) ─────────
        if col == 0:
            ax.text(
                -0.08, 0.5, feature_labels[row],
                transform=ax.transAxes,
                fontsize=20, fontweight='bold',
                va='center', ha='right', rotation=90,
            )


# ── 5. Shared colorbar ────────────────────────────────────────────────────────
cbar = fig.colorbar(
    mesh_handle, ax=axs,
    orientation='horizontal',
    extend='both',
    fraction=0.02,
    pad=0.03,
    aspect=40,
)
label_text = (
    'Difference in Fraction of EPEs associated with Features (% points)'
    if mode == 'absolute' else
    'Relative Difference (%)'
)
cbar.set_label(label_text, fontsize=20)
cbar.ax.tick_params(labelsize=14)


# ── 6. Save ───────────────────────────────────────────────────────────────────
save_path = os.path.join(out_folder, f'Composite_{mode.capitalize()}_Diff_Significance_{percentile}.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"\n🎉 Done! Composite figure saved to:\n   {save_path}")