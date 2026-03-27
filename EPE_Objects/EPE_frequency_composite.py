import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import os
import sys
from dask.diagnostics import ProgressBar

# Clear system-wide Python packages (optional)
sys.path = [p for p in sys.path if '/cluster/software/' not in p]
if 'CONDA_PREFIX' in os.environ:
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    sys.path.insert(0, os.path.join(os.environ['CONDA_PREFIX'], 'lib', pyver, 'site-packages'))
os.environ['PYTHONNOUSERSITE'] = '1'


# ── CONFIG ─────────────────────────────────────────────────────────────────────
years       = range(2001, 2021)
models      = ['HCLIM', 'MetUM', 'RACMO2']          
model_labels = {'HCLIM': 'HCLIM', 'MetUM': 'MetUM', 'RACMO2': 'RACMO'}
percentiles = [99, 999]                               
col_labels  = ['99th Percentile', '99.9th Percentile']

out_file = 'EPE_frequency_composite_3models_2percentiles.png'

# Discrete colorbar boundaries (same as your original)
bounds = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5]


# ── HELPERS ────────────────────────────────────────────────────────────────────
def get_mask_paths(model, percentile, years):
    if model == 'ERA5':
        return [
            # f"/cluster/work/users/mmarco02/moaap_project/"
            f"moaap_output_ERA5_test_EPE_dry_thr0.1_{percentile}/"
            f"{year}/{year}_ERA5_ANT_{year}_EPE_ObjectMasks__dt-1h_MOAAP-masks.nc"
            for year in years
        ]
    else:
        return [
            # f"/cluster/work/users/mmarco02/moaap_project/"
            f"moaap_output_{model}_test_EPE_dry_thr0.1_{percentile}/"
            f"{year}/{year}_{model}_mask.nc"
            for year in years
        ]


def compute_frequency(mask_files, chunks={'time': 1000}):

    print(f"   Opening {len(mask_files)} files lazily ...")
    ds = xr.open_mfdataset(mask_files, combine='by_coords', parallel=True, chunks=chunks)
    mask = ds['EPE_mask']
    with ProgressBar():
        freq = (mask.astype(float).sum(dim='time') / mask.sizes['time']).compute()
    return freq


def add_map_decorations(ax, rlon, rlat, rotated_crs):
    """Coastlines, gridlines, and domain boxes."""
    ax.set_extent([-180, 180, -90, -10], crs=ccrs.PlateCarree())
    ax.coastlines(color='#969696')
    gl = ax.gridlines(
        draw_labels=False,
        xlocs=[-180, -120, -60, 0, 60, 120, 180],
        ylocs=[-90, -75, -60, -45, -30],
        color='gray', linewidth=0.5, alpha=0.7,
    )
    gl.xlabel_style = {'size': 7}
    gl.ylabel_style = {'size': 7}

    # Outer domain box (yellow)
    top   = np.column_stack([rlon, np.full_like(rlon, rlat[-1])])
    bot   = np.column_stack([rlon, np.full_like(rlon, rlat[0])])
    left  = np.column_stack([np.full_like(rlat, rlon[0]),  rlat])
    right = np.column_stack([np.full_like(rlat, rlon[-1]), rlat])
    outer = np.vstack([bot, right, top[::-1], left[::-1]])
    ax.fill(outer[:, 0], outer[:, 1], facecolor='none', edgecolor='#EABB0F',
            linewidth=2, transform=rotated_crs, zorder=20)


# ── 1. Load domain grid ────────────────────────────────────────────────────────
domain_ds    = xr.open_dataset('PolarRES_WP3_Antarctic_domain_expanded.nc')
rlat         = domain_ds['rlat'].values
rlon         = domain_ds['rlon'].values
rotated_pole = domain_ds['rotated_pole']
rotated_crs  = ccrs.RotatedPole(
    pole_longitude=float(rotated_pole.attrs['grid_north_pole_longitude']),
    pole_latitude =float(rotated_pole.attrs['grid_north_pole_latitude']),
)
X, Y = np.meshgrid(rlon, rlat)
domain_ds.close()


# ── 2. Pre-compute all frequency maps ─────────────────────────────────────────
print('\n📂 Computing frequency maps for all model/percentile combinations ...')
freq_data = {}   # freq_data[(model, percentile)] = freq DataArray

for model in models:
    for p in percentiles:
        print(f'\n🔄  {model}  |  {p}th percentile')
        paths = get_mask_paths(model, p, years)
        freq_data[(model, p)] = compute_frequency(paths)

print('\n All frequency maps ready.')


# ── 3. Build composite figure (3 rows × 2 columns) ────────────────────────────
n_rows = len(models)      # 3
n_cols = len(percentiles) # 2

cmap = plt.get_cmap('hot_r')
norm = BoundaryNorm(bounds, cmap.N, extend='both')

fig, axs = plt.subplots(
    n_rows, n_cols,
    figsize=(18, 24),
    subplot_kw={'projection': ccrs.Orthographic(0, -90)},
)

mesh_handle = None

for row, model in enumerate(models):
    for col, p in enumerate(percentiles):
        ax = axs[row, col]
        freq_pct = freq_data[(model, p)].values * 100   # convert to %

        add_map_decorations(ax, rlon, rlat, rotated_crs)

        mesh_handle = ax.pcolormesh(
            X, Y, freq_pct,
            cmap=cmap, norm=norm,
            shading='auto', transform=rotated_crs,
        )

        # Column header: percentile label on top row only
        if row == 0:
            ax.set_title(col_labels[col], fontsize=18, fontweight='bold', pad=8)

        # Row label: model name on left column only
        if col == 0:
            ax.text(
                -0.08, 0.5, model_labels[model],
                transform=ax.transAxes,
                fontsize=18, fontweight='bold',
                va='center', ha='right', rotation=90,
            )


# ── 4. Single shared colorbar ─────────────────────────────────────────────────
cbar = fig.colorbar(
    mesh_handle, ax=axs,
    orientation='horizontal',
    extend='both',
    fraction=0.03,
    pad=0.04,
    aspect=40,
    ticks=bounds,
    spacing='uniform',
)
cbar.set_label('EPE Objects Frequency (%)', fontsize=18)
cbar.ax.tick_params(labelsize=13)


# ── 5. Save ───────────────────────────────────────────────────────────────────
plt.savefig(out_file, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'\n Done! Saved to: {out_file}')