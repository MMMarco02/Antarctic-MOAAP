import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import cartopy.crs as ccrs
import os
import sys
from dask.diagnostics import ProgressBar



# Clear system-wide Python packages (optional)
sys.path = [p for p in sys.path if '/cluster/software/' not in p]

# Ensure conda env packages are first (guarded)
if 'CONDA_PREFIX' in os.environ:
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    sys.path.insert(0, os.path.join(os.environ['CONDA_PREFIX'], 'lib', pyver, 'site-packages'))

# Disable user site packages
os.environ['PYTHONNOUSERSITE'] = '1'

print("All imports successful")


#################################################################################################

# Function to compute multi-year frequency map from yearly mask files
def compute_multi_year_frequency(mask_files, chunks={'time': 1000}):

    total_files = len(mask_files)
    print(f"Opening {total_files} yearly mask files lazily with dask...")

    # Open all files lazily
    ds_all = xr.open_mfdataset(mask_files, combine='by_coords', parallel=True, chunks=chunks)
    mask = ds_all['EPE_mask']

    print(f"Dataset opened lazily. Total timesteps: {mask.sizes['time']}")
    print("Computing multi-year frequency map (this may take a while)...")

    # Compute frequency lazily, showing a progress bar
    with ProgressBar():
        freq = (mask.astype(float).sum(dim='time') / mask.sizes['time']).compute()

    print("Frequency map computation complete!")
    return freq


# Function to plot frequency map, currently not used
def plot_frequency_map(freq, rlon, rlat, rotated_pole=None, out_file=None):

    print(f"Plotting frequency map to {out_file}...")

    # Convert frequency to percentage and days/year
    freq_percent = freq.values * 100
    freq_days = freq.values * 24 * 20 # years from 2001 to 2020 inclusive

    if rotated_pole is not None:
        rotated_crs = ccrs.RotatedPole(
            pole_longitude=float(rotated_pole.attrs['grid_north_pole_longitude']),
            pole_latitude=float(rotated_pole.attrs['grid_north_pole_latitude'])
        )
    else:
        rotated_crs = ccrs.PlateCarree()

    X, Y = np.meshgrid(rlon, rlat)
    fig, ax = plt.subplots(figsize=(12,8), subplot_kw={'projection': ccrs.Orthographic(0,-90)})
    ax.set_extent([-180, 180, -90, -10], crs=ccrs.PlateCarree())
    ax.coastlines(color="#969696")
    gl = ax.gridlines(draw_labels=False, xlocs=[-180,-120,-60,0,60,120,180],
                      ylocs=[-90,-75,-60,-45,-30], color="gray", linewidth=0.5, alpha=0.7)
    gl.xlabel_style = {"size":8}; gl.ylabel_style = {"size":8}

    # Outer domain boundary
    top = np.column_stack([rlon, np.full_like(rlon, rlat[-1])])
    bot = np.column_stack([rlon, np.full_like(rlon, rlat[0])])
    left = np.column_stack([np.full_like(rlat, rlon[0]), rlat])
    right = np.column_stack([np.full_like(rlat, rlon[-1]), rlat])
    outer = np.vstack([bot, right, top[::-1], left[::-1]])
    ax.fill(outer[:, 0], outer[:, 1], facecolor="none", edgecolor="#EABB0F",
            linewidth=3, transform=rotated_crs, zorder=20)

    # Inner domain
    inner_domain = xr.open_dataset("/cluster/work/users/mmarco02/moaap_project/PolarRES_WP3_Antarctic_domain.nc")
    rlon_inner = inner_domain["rlon"].values
    rlat_inner = inner_domain["rlat"].values
    if rlon_inner is not None and rlat_inner is not None:
        top_i = np.column_stack([rlon_inner, np.full_like(rlon_inner, rlat_inner[-1])])
        bot_i = np.column_stack([rlon_inner, np.full_like(rlon_inner, rlat_inner[0])])
        left_i = np.column_stack([np.full_like(rlat_inner, rlon_inner[0]), rlat_inner])
        right_i = np.column_stack([np.full_like(rlat_inner, rlon_inner[-1]), rlat_inner])
        inner = np.vstack([bot_i, right_i, top_i[::-1], left_i[::-1]])
        ax.plot(inner[:, 0], inner[:, 1], color="red", linewidth=2, transform=rotated_crs)


    pcm = ax.pcolormesh(X, Y, freq_percent, cmap='hot_r', shading='auto', transform=rotated_crs)

    # --- LEFT COLORBAR (%) ---
    cbar1 = plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    cbar1.set_label("Frequency of EPE detection (%)")

    # -- RIGHT COLORBAR (days/year) ---
    cbar2 = plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.10)
    cbar2.set_label("Frequency (days per year)")

    # Convert current % ticks → days/year units
    ticks_percent = cbar2.get_ticks()
    ticks_days = ticks_percent / 100 * 24 * 21
    cbar2.set_ticks(ticks_percent)
    cbar2.set_ticklabels([f"{t:.1f}" for t in ticks_days])


    ax.set_title("Multi-year EPE Frequency Climatology 2001-2020")
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved multi-year frequency map: {out_file}")


def add_map_decorations(ax, rlon, rlat, rotated_crs):
    """Helper to add coastlines, gridlines, and domain boxes to an axis."""
    ax.set_extent([-180, 180, -90, -10], crs=ccrs.PlateCarree())
    ax.coastlines(color="#969696")
    gl = ax.gridlines(draw_labels=False, xlocs=[-180,-120,-60,0,60,120,180],
                      ylocs=[-90,-75,-60,-45,-30], color="gray", linewidth=0.5, alpha=0.7)
    gl.xlabel_style = {"size":10}; gl.ylabel_style = {"size":10}

    # Outer domain boundary
    top = np.column_stack([rlon, np.full_like(rlon, rlat[-1])])
    bot = np.column_stack([rlon, np.full_like(rlon, rlat[0])])
    left = np.column_stack([np.full_like(rlat, rlon[0]), rlat])
    right = np.column_stack([np.full_like(rlat, rlon[-1]), rlat])
    outer = np.vstack([bot, right, top[::-1], left[::-1]])
    ax.fill(outer[:, 0], outer[:, 1], facecolor="none", edgecolor="#EABB0F",
            linewidth=2, transform=rotated_crs, zorder=20)

# Functionto plot EPE Objects frequency maps for both percentiles side by side with a discrete, non-linear color scale. 
# Currently used.
def plot_dual_frequency(freq99, freq999, rlon, rlat, rotated_pole, out_file):
    
    rotated_crs = ccrs.RotatedPole(
        pole_longitude=float(rotated_pole.attrs['grid_north_pole_longitude']),
        pole_latitude=float(rotated_pole.attrs['grid_north_pole_latitude'])
    )

    X, Y = np.meshgrid(rlon, rlat)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), 
                             subplot_kw={'projection': ccrs.Orthographic(0,-90)})
    
    percentiles = ["99th Percentile", "99.9th Percentile"]
    data_list = [freq99.values * 100, freq999.values * 100] 

    # --- DEFINE DISCRETE BOUNDARIES ---
    # We create bins that are small at the start and larger at the end
    # This "zooms in" on the low frequencies of the 99.9th
    bounds = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5]
    
    # Get a colormap and create a normalization based on these bounds
    cmap = plt.get_cmap('hot_r')
    norm = BoundaryNorm(bounds, cmap.N, extend='both')

    for i, ax in enumerate(axes):
        add_map_decorations(ax, rlon, rlat, rotated_crs)
        
        # Plot data using the discrete normalization
        pcm = ax.pcolormesh(X, Y, data_list[i], cmap=cmap, 
                           shading='auto', transform=rotated_crs, 
                           norm=norm)
        
        ax.set_title(f"{percentiles[i]}", fontsize=18, fontweight='bold', pad=15)

    # --- CREATE DISCRETE COLORBAR ---
    cbar = fig.colorbar(pcm, ax=axes, orientation='horizontal', 
                        fraction=0.05, pad=0.08, aspect=35, 
                        ticks=bounds, spacing='uniform')
    
    cbar.set_label("EPE Objects Frequency (%)", fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved discrete non-linear dual plot: {out_file}")

# --------------------
# Main Execution
# --------------------
if __name__ == "__main__":
    years = range(2001, 2021) # Then 2001 to 2020 inclusive
    data_type = "ERA5" # ERA5 or HCLIM or MetUM or RACMO2
    
    # Domain parameters
    domain_ds = xr.open_dataset("PolarRES_WP3_Antarctic_domain_expanded.nc")
    rlat = domain_ds["rlat"].values
    rlon = domain_ds["rlon"].values
    rotated_pole = domain_ds["rotated_pole"]

    # Calculate frequencies for both
    freq_results = {}

    for p in [99, 999]:
        print(f"\n--- Processing {p}th percentile ---")
        if data_type == "ERA5":
            pr_path = [f"moaap_output_ERA5_test_EPE_dry_thr0.1_{p}/{year}/{year}_ERA5_ANT_{year}_EPE_ObjectMasks__dt-1h_MOAAP-masks.nc" for year in years]
            pr_name = "tp"
        else: # HCLIM or RACMO2 or MetUM
            pr_path = [f"moaap_output_{data_type}_test_EPE_dry_thr0.1_{p}/{year}/{year}_{data_type}_mask.nc" for year in years]
            pr_name = "pr"
        freq_results[p] = compute_multi_year_frequency(pr_path)

    # Plot both together
    out_name = f'EPE_frequency_comparison_{data_type}_99_vs_999.png'
    plot_dual_frequency(freq_results[99], freq_results[999], rlon, rlat, rotated_pole, out_name)