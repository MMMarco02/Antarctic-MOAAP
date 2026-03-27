# Script to compute mean annual precipitation differences between RCMs and ERA5, and plot the results in a 6-panel figure.

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import os
import matplotlib.colors as mcolors

# --- Configuration ---
# base_dir = "/cluster/work/users/mmarco02/moaap_project" # Configure this to your base directory
base_dir = os.path.abspath(".")
start_year = 2001
end_year = 2020
years = range(start_year, end_year + 1)

# File patterns
era5_pattern   = "ERA5/{year}/remapped/ANT_pr_{year}_remapped.nc"
hclim_pattern  = "PolarRES/HCLIM/nested/{year}/ANT_HCLIM_pr_{year}_nested.nc"
metum_pattern  = "PolarRES/MetUM/nested/{year}/ANT_MetUM_pr_{year}_nested.nc"
racmo2_pattern = "PolarRES/RACMO2/nested/{year}/ANT_RACMO2_pr_{year}_nested_con.nc"

output_plot_file = f"precip_comparison_HCLIM_MetUM_RACMO_vs_ERA5_{start_year}-{end_year}.png"

POLE_LAT = 5.0
POLE_LON = 20.0

# Function to extract and compute mean annual precipitation 
def get_mean_annual_precip(base_dir, years, file_pattern, var_name, time_dim_name):
    annual_sums = []
    for year in years:
        fpath = os.path.join(base_dir, file_pattern.format(year=year))
        print(f"  Processing {year}...", flush=True)
        ds = xr.open_dataset(fpath, chunks={time_dim_name: 100})  # dask, 100 steps at a time
        if time_dim_name != 'time':
            ds = ds.rename_dims({time_dim_name: 'time'}).rename({time_dim_name: 'time'})
        annual_sum = (ds[var_name] * 1000).sum(dim='time').compute().values  # compute year by year
        annual_sums.append(annual_sum)
        ds.close()
    return np.mean(annual_sums, axis=0)

# Function to plot the 6-panel comparison
def plot_six_panel_comparison(hclim_da, metum_da, racmo2_da, era5_da, rlon, rlat, output_file, pole_lat, pole_lon):
    # 1. Calculations
    def get_diffs(rcm, ref):
        abs_diff = rcm - ref
        rel_diff = (abs_diff / np.where(ref > 0, ref, np.nan)) * 100
        return abs_diff, rel_diff

    hclim_abs,  hclim_rel  = get_diffs(hclim_da,  era5_da)
    metum_abs,  metum_rel  = get_diffs(metum_da,  era5_da)
    racmo2_abs, racmo2_rel = get_diffs(racmo2_da, era5_da)

    rotated_crs = ccrs.RotatedPole(pole_latitude=pole_lat, pole_longitude=pole_lon)

    # Inner boundary
    inner_domain_file = xr.open_dataset("PolarRES_WP3_Antarctic_domain.nc")
    rlon_inner = inner_domain_file['rlon'].values
    rlat_inner = inner_domain_file['rlat'].values
    inner_domain_file.close()

    top_i  = np.column_stack([rlon_inner, np.full_like(rlon_inner, rlat_inner[-1])])
    bot_i  = np.column_stack([rlon_inner, np.full_like(rlon_inner, rlat_inner[0])])
    left_i = np.column_stack([np.full_like(rlat_inner, rlon_inner[0]),  rlat_inner])
    right_i= np.column_stack([np.full_like(rlat_inner, rlon_inner[-1]), rlat_inner])
    inner_boundary_path = np.vstack([bot_i, right_i, top_i[::-1], left_i[::-1]])

    # 2. Setup Figure
    fig, axes = plt.subplots(
        2, 3, figsize=(26, 20),
        subplot_kw={"projection": ccrs.Orthographic(0, -90)}
    )
    plt.subplots_adjust(bottom=0.15, top=0.95, hspace=0.4, wspace=0.05)

    # Styling
    cmap     = plt.get_cmap("BrBG")
    levs_abs = [-1000, -500, -250, -100, -50, -20, -10, 10, 20, 50, 100, 250, 500, 1000]
    levs_rel = [-100, -80, -60, -40, -20, -10, 10, 20, 40, 60, 80, 100]
    norm_abs = mcolors.BoundaryNorm(boundaries=levs_abs, ncolors=cmap.N, extend='both')
    norm_rel = mcolors.BoundaryNorm(boundaries=levs_rel, ncolors=cmap.N, extend='both')

    data_to_plot = [
        (hclim_abs,  norm_abs, "HCLIM"),
        (metum_abs,  norm_abs, "MetUM"),
        (racmo2_abs, norm_abs, "RACMO"),
        (hclim_rel,  norm_rel, "HCLIM"),
        (metum_rel,  norm_rel, "MetUM"),
        (racmo2_rel, norm_rel, "RACMO"),
    ]

    plot_handles = []
    for i, ax in enumerate(axes.flat):
        data, norm, title = data_to_plot[i]
        ax.set_extent([-180, 180, -90, -10], crs=ccrs.PlateCarree())
        ax.coastlines(color="#4d4d4d", resolution='50m', linewidth=0.8)
        pcm = ax.pcolormesh(rlon, rlat, data, cmap=cmap, norm=norm, transform=rotated_crs, shading='auto')
        ax.plot(inner_boundary_path[:, 0], inner_boundary_path[:, 1],
                color="red", linewidth=2.5, transform=rotated_crs, zorder=10)
        
        ax.set_title(title, fontsize=20, fontweight='bold', pad=20) if i < 3 else ax.set_title('', fontsize=20, pad=20)
        plot_handles.append(pcm)

    # 3. Colorbars
    cax1 = fig.add_axes([0.25, 0.54, 0.5, 0.015])
    cb1  = fig.colorbar(plot_handles[0], cax=cax1, orientation='horizontal', extend='both')
    cb1.set_label("Absolute Difference (mm/year)", fontsize=20, labelpad=10)
    cb1.ax.tick_params(labelsize=14)

    cax2 = fig.add_axes([0.25, 0.10, 0.5, 0.015])
    cb2  = fig.colorbar(plot_handles[3], cax=cax2, orientation='horizontal', extend='both')
    cb2.set_label("Relative Difference (%)", fontsize=20, labelpad=10)
    cb2.ax.tick_params(labelsize=14)

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {output_file}")


# --- Main Execution ---
if __name__ == "__main__":
    print("Loading ERA5...")
    era5   = get_mean_annual_precip(base_dir, years, era5_pattern,   'tp', 'valid_time')
    print("Loading HCLIM...")
    hclim  = get_mean_annual_precip(base_dir, years, hclim_pattern,  'pr', 'time')
    print("Loading MetUM...")
    metum  = get_mean_annual_precip(base_dir, years, metum_pattern,  'pr', 'time')
    print("Loading RACMO2...")
    racmo2 = get_mean_annual_precip(base_dir, years, racmo2_pattern, 'pr', 'time')

    # Extract grid coordinates from one HCLIM file
    _ds = xr.open_dataset(os.path.join(base_dir, hclim_pattern.format(year=start_year)))
    rlon, rlat = _ds['rlon'].values, _ds['rlat'].values
    _ds.close()

    print("Generating 6-panel comparison...")
    plot_six_panel_comparison(hclim, metum, racmo2, era5, rlon, rlat, output_plot_file, POLE_LAT, POLE_LON)
