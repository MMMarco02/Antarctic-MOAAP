# This snippet contains the main EPE object detection function and a plotting helper for visualizing detected objects on precipitation maps. 
# It is designed to work with the PolarRES and ERA5 datasets, using seasonal thresholds to identify extreme precipitation events. 
# The code includes optimizations for memory usage and processing speed, as well as flexible plotting options.

import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
from scipy.ndimage import gaussian_filter, label
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LogNorm
import cartopy.crs as ccrs
from tqdm import tqdm
import time


# Clear system-wide Python packages (optional)
sys.path = [p for p in sys.path if '/cluster/software/' not in p]

# Ensure conda env packages are first (guarded)
if 'CONDA_PREFIX' in os.environ:
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    sys.path.insert(0, os.path.join(os.environ['CONDA_PREFIX'], 'lib', pyver, 'site-packages'))

# Disable user site packages
os.environ['PYTHONNOUSERSITE'] = '1'

print("All imports successful")

################################################################################################
# ---------- Small helper ----------

def build_thresh_per_time(pr_da, seasonal_thresholds):

    months = pd.DatetimeIndex(pr_da['time'].values).month
    months_to_seasons = {12:'DJF',1:'DJF',2:'DJF',3:'MAM',4:'MAM',5:'MAM',
                         6:'JJA',7:'JJA',8:'JJA',9:'SON',10:'SON',11:'SON'}
    seasons = [months_to_seasons[m] for m in months]
    # stack arrays in order of time
    thresh_list = [ seasonal_thresholds[s] for s in seasons ]
    return np.stack(thresh_list, axis=0)   # shape (time, y, x)


# ---------- Main wrapper ----------

def detect_extreme_precip_objects(
        ncpath,
        seasonal_thresholds,
        varname='pr',           # Default for PolarRES RCMs
        time_name='time',       # Default for PolarRES RCMs
        y_name='rlat', x_name='rlon',
        grid_spacing_km=11.17,  # Approximate for 0.11° at Antarctic latitudes
        gaussian_sigma_px=1.0,
        min_area_km2=25000,
        connectivity=8,
        n_workers=1,
        out_mask_path=None):

    # Load with chunking
    ds = xr.open_dataset(ncpath, chunks={time_name: 200})
    
    # Check units: HCLIM is in 'm', ERA5 might be 'm' or 'mm'. 
    # We convert to mm for consistency if it's in meters.
    pr_da = ds[varname]
    
    pr = pr_da.rename({time_name: 'time', y_name: 'y', x_name: 'x'})

    # Precompute thresholds matching time
    thresh_data = build_thresh_per_time(pr, seasonal_thresholds)
    nt, ny, nx = pr.sizes["time"], pr.sizes["y"], pr.sizes["x"]


    pixel_area_km2 = float(grid_spacing_km)**2
    out = np.zeros((nt, ny, nx), dtype=np.uint8)

    print(f"Detector: processing {nt} timesteps...")

    block_size = 200  # process 200-hour blocks
    for i0 in range(0, nt, block_size):
        print(f"Processing block {i0}–{min(i0 + block_size, nt)}")
        i1 = min(i0 + block_size, nt)
        
        # Load block and handle small precision noise
        pr_block = pr.isel(time=slice(i0, i1)).values.astype(np.float32)
        pr_block[pr_block <= 1e-4] = 0.0
        thresh_block = thresh_data[i0:i1].astype(np.float32)

        # Smooth spatial dims
        pr_smooth_block = gaussian_filter(
            pr_block, sigma=(0, gaussian_sigma_px, gaussian_sigma_px), mode='nearest'
        )

        thresholded_block = (pr_smooth_block > thresh_block).astype(np.uint8)

        for t_local in range(i1 - i0):
            binary = thresholded_block[t_local]
            if binary.sum() == 0: continue

            struct = np.ones((3, 3)) if connectivity == 8 else np.array([[0,1,0],[1,1,1],[0,1,0]])
            labeled, nfeat = label(binary, structure=struct)

            if nfeat == 0: continue

            mask2d = np.zeros_like(binary, dtype=np.uint8)
            # Optimization: use np.unique with counts to filter labels quickly
            lab_ids, counts = np.unique(labeled[labeled > 0], return_counts=True)
            keep_ids = lab_ids[(counts * pixel_area_km2) >= min_area_km2]
            
            if keep_ids.size > 0:
                mask2d = np.isin(labeled, keep_ids).astype(np.uint8)
                out[i0 + t_local] = mask2d

    da_mask = xr.DataArray(
        out,
        coords={'time': pr['time'].values, 'y': pr['y'].values, 'x': pr['x'].values},
        dims=('time', 'y', 'x'),
        name='EPE_mask'
    )

    if out_mask_path:
        os.makedirs(os.path.dirname(out_mask_path), exist_ok=True)
        da_mask.to_dataset().to_netcdf(out_mask_path)
    
    return da_mask

# This function used to detect EPE_Objects for ERA5, but we now have a more optimized version above that can handle all datasets.
def detect_extreme_precip_objects_era5(
        ncpath,
        seasonal_thresholds,
        varname='tp',
        time_name='valid_time',
        y_name='rlat', x_name='rlon',
        grid_spacing_km=11.1,
        gaussian_sigma_px=1.0,
        min_area_km2=25000,
        connectivity=8,
        n_workers=1,
        out_mask_path=None):

    # Load with chunking to avoid loading 51 GB all at once
    ds = xr.open_dataset(ncpath, chunks={'valid_time': 200})
    pr = ds[varname].rename({time_name: 'time', y_name: 'y', x_name: 'x'})

    # Precompute thresholds aligned with time
    thresh_data = build_thresh_per_time(pr, seasonal_thresholds)  # numpy array (nt, y, x)
    nt = pr.sizes["time"]
    ny = pr.sizes["y"]
    nx = pr.sizes["x"]

    pixel_area_km2 = float(grid_spacing_km)**2
    out = np.zeros((nt, ny, nx), dtype=np.uint8)

    print(f"Detector: dataset opened with chunking, nt={nt}")

    block_size = 200  # process 200-hour blocks

    # --- PROCESS DATA BLOCK-BY-BLOCK ---
    for i0 in range(0, nt, block_size):
        i1 = min(i0 + block_size, nt)

        print(f"Processing block {i0}–{i1}")

        # Load just this block into memory (instead of whole 51 GB)
        pr_block = pr.isel(time=slice(i0, i1)).values.astype(np.float32)
        pr_block[pr_block <= 1e-4] = 0.0
        thresh_block = thresh_data[i0:i1].astype(np.float32)

        # Smooth only spatial dims
        pr_smooth_block = gaussian_filter(
            pr_block,
            sigma=(0, gaussian_sigma_px, gaussian_sigma_px),
            mode='nearest',
            truncate=3.0
        )

        # Prethresholding step (much faster)
        thresholded_block = (pr_smooth_block > thresh_block).astype(np.uint8)

        # --- SERIAL or PARALLEL PER TIMESTEP ---
        if n_workers == 1:
            for t_local in range(i1 - i0):
                binary = thresholded_block[t_local]

                if binary.sum() == 0:
                    continue

                struct = np.ones((3, 3)) if connectivity == 8 else np.array(
                    [[0,1,0],[1,1,1],[0,1,0]], dtype=int
                )

                labeled, nfeat = label(binary, structure=struct)

                if nfeat == 0:
                    continue

                mask2d = np.zeros_like(binary, dtype=np.uint8)
                for label_id in range(1, nfeat + 1):
                    comp = (labeled == label_id)
                    area = comp.sum() * pixel_area_km2
                    if area >= min_area_km2:
                        mask2d[comp] = 1

                out[i0 + t_local] = mask2d

        else:
            # TODO: optional: parallel version if needed
            raise NotImplementedError("Parallel block processing not implemented yet in optimized version.")

    # Wrap into DataArray
    da_mask = xr.DataArray(
        out,
        coords={'time': pr['time'].values, 'y': pr['y'].values, 'x': pr['x'].values},
        dims=('time', 'y', 'x'),
        name='EPE_mask',
        attrs={
            'description': 'Binary mask of extreme precipitation objects (1=object)',
            'min_area_km2': min_area_km2,
            'gaussian_sigma_px': gaussian_sigma_px,
            'pixel_area_km2': pixel_area_km2,
        }
    )

    if out_mask_path:
        da_mask.to_dataset().to_netcdf(out_mask_path, mode='w')
        print(f"Saved mask to {out_mask_path}")

    return da_mask



# ---------- Simple plotting helpers ----------

def plot_time_index_pr_and_contours(
        ncpath, da_mask, time_index=0,
        varname='pr', time_name='time',  
        y_name='rlat', x_name='rlon',
        rotated_pole="rotated_pole",
        convert_to_geographic=True,
        power_norm_gamma=0.35
    ):
    
    print(f"Loading timestep {time_index}...")
    ds = xr.open_dataset(ncpath)
    
    # Unit handling: Convert meters to mm for the plot
    pr_da = ds[varname].isel({time_name: time_index})
    if pr_da.attrs.get('units') == 'm':
        pr2d = (pr_da * 1000.0).values
    else:
        pr2d = pr_da.values
        
    mask2d = da_mask.isel(time=time_index).values

    rlon = ds[x_name].values
    rlat = ds[y_name].values

    # Get projection info
    rp = ds[rotated_pole]
    rotated_crs = ccrs.RotatedPole(
        pole_longitude=float(rp.grid_north_pole_longitude),
        pole_latitude=float(rp.grid_north_pole_latitude)
    )

    inner_domain = xr.open_dataset("PolarRES_WP3_Antarctic_domain.nc")
    rlon_inner = inner_domain["rlon"].values
    rlat_inner = inner_domain["rlat"].values

    # Plotting setup
    fig, ax = plt.subplots(
        figsize=(12, 8),
        subplot_kw={"projection": ccrs.Orthographic(0, -90)}
    )

    ax.set_extent([-180, 180, -90, -10], crs=ccrs.PlateCarree())
    ax.coastlines(color="#969696")

    gl = ax.gridlines(draw_labels=True,
                      xlocs=[-180, -120, -60, 0, 60, 120, 180],
                      ylocs=[-90, -75, -60, -45, -30],
                      color="gray", linewidth=0.5, alpha=0.7)
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}

    # Outer domain boundary
    top = np.column_stack([rlon, np.full_like(rlon, rlat[-1])])
    bot = np.column_stack([rlon, np.full_like(rlon, rlat[0])])
    left = np.column_stack([np.full_like(rlat, rlon[0]), rlat])
    right = np.column_stack([np.full_like(rlat, rlon[-1]), rlat])
    outer = np.vstack([bot, right, top[::-1], left[::-1]])
    ax.fill(outer[:, 0], outer[:, 1], facecolor="none", edgecolor="#EABB0F",
            linewidth=3, transform=rotated_crs, zorder=20)

    # Inner domain 
    inner_domain_file = ""
    if rlon_inner is not None and rlat_inner is not None:
        top_i = np.column_stack([rlon_inner, np.full_like(rlon_inner, rlat_inner[-1])])
        bot_i = np.column_stack([rlon_inner, np.full_like(rlon_inner, rlat_inner[0])])
        left_i = np.column_stack([np.full_like(rlat_inner, rlon_inner[0]), rlat_inner])
        right_i = np.column_stack([np.full_like(rlat_inner, rlon_inner[-1]), rlat_inner])
        inner = np.vstack([bot_i, right_i, top_i[::-1], left_i[::-1]])
        ax.plot(inner[:, 0], inner[:, 1], color="red", linewidth=2, transform=rotated_crs)

    # Precipitation base
    X, Y = np.meshgrid(rlon, rlat)
    pcm = ax.pcolormesh(
        X, Y, pr2d,
        cmap="YlGnBu",
        norm=PowerNorm(gamma=power_norm_gamma),
        shading='auto',
        transform=rotated_crs,
        zorder=0
    )
    cbar = plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Precipitation (mm/h)")

    # Contours of objects
    if np.any(mask2d):
        ax.contour(X, Y, mask2d, levels=[0.5], colors="white", 
                   linewidths=2.0, transform=rotated_crs, zorder=10)

    # Dynamic Title
    time_val = str(ds[time_name].values[time_index])
    ax.set_title(f"Precipitation + EPE Objects\n{time_val}")
    plt.tight_layout()

    return fig, ax


# ---------- Main part ----------

start_time = time.time()

if __name__ == "__main__":

    # Parameters 
    data_type = "RACMO2"  # "HCLIM" or "ERA5" or "MetUM" or "RACMO2"
    pr_base_path = os.abspath("") # Set to current dir or adjust as needed
    years = (range(2001, 2021))  # then 2000 to 2020 inclusive
    sigma = 2.0 # Gaussian smoothing in pixels
    min_area = 50000  # km2
    percentile = 99.9
    
    # Configuration Mapping
    # Note: Define your own paths!
    config = {
        "ERA5": {
            "var": "tp", "time": "valid_time", 
            "in_pattern": "ERA5/{year}/remapped/ANT_pr_{year}_remapped.nc",
            "out_pattern": f"moaap_output_ERA5_test_EPE_dry_thr0.1_{percentile}/{{year}}/{{year}}_ERA5_mask.nc"
        },
        "HCLIM": {
            "var": "pr", "time": "time", 
            "in_pattern": "PolarRES/HCLIM/nested/{year}/ANT_HCLIM_pr_{year}_nested.nc",
            "out_pattern": f"moaap_output_HCLIM_test_EPE_dry_thr0.1_{percentile}/{{year}}/{{year}}_HCLIM_mask.nc"
        },
        "MetUM": {
            "var": "pr", "time": "time", 
            "in_pattern": "PolarRES/MetUM/nested/{year}/ANT_MetUM_pr_{year}_nested.nc",
            "out_pattern": f"moaap_output_MetUM_test_EPE_dry_thr0.1_{percentile}/{{year}}/{{year}}_MetUM_mask.nc"
        },
        "RACMO2": {
            "var": "pr", "time": "time", 
            "in_pattern": "PolarRES/RACMO2/nested/{year}/ANT_RACMO2_pr_{year}_nested_con.nc",
            "out_pattern": f"moaap_output_RACMO2_test_EPE_dry_thr0.1_{percentile}/{{year}}/{{year}}_RACMO2_mask.nc"
        }
    }

    # load seasonal thresholds (must align on y/x grid)
    # percentiles_path = "/cluster/work/users/mmarco02/moaap_project/ERA5/pr_percentile_computation/remapped_dry"
    # Note: Define your own paths!
    if data_type == "ERA5":
        percentiles_path = "/cluster/work/users/mmarco02/moaap_project/ERA5/pr_percentile_computation/seasonal_percentiles_thr0.1/remapped"
        season_files = {
            'DJF': os.path.join(percentiles_path, f'DJF_p{percentile}_remapped.nc'),
            'MAM': os.path.join(percentiles_path, f'MAM_p{percentile}_remapped.nc'),
            'JJA': os.path.join(percentiles_path, f'JJA_p{percentile}_remapped.nc'),
            'SON': os.path.join(percentiles_path, f'SON_p{percentile}_remapped.nc'),
        }

    else:
        percentiles_path = f"/cluster/work/users/mmarco02/moaap_project/ERA5/pr_percentile_computation/seasonal_percentiles_thr0.1_{data_type}"
        season_files = {
            'DJF': os.path.join(percentiles_path, f'ANT_{data_type}_pr_DJF_p{percentile}.nc'),
            'MAM': os.path.join(percentiles_path, f'ANT_{data_type}_pr_MAM_p{percentile}.nc'),
            'JJA': os.path.join(percentiles_path, f'ANT_{data_type}_pr_JJA_p{percentile}.nc'),
            'SON': os.path.join(percentiles_path, f'ANT_{data_type}_pr_SON_p{percentile}.nc'),
        }
    seasonal_thresholds = {}
    for season, path in season_files.items():
        ds = xr.open_dataset(path)
        
        # select variable 'tp'
        da = ds[config[data_type]["var"]]

        # remove any extra singleton dims like 'bnds'
        for dim in da.dims:
            if da[dim].size == 1:
                da = da.squeeze(dim, drop=True)

        # ensure dims are (rlat, rlon)
        da = da.transpose('rlat', 'rlon')

        # store as NumPy array for your detector
        seasonal_thresholds[season] = da.values


    # Plotting parameters (for debugging/visualization purpose)
    plot_mode = True # True/False
    plot_year = 2003  # which year to plot examples for
    plot_start_idx = 5000
    plot_end_idx = 5060
    plot_folder = f"zzz_EPE_plots_{plot_year}_{data_type}_{percentile}"
    os.makedirs(plot_folder, exist_ok=True)

    for year in years:
        input_nc = os.path.join(pr_base_path, config[data_type]["in_pattern"].format(year=year))
        output_nc = os.path.join(pr_base_path, config[data_type]["out_pattern"].format(year=year))

        print(f"Processing {data_type} for year {year}...")
        mask = detect_extreme_precip_objects(
            ncpath=input_nc,
            seasonal_thresholds=seasonal_thresholds,
            varname=config[data_type]["var"],
            time_name=config[data_type]["time"],
            grid_spacing_km=11.17, # Update if HCLIM resolution differs
            gaussian_sigma_px=sigma,
            min_area_km2=min_area,
            out_mask_path=output_nc
        )

        print(f"Year {year} done. Mask shape: {mask.shape}")

        # 2. Plotting (if enabled, for debugging/visualization)
        if plot_year == year and plot_mode:
            print(f"Generating plots for {data_type}...")
            for t_idx in range(plot_start_idx, plot_end_idx + 1):
                fig, ax = plot_time_index_pr_and_contours(
                    ncpath=input_nc,
                    da_mask=mask,
                    time_index=t_idx,
                    varname=config[data_type]["var"],    
                    time_name=config[data_type]["time"], 
                    y_name='rlat',
                    x_name='rlon',
                    rotated_pole="rotated_pole"
                )
                plt.savefig(f"{plot_folder}/EPE_PR_plot_{year}_t{t_idx:03d}.png", dpi=150)
                plt.close(fig)

    
    elapsed = time.time() - start_time
    print(f"Total elapsed time: {elapsed/60:.2f} minutes")

    print("All done.")


