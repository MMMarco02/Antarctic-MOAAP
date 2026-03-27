import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
from scipy.ndimage import gaussian_filter, label
from concurrent.futures import as_completed
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LogNorm
import cartopy.crs as ccrs
from tqdm import tqdm
import time


print("All imports successful")

################################################################################################
# ----------  Helper Functions ----------

# This function sums hourly precipitation into 6-hour bins centered at 00, 06, 12, and 18 UTC.
# Specifically:
    #00 = sum(22,23,00,01,02,03)
    #06 = sum(04,05,06,07,08,09)
    #12 = sum(10,11,12,13,14,15)
    #18 = sum(16,17,18,19,20,21)

def sum_to_6h_centered(pr_hourly):
    """
    Optimized version using rolling window.
    Sum hourly precipitation into 6-hour bins.
    Window for center 12 (sum 10,11,12,13,14,15) ends at 15.
    """
    # 1. Sort and Ensure hourly frequency (fills gaps with NaN/0)
    pr = pr_hourly.sortby("time")
    
    # Optional: Fill missing hours if your time axis is irregular
    # pr = pr.resample(time="1H").asfreq().fillna(0)

    # 2. Calculate rolling 6-hour sum
    # By default, rolling(time=6) puts the value at the right edge of the window.
    # E.g., at t=15, it sums [10, 11, 12, 13, 14, 15].
    pr_rolled = pr.rolling(time=6, center=False).sum()

    # 3. We want centers at 00, 06, 12, 18.
    # The rolling sum for center 12 ends at 15.
    # The rolling sum for center 00 ends at 03.
    # So we select times where hour % 6 == 3.
    
    # Filter for times 03, 09, 15, 21
    is_target_end_time = pr_rolled.time.dt.hour % 6 == 3
    pr_selection = pr_rolled.isel(time=is_target_end_time)

    # 4. Shift the time labels back by 3 hours to match the center
    # 15:00 becomes 12:00
    new_times = pr_selection.time - pd.Timedelta(hours=3)
    pr_selection = pr_selection.assign_coords(time=new_times)

    return pr_selection



# Function to detect 6-hourlyprecipitation objects in ERA5 data (similar to RCM but with ERA5-specific handling)
def detect_precip_objects_era5(
        ncpath,
        varname='tp',
        time_name='valid_time',
        y_name='rlat', x_name='rlon',
        grid_spacing_km=11.17,
        gaussian_sigma_px=1.0,
        min_area_km2=25000, 
        accum_threshold=0.0006, # masking threshold in meters (0.0006 m = 0.6 mm)
        connectivity=8,
        n_workers=1,
        out_mask_path=None):

    
    # 1. Load and prepare hourly data as a full Dataset
    # We load the dataset and rename the dimensions/coordinates for cleaner handling.
    ds = xr.open_dataset(ncpath)
    ds = ds.rename({time_name: 'time', y_name: 'y', x_name: 'x'})
    
    # Isolate the Data Variable for cleaner operation
    pr_hourly = ds[varname]

    # 2. Convert to 6-hourly summed precipitation
    pr_6hr = sum_to_6h_centered(pr_hourly)
    print("Detector: Converted to 6-hourly summed precipitation.")
    
    # C. Load the resulting 6-hourly DataArray.
    pr = pr_6hr.load()

    # Update metadata
    nt = pr.sizes["time"]
    ny = pr.sizes["y"]
    nx = pr.sizes["x"]
    print(f"Detector: 6-hourly data loaded, new nt={nt}")

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
        thresh_block = np.full_like(pr_block, accum_threshold, dtype=np.float32)


        # Smooth only spatial dims
        pr_smooth_block = gaussian_filter(
            pr_block,
            sigma=(0, gaussian_sigma_px, gaussian_sigma_px),
            mode='nearest',
            truncate=3.0
        )

        # Prethresholding step (much faster)
        thresholded_block = (pr_smooth_block >= thresh_block).astype(np.uint8)

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

    
    # NOTE: You MUST ensure the 'lat' and 'lon' coordinates are correctly attached 
    # to the final output mask DataArray, as they are needed for plotting and downstream analysis.
    # You can get them from the original dataset:
    lat2D = ds['lat'].values if 'lat' in ds else np.meshgrid(ds['x'], ds['y'])[1]
    lon2D = ds['lon'].values if 'lon' in ds else np.meshgrid(ds['x'], ds['y'])[0]
    
    # Wrap into DataArray
    da_mask = xr.DataArray(
        out,
        coords={'time': pr['time'].values, 'y': pr['y'].values, 'x': pr['x'].values},
        dims=('time', 'y', 'x'),
        name='PR_mask',
        attrs={
            'description': 'Binary mask of precipitation objects (1=object)',
            'min_area_km2': min_area_km2,
            'gaussian_sigma_px': gaussian_sigma_px,
            'pixel_area_km2': pixel_area_km2,
        }
    )

    if out_mask_path:
        da_mask.to_dataset().to_netcdf(out_mask_path, mode='w')
        print(f"Saved mask to {out_mask_path}")

    return da_mask


# Analogue function for RCMs is defined in the same way but with different variable names and coordinate handling, see below.
def detect_precip_objects_rcm(
        ncpath,
        varname='pr',              # CHANGED: RCMs use 'pr'
        time_name='time',          # CHANGED: RCMs use 'time'
        y_name='rlat', x_name='rlon',
        grid_spacing_km=11.17,      
        gaussian_sigma_px=1.0,
        min_area_km2=25000,
        accum_threshold=0.0006,    # masking threshold in meters (0.0006 m = 0.6 mm)
        connectivity=8,
        n_workers=1,
        out_mask_path=None,
        model_type=None,
         ):         # New parameter to specify model type

    
    # 1. Load and prepare data
    print(f"Detector: Opening {model_type} file {ncpath}...")
    ds = xr.open_dataset(ncpath)

    # Handle renaming safely. 
    # If the file already has 'time', 'y', or 'x', we avoid renaming conflicts.
    rename_dict = {}
    if time_name != 'time': rename_dict[time_name] = 'time'
    if y_name != 'y': rename_dict[y_name] = 'y'
    if x_name != 'x': rename_dict[x_name] = 'x'
    
    if rename_dict:
        ds = ds.rename(rename_dict)
    
    # Isolate the Data Variable
    # HCLIM 'pr' is usually in meters (m), same as ERA5 'tp'
    pr_hourly = ds[varname]

    # 2. Convert to 6-hourly summed precipitation
    # NOTE: This assumes you have 'sum_to_6h_centered' defined elsewhere in your scope
    pr_6hr = sum_to_6h_centered(pr_hourly)
    print("Detector: Converted to 6-hourly summed precipitation.")
    
    # 3. Load data into memory
    pr = pr_6hr.load()

    # Update metadata
    nt = pr.sizes["time"]
    ny = pr.sizes["y"]
    nx = pr.sizes["x"]
    print(f"Detector: 6-hourly data loaded, nt={nt}, ny={ny}, nx={nx}")

    pixel_area_km2 = float(grid_spacing_km)**2
    out = np.zeros((nt, ny, nx), dtype=np.uint8)

    block_size = 200 

    # --- PROCESS DATA BLOCK-BY-BLOCK ---
    for i0 in range(0, nt, block_size):
        i1 = min(i0 + block_size, nt)

        if i0 % 1000 == 0: # Reduce print spam
            print(f"Processing block {i0}–{i1}...")

        # Load block
        pr_block = pr.isel(time=slice(i0, i1)).values.astype(np.float32)
        thresh_block = np.full_like(pr_block, accum_threshold, dtype=np.float32)

        # Smooth spatial dims
        pr_smooth_block = gaussian_filter(
            pr_block,
            sigma=(0, gaussian_sigma_px, gaussian_sigma_px),
            mode='nearest',
            truncate=3.0
        )

        # Prethresholding
        thresholded_block = (pr_smooth_block >= thresh_block).astype(np.uint8)

        # --- SERIAL PROCESSING ---
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
            raise NotImplementedError("Parallel block processing not implemented.")

    # --- COORDINATE HANDLING FOR RCMs ---
    # RCM files on rotated grids often don't have 2D lat/lon variables. 
    # We must generate them if they are missing to ensure the output is usable.
    
    coords = {'time': pr['time'].values, 'y': pr['y'].values, 'x': pr['x'].values}
    
    # Try to grab lat/lon if they exist, otherwise generate them from rlat/rlon
    if 'lat' in ds:
        vals_lat = ds['lat'].values
        vals_lon = ds['lon'].values
    else:
        print("Detector: Generating 2D coordinate meshgrid...")
        vals_lon, vals_lat = np.meshgrid(ds['x'].values, ds['y'].values)

    # Wrap into DataArray
    da_mask = xr.DataArray(
        out,
        coords=coords,
        dims=('time', 'y', 'x'),
        name='PR_mask',
        attrs={
            'description': 'Binary mask of precipitation objects ({model_type}) (1=object)'.format(model_type=model_type),
            'model': model_type,
            'min_area_km2': min_area_km2,
            'gaussian_sigma_px': gaussian_sigma_px,
            'pixel_area_km2': pixel_area_km2,
        }
    )

    # Attach 2D lat/lon as coordinates (optional but recommended for plotting)
    da_mask.coords['lat'] = (('y', 'x'), vals_lat)
    da_mask.coords['lon'] = (('y', 'x'), vals_lon)

    if out_mask_path:
        # Create a clean dataset to save
        out_ds = da_mask.to_dataset()
        
        # Preserve rotated pole info if available (crucial for HCLIM plotting)
        if 'rotated_pole' in ds:
            out_ds['rotated_pole'] = ds['rotated_pole']
            out_ds['PR_mask'].attrs['grid_mapping'] = 'rotated_pole'

        out_ds.to_netcdf(out_mask_path, mode='w')
        print(f"Saved {model_type} mask to {out_mask_path}")

    return da_mask



# ---------- Plotting functions ----------

def plot_time_index_pr_and_contours(
        pr6_data,   # <--- PASS THE PRE-CALCULATED DATA HERE
        da_mask, 
        time_index=0,
        rotated_pole_data=None, # Pass the rotated pole object or dataset
        x_name='rlon', y_name='rlat',
        power_norm_gamma=0.35 # for plotting, not related to detection threshold
    ):
    
    # Extract the specific timestep
    pr2d = (pr6_data.isel(time=time_index) * 1000.).values # convert to mm
    mask2d = da_mask.isel(time=time_index).values
    current_time = pr6_data.time.values[time_index]

    rlon = pr6_data[x_name].values
    rlat = pr6_data[y_name].values

    # Setup Projection
    if rotated_pole_data is not None:
        pole_lon = float(rotated_pole_data.grid_north_pole_longitude)
        pole_lat = float(rotated_pole_data.grid_north_pole_latitude)
    else:
        # Default fallback or look in attributes
        pole_lon = 20.0 # from PolarCORDEX grid
        pole_lat = 5.0 
        
    rotated_crs = ccrs.RotatedPole(pole_longitude=pole_lon, pole_latitude=pole_lat)

    inner_domain = xr.open_dataset("PolarRES_WP3_Antarctic_domain.nc")
    rlon_inner = inner_domain["rlon"].values
    rlat_inner = inner_domain["rlat"].values

    # Figure + projection
    fig, ax = plt.subplots(
        figsize=(12, 8),
        subplot_kw={"projection": ccrs.Orthographic(0, -90)}
    )

    # Map features
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

    # Inner domain (optional)
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
    cbar.set_label("Precipitation (mm/6h)")

    # Contours of objects
    if np.any(mask2d):
        ax.contour(
            X, Y, mask2d,
            levels=[0.5],
            colors="white",
            linewidths=2.0,
            linestyles="solid",
            transform=rotated_crs,
            zorder=10
        )

    time_val = pr6_data.time.values[time_index]
    ax.set_title(f"Precipitation + PR Objects\n{str(time_val)}")
    plt.tight_layout()

    return fig, ax


# ---------- Main block ----------

start_time = time.time()

if __name__ == "__main__":
    start_time = time.time()

    # --- 1. User Parameters ---
    data_type = "RACMO2"  # "ERA5" or "HCLIM" or "MetUM" or "RACMO2"
    pr_base_path = "/cluster/work/users/mmarco02/moaap_project/"
    years = list(range(2001, 2021))
    sigma = 2.0
    min_area = 50000  # km2

    # --- 2. Plotting parameters (for debugging/visualization) ---
    plot_mode = True 
    plot_year = 2006  
    plot_start_idx = 700
    plot_end_idx = 720
    plot_folder = f"zzz_PR_plots_{plot_year}_summed_{data_type}"
    os.makedirs(plot_folder, exist_ok=True)

    # --- 3. Main Loop ---
    for year in years:
        print(f"\nProcessing year {year}...")

        # Setup paths and names based on data source
        if data_type in ["HCLIM", "MetUM"]:
            input_nc = os.path.join(pr_base_path, f"PolarRES/{data_type}/nested/{year}/ANT_{data_type}_pr_{year}_nested.nc")
            #input_nc = os.path.join(pr_base_path, f"PolarRES/{data_type}/nested/{year}/ANT_{data_type}_pr_{year}_nested_con.nc")
            out_mask_path = os.path.join(pr_base_path, f"moaap_output_{data_type}_test_PR_all_objects_summed/{year}/{year}_{data_type}_ANT_{year}_PR_ObjectMasks__dt-6h_MOAAP-masks.nc")
            var_name = 'pr'
            t_name = 'time'
            # Use the RCMs-specific detector
            detector_func = detect_precip_objects_rcm

        elif data_type == "RACMO2":
                    # Setup paths and names based on data source
        if data_type in ["HCLIM", "MetUM", "RACMO2"]:
            input_nc = os.path.join(pr_base_path, f"PolarRES/{data_type}/nested/{year}/ANT_{data_type}_pr_{year}_nested_con.nc")
            out_mask_path = os.path.join(pr_base_path, f"moaap_output_{data_type}_test_PR_all_objects_summed/{year}/{year}_{data_type}_ANT_{year}_PR_ObjectMasks__dt-6h_MOAAP-masks.nc")
            var_name = 'pr'
            t_name = 'time'
            # Use the RCMs-specific detector
            detector_func = detect_precip_objects_rcm
        
        else: # ERA5
            input_nc = os.path.join(pr_base_path, f"ERA5/{year}/remapped/ANT_pr_{year}_remapped.nc")
            out_mask_path = os.path.join(pr_base_path, f"moaap_output_ERA5_test_PR_all_objects_summed/{year}/{year}_ERA5_ANT_{year}_PR_ObjectMasks__dt-6h_MOAAP-masks.nc")
            var_name = 'tp'
            t_name = 'valid_time'
            # Use the ERA5-specific detector
            detector_func = detect_precip_objects_era5

        os.makedirs(os.path.dirname(out_mask_path), exist_ok=True)

        # A. RUN DETECTOR
        # This calculates the mask and saves it to NetCDF
        print(f"Running {data_type} detector...")
        mask = detector_func(
            ncpath=input_nc,
            grid_spacing_km=11.17,
            gaussian_sigma_px=sigma,
            min_area_km2=min_area,
            connectivity=8,
            n_workers=1,
            out_mask_path=out_mask_path,
            model_type=data_type,
        )
        print(f"Year {year} mask completed. Shape: {mask.shape}")

        # B. PLOTTING BLOCK
        # We only do this for the specific plot_year 
        if plot_mode and year == plot_year:
            print(f"Preparing plots for {year}...")
            
            # Open the source file ONCE for the whole year
            ds_raw = xr.open_dataset(input_nc)
            
            # Calculate 6-hour sums ONCE for the whole year using the optimized function
            # This ensures the plots match the data the detector saw
            pr6_for_plotting = sum_to_6h_centered(ds_raw[var_name]).load()
            
            # Loop through the requested time indices
            for t_idx in range(plot_start_idx, plot_end_idx + 1):
                # Check if index is valid for this year
                if t_idx >= len(pr6_for_plotting.time):
                    break
                    
                fig, ax = plot_time_index_pr_and_contours(
                    pr6_data=pr6_for_plotting,  # Pass the calculated sums
                    da_mask=mask,               # Pass the mask we just generated
                    time_index=t_idx,
                    rotated_pole_data=ds_raw['rotated_pole'] if 'rotated_pole' in ds_raw else None,
                    x_name='rlon', 
                    y_name='rlat'
                )
                
                plt.savefig(f"{plot_folder}/PR_plot_{year}_t{t_idx:03d}.png", dpi=150)
                plt.close(fig)
            
            ds_raw.close() # Clean up file handle
            print(f"Plots saved to {plot_folder}")

    elapsed = time.time() - start_time
    print(f"\nTotal elapsed time: {elapsed/60:.2f} minutes")
    print("All done.")