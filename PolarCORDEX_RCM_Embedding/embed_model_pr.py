# This script does the same as embed_model_data.py but is specifically optimized for the 'pr' variable, which has a unique temporal structure (1-hourly with accumulations).
# It includes a 1-hour temporal trimming step to ensure proper alignment of the precipitation data between ERA5 and the RCM, and it generates diagnostic plots to check for discontinuities at the RCM-ERA5 boundary.



import os
import re
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from tqdm import tqdm
import cartopy.crs as ccrs
import time
import sys
import gc

start_time = time.time()


print("All imports successful")

#########################################################################################
# --- CONFIGURATION ---
# NOTE: Update these paths to match the outputs from your CDO steps
data_type = "RACMO2"  # "HCLIM" or "MetUM" or "RACMO2"
rcm_folder = os.path.abspath(f'PolarRES/{data_type}') 
era5_folder = os.path.abspath('ERA5')

# Variable names, the same for RCM, ERA5 and merged
var_list=["pr"] # 1 hourly variable

years = range(2001, 2021)  # 2001-2020

# --- NEW BLENDING CONSTANT ---
BLENDING_WIDTH_POINTS = 0 
print(f"Blending width set to: {BLENDING_WIDTH_POINTS} grid points.")
# -----------------------------------------------------------------------------------------
##########################################################################################
##########################################################################################
# NEEDED FUNCTIONS
# --- Define Tanh Blending Function ---
def tanh_blend_func(L, k=3.0):
    """
    Creates a 1D tanh blending array W(x) from 1.0 to 0.0 over L points.
    L: blending width (BLENDING_WIDTH_POINTS)
    k: steepness factor (5.0 is a good default for a steep but smooth taper)
    """
    x_prime = np.linspace(-k, k, L)
    weights = 0.5 * (1.0 - np.tanh(x_prime))
    # Renormalize to ensure clean 1.0 and 0.0 ends
    weights = (weights - weights[-1]) / (weights[0] - weights[-1])
    return weights


def quantitative_boundary_check(ds_outer, ds_inner, var_name, lat_start, lon_slice_indices, year):
    """
    Calculates and plots the quantitative difference (bias) profile 
    between RCM and ERA5 at the inner (RCM) southern boundary.
    """
    MERGE_VAR_NAME = var_name
    
    # Select the first row of grid cells just inside the RCM inner domain (Southern Edge)
    # RCM's first row is index 0
    rcm_boundary_values = ds_inner[MERGE_VAR_NAME].isel(time=0).values[0, :]
    
    # ERA5's corresponding values at the boundary (lat_start index)
    boundary_lat_idx = lat_start 
    era5_boundary_values = ds_outer[MERGE_VAR_NAME].isel(time=0).values[boundary_lat_idx, lon_slice_indices]

    # --- Calculate Differences ---
    difference = rcm_boundary_values - era5_boundary_values
    
    # --- Print Statistics ---
    print("\n--- Quantitative Boundary Bias Check (Southern Edge, First Time Step) ---")
    print(f"Variable: {MERGE_VAR_NAME}, Year: {year}")
    print(f"Mean Difference (RCM - ERA5): {np.mean(difference):.3f}")
    print(f"Max Absolute Difference: {np.max(np.abs(difference)):.3f}")
    print(f"Standard Deviation of Difference: {np.std(difference):.3f}")

    # --- Plot the Profile ---
    try:
        units = ds_outer[MERGE_VAR_NAME].units
    except:
        units = "units_unknown"

    PLOT_DIR = "Boundary_Checks"
    os.makedirs(PLOT_DIR, exist_ok=True)
    filename = os.path.join(PLOT_DIR, f'{MERGE_VAR_NAME}_{year}_boundary_profile_check.png')
    
    plt.figure(figsize=(10, 4))
    plt.plot(era5_boundary_values, label='ERA5 (Outer Value)')
    plt.plot(rcm_boundary_values, label='RCM (Inner Value)')
    plt.plot(difference, color='red', linestyle='--', label='Difference (RCM - ERA5)')
    plt.title(f'{MERGE_VAR_NAME} Profiles at RCM Southern Boundary ({year})')
    plt.xlabel("Longitude Index")
    plt.ylabel(f"Value ({units})")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    print(f"💾 Boundary profile plot saved to: {filename}")
    # ---------------------------------------------------------------------------------


def smooth_blend_and_save(file_era5_outer, file_rcm_inner, file_merged_out, var_name, blend_w):
    """
    Specifically optimized for 'pr' (Precipitation) with 1-hour temporal trimming.
    """
    if not os.path.exists(file_era5_outer) or not os.path.exists(file_rcm_inner):
        return None

    # --- 1. Load Data ---
    ds_outer = xr.open_dataset(file_era5_outer)
    ds_inner = xr.open_dataset(file_rcm_inner)

    # Standardize ERA5 Time Dimension immediately
    if 'valid_time' in ds_outer.dims:
        ds_outer = ds_outer.rename_dims({'valid_time': 'time'})
        ds_outer = ds_outer.rename_vars({'valid_time': 'time'})

    # --- 2. TEMPORAL TRIMMING (Precipitation Logic) ---
    print(f"Applying 1-hour temporal trimming for {var_name}...")
    
    # ERA5: Drop hour 00:00 (Previous year's accumulation)
    ds_outer = ds_outer.isel(time=slice(1, None))
    
    # HCLIM: Drop hour 00:00 of next year (already provided in next year's file)
    ds_inner = ds_inner.isel(time=slice(0, -1))
    
    # Synchronize lengths (handles Leap Years vs Non-Leap years automatically)
    min_steps = min(len(ds_outer.time), len(ds_inner.time))
    ds_outer = ds_outer.isel(time=slice(0, min_steps))
    ds_inner = ds_inner.isel(time=slice(0, min_steps))

    # --- 3. VARIABLE ALIGNMENT ---
    # According to your ncdump, ERA5 is 'tp' and HCLIM is 'pr'
    ERA5_INTERNAL_VAR = 'tp' if 'tp' in ds_outer.variables else 'pr'
    MERGE_VAR_NAME = 'pr' # We want the output to be 'pr'

    if ERA5_INTERNAL_VAR != MERGE_VAR_NAME:
        ds_outer = ds_outer.rename_vars({ERA5_INTERNAL_VAR: MERGE_VAR_NAME})

    # Use HCLIM's time as the master coordinate
    ds_outer['time'] = ds_inner['time']

    # --- 4. SPATIAL ALIGNMENT ---
    _, n_inner_lat, n_inner_lon = ds_inner[MERGE_VAR_NAME].shape 
    _, n_outer_lat, n_outer_lon = ds_outer[MERGE_VAR_NAME].shape

    lat_start = (n_outer_lat - n_inner_lat) // 2
    lon_start = (n_outer_lon - n_inner_lon) // 2

    lat_slice_indices = slice(lat_start, lat_start + n_inner_lat)
    lon_slice_indices = slice(lon_start, lon_start + n_inner_lon)

    # --- 5. BLENDING ---
    ds_merged = ds_outer.copy(deep=False)  # Before it was true: it worked but took a lot of time
    inner_time_size = ds_inner.sizes['time']
    
    # Weight Map
    weight_rcm = np.zeros((n_outer_lat, n_outer_lon), dtype=np.float32)
    
    # Since blend_w is 0 (Hard Nest), we just fill the inner box with 1.0
    if blend_w == 0:
        weight_rcm[lat_slice_indices, lon_slice_indices] = 1.0
    else:
        # (Your Tanh logic would go here if blend_w > 0)
        # For now, keeping it simple as per your config:
        weight_rcm[lat_slice_indices, lon_slice_indices] = 1.0

    # Apply Merge
    outer_np = ds_outer[MERGE_VAR_NAME].values.copy() 
    inner_np = ds_inner[MERGE_VAR_NAME].values.copy() 

    # Masking for the swap
    outer_np[:, lat_slice_indices, lon_slice_indices] = inner_np
    ds_merged[MERGE_VAR_NAME].values = outer_np

    # --- 6. SAVE WITH COMPRESSION ---
    # This reduces file size significantly on Betzy
    encoding = {MERGE_VAR_NAME: {'zlib': True, 'complevel': 4}}
    
    os.makedirs(os.path.dirname(file_merged_out), exist_ok=True)
    ds_merged.to_netcdf(file_merged_out, encoding=encoding)

    return ds_merged, ds_outer, ds_inner, lat_start, lon_slice_indices, lat_slice_indices
    


def plot_discontinuity_check(ds_merged, var_name, lat_slice_indices, lon_slice_indices):
    """
    Generates a quick validation plot, optimized to detect discontinuities.
    Uses the 'RdBu_r' colormap and tight limits.
    """
    variable_name = var_name
    
    # Plotting for the first time step only
    t_idx = 0 
    
    # Create the output directory
    OUTPUT_DIR = f"{variable_name}_plots"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Coordinate Setup ---
    rlat = ds_merged['rlat']
    rlon = ds_merged['rlon']
    rlon2D, rlat2D = np.meshgrid(rlon, rlat) 

    rp_params = ds_merged['rotated_pole'].attrs
    rotated_crs = ccrs.RotatedPole(
        pole_longitude=rp_params['grid_north_pole_longitude'],
        pole_latitude=rp_params['grid_north_pole_latitude']
    )
    
    # --- Data and Limit Setup (Optimized for Discontinuity Check) ---
    data_slice = ds_merged[variable_name].isel(time=t_idx)
    time_label = ds_merged['time'].isel(time=t_idx).dt.strftime('%Y-%m-%d').item()

    # Calculate tight symmetric limits (mean +/- 1.5 standard deviation)
    mu = data_slice.values.mean()
    sigma = data_slice.values.std()
    K_FACTOR = 1.5 # Amplifies local difference
    vmin = mu - K_FACTOR * sigma
    vmax = mu + K_FACTOR * sigma
    
    # --- Boundary Coordinate Extraction ---
    rlon_inner = ds_merged['rlon'].isel(rlon=lon_slice_indices).values
    rlat_inner = ds_merged['rlat'].isel(rlat=lat_slice_indices).values


    # --- Setup Figure and Plot ---
    fig = plt.figure(figsize=(8, 7)) 
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(0, -90)) 
    
    ax.set_extent([-180, 180, -90, -20], crs=ccrs.PlateCarree())
    ax.coastlines(color="#969696")
    ax.set_title(f'Discontinuity Check: {variable_name} ({time_label})')

    im = ax.pcolormesh(
        rlon2D, rlat2D, data_slice.values,
        cmap='RdBu_r', # High-contrast diverging map
        vmin=vmin, vmax=vmax, # Tight limits to highlight small jumps
        transform=rotated_crs, shading='auto'
    )
    
    # Add colorbar (simple, no unit specified)
    plt.colorbar(im, ax=ax, orientation='vertical', label=f'{variable_name} (Tight Range)', fraction=0.04, pad=0.04)

    # Overlay the RCM Boundary (red dashed line)
    if rlon_inner is not None and rlat_inner is not None:
        top_i = np.column_stack([rlon_inner, np.full_like(rlon_inner, rlat_inner[-1])])
        bot_i = np.column_stack([rlon_inner, np.full_like(rlon_inner, rlat_inner[0])])
        left_i = np.column_stack([np.full_like(rlat_inner, rlon_inner[0]), rlat_inner])
        right_i = np.column_stack([np.full_like(rlat_inner, rlon_inner[-1]), rlat_inner])
        inner = np.vstack([bot_i, right_i, top_i[::-1], left_i[::-1]])
        ax.plot(inner[:, 0], inner[:, 1], color="red", linewidth=1.0, linestyle='--', transform=rotated_crs)

    # Finalize and Save
    plt.tight_layout()
    filename = os.path.join(OUTPUT_DIR, f'{variable_name}_discontinuity_check_{time_label}.png') 
    plt.savefig(filename, dpi=200)
    plt.close(fig) # Close the figure to free up memory

    print(f"✅ Discontinuity check plot saved for {var_name} in '{OUTPUT_DIR}' directory.")


# ######################################################################################
# --- MAIN EXECUTION LOOP ---
# ######################################################################################
# --- Representative Variables to Plot ---
PLOT_VARS = ["pr"]

print("Starting main blending execution...")
for year in tqdm(years, desc="Total Years"):
    for var_name in tqdm(var_list, desc=f"Year {year} Variables"):
        
        # --- 1. Define Paths ---
        # Assuming file naming convention from previous steps
        FILE_ERA5_OUTER = os.path.join(era5_folder, str(year), "remapped", f"ANT_{var_name}_{year}_remapped.nc")
        FILE_RCM_INNER = os.path.join(rcm_folder, "remapped", str(year), f"ANT_{data_type}_{var_name}_{year}_remapped.nc") 
        FILE_MERGED = os.path.join(rcm_folder, "nested", str(year), f"ANT_{data_type}_{var_name}_{year}_nested_con.nc") # Added year to output name

        # --- 2. Execute Blending ---
        try:
            results = smooth_blend_and_save(FILE_ERA5_OUTER, FILE_RCM_INNER, FILE_MERGED, var_name, BLENDING_WIDTH_POINTS)
        except Exception as e:
            print(f"❌ Error blending {var_name} for {year}: {e}")
            continue # Move to next variable/year

        # 3. Conditional Plotting (Only for your chosen few)
        if results is not None and var_name in PLOT_VARS:
            ds_merged, ds_outer, ds_inner, lat_start, lon_slice_indices, lat_slice_indices = results
            
            # Execute Quantitative Boundary Check
            try:
                quantitative_boundary_check(ds_outer, ds_inner, var_name, lat_start, lon_slice_indices, year)
            except Exception as e:
                print(f"⚠️ Boundary plot failed for {var_name}: {e}")
                
            # Execute Visualization Check
            try:
                plot_discontinuity_check(ds_merged, var_name, lat_slice_indices, lon_slice_indices)
            except Exception as e:
                print(f"⚠️ Discontinuity plot failed for {var_name}: {e}")

        # 4. Clean up memory (Important!)
        if results is not None:
            # result index 0, 1, 2 are the xarray datasets
            for i in range(3): results[i].close()
            del results
            gc.collect()


elapsed_time = time.time() - start_time
print(f"\nTotal script execution time: {elapsed_time:.2f} seconds")