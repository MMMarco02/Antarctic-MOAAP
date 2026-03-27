# This script performs the embedding of RCM data into the ERA5 background for the specified variables and years.
# It includes a blending mechanism (currently set to 0 points, i.e., no blending) and generates diagnostic plots to check for discontinuities at the RCM-ERA5 boundary.
# The script is structured to be modular, with functions for blending, plotting, and quantitative checks, and a main loop that iterates over the years and variables.
# It representes the final step in the nesting process, where the merged datasets are saved for use in the PolarRES project.

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

########################################################################################
# --- CONFIGURATION ---
# NOTE: Update these paths to match the outputs from your CDO steps
model_type = "RACMO2"  # Options: "HCLIM" or "MetUM" or "RACMO2"
rcm_folder = os.path.abspath(f'PolarRES/{model_type}') 
era5_folder = os.path.abspath('ERA5/')


# Variable names, the same for RCM, ERA5 and merged
# This was a previous list, but it is to show that you arbitrarily configure the variables you want to blend and track. 
# The only requirement is that the variable name (e.g., 'zg500') must be present in both the RCM and ERA5 files (after any necessary renaming).
#var_list = [
#    "ua200", "ua300", "ua400", "ua500", "ua600", "ua700", "ua850", "ua925", "ua1000",
#    "va200", "va300", "va400", "va500", "va600", "va700", "va850", "va925", "va1000",
#    "ta850", 
#    "zg300", "zg400", "zg500", "zg600", "zg700", "zg850", "zg925", "zg1000",
#    "hus300", "hus400", "hus500", "hus600", "hus700", "hus850", "hus925", "hus1000"] # 6 hourly variables (MISSING THE 750 HPA LEVELS)

var_list = [
    "ua300", "ua850",
    "va300", "va850",
    "ta850", 
    "zg500",
    "hus850",
    "ivte_computed", "ivtn_computed"] # 6 hourly variables 


years = range(2001, 2021)  

# --- NEW BLENDING CONSTANT ---
# After testing, belding was not applied (kep to 0 points). However, the logic is fully implemented and can be activated by changing this constant to a positive integer (e.g., 5 or 10).
BLENDING_WIDTH_POINTS = 0 
print(f"Blending width set to: {BLENDING_WIDTH_POINTS} grid points.")
# -----------------------------------------------------------------------------------------

##########################################################################################
# --- NEEDED FUNCTIONS ---
##########################################################################################

# --- Define Tanh Blending Function (for blending) ---
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

# --- Define Quantitative Boundary Check Function ---
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
    Performs the blending of the RCM data into the ERA5 canvas and saves the result.
    Includes logic to handle ERA5 variable name differences (u/v/t/z/q) and 4D data (pressure_level).
    """

    # Little check
    if not os.path.exists(file_era5_outer):
        print(f"MISSING ERA5: {file_era5_outer}")
    if not os.path.exists(file_rcm_inner):
        print(f"MISSING RCM: {file_rcm_inner}")
    
    # --- 1. Load Data ---
    if not os.path.exists(file_era5_outer) or not os.path.exists(file_rcm_inner):
        return None

    # Load and immediately drop valid_time to prevent decoding conflicts
    ds_outer = xr.open_dataset(file_era5_outer).drop_vars('valid_time', errors='ignore')
    ds_inner = xr.open_dataset(file_rcm_inner).drop_vars('valid_time', errors='ignore')

        
    # Try to find the variable name
    if var_name in ds_inner.variables:
        MERGE_VAR_NAME = var_name
    elif var_name.replace('_computed', '') in ds_inner.variables:
        MERGE_VAR_NAME = var_name.replace('_computed', '')
    else:
        raise ValueError(f"Could not find {var_name} or stripped version in {file_rcm_inner}")

    print(f"Using internal variable name: {MERGE_VAR_NAME}")


    # --- 2. Alignment and Preparation ---

    # 2a. Determine internal variable names
    # (This is the logic we just added)
    if var_name in ds_inner.variables:
        MERGE_VAR_NAME = var_name
    elif var_name.replace('_computed', '') in ds_inner.variables:
        MERGE_VAR_NAME = var_name.replace('_computed', '')
    else:
        raise ValueError(f"Could not find {var_name} in {file_rcm_inner}")

    rcm_var_name_in_file = MERGE_VAR_NAME 
    
    # Determine ERA5 variable name
    if MERGE_VAR_NAME.startswith('ivt'):
        # Force uppercase if that's what ERA5 uses internally
        if MERGE_VAR_NAME == 'ivte':
            era5_var_name_in_file = 'IVTE'
        elif MERGE_VAR_NAME == 'ivtn':
            era5_var_name_in_file = 'IVTN'
        else:
            era5_var_name_in_file = MERGE_VAR_NAME 
    else:
        era5_var_prefix = MERGE_VAR_NAME[:1] 
        if era5_var_prefix == 'h' and MERGE_VAR_NAME.startswith('hus'):
             era5_var_prefix = 'q'
        era5_var_name_in_file = era5_var_prefix
    
    # 2b. Check for 4D Data (Pressure Levels) and select the level
    if 'pressure_level' in ds_outer.dims:
        # Use .isel to collapse the 4D (T, Z, Y, X) array to 3D (T, Y, X)
        ds_outer = ds_outer.isel(pressure_level=0)
        print(f"INFO: Sliced pressure_level=0 from {file_era5_outer}.")

    # 2c. Rename dimensions if necessary (e.g., 'valid_time' to 'time')
    if 'valid_time' in ds_outer.dims:
        ds_outer = ds_outer.rename_dims({'valid_time': 'time'})

    # 2d. Rename variables to the common MERGE_VAR_NAME (e.g., 'ua200')
    # Inner RCM file (optional rename check, typically not needed if CDO output is clean)
    if rcm_var_name_in_file not in ds_inner.variables:
         raise ValueError(f"RCM var '{rcm_var_name_in_file}' not found in inner file.")

    # Outer ERA5 file (Rename e.g., 'u' to 'ua200')
    if era5_var_name_in_file in ds_outer.variables:
        ds_outer = ds_outer.rename_vars({era5_var_name_in_file: MERGE_VAR_NAME})
    else:
        # If the expected ERA5 variable ('u', 'v', etc.) is not found, something is wrong.
        raise ValueError(f"ERA5 var '{era5_var_name_in_file}' not found in outer file. Check file: {file_era5_outer}")


    # 2e. Slice the outer data to match the time period of the inner data
    inner_time_size = ds_inner.sizes['time']
    ds_outer = ds_outer.isel(time=slice(0, inner_time_size))

    # --- 3. Calculate Slice Indices (based on RCM size) ---
    # Data is now 3D (T, Y, X), so shape unpacking will work.
    _, n_inner_lat, n_inner_lon = ds_inner[MERGE_VAR_NAME].shape 
    _, n_outer_lat, n_outer_lon = ds_outer[MERGE_VAR_NAME].shape

    lat_start = (n_outer_lat - n_inner_lat) // 2
    lon_start = (n_outer_lon - n_inner_lon) // 2

    lat_slice_indices = slice(lat_start, lat_start + n_inner_lat)
    lon_slice_indices = slice(lon_start, lon_start + n_inner_lon)

    # --- 4. IMPLEMENT THE SMOOTH BLENDING ZONE ---
    
    # 4a. Create the merged dataset and replace time coordinate
    ds_merged = ds_outer.copy(deep=True) 
    ds_merged['time'] = ds_inner['time'] 

    # 4b. Initialize the RCM weight map (W_RCM)
    weight_rcm = np.zeros((n_outer_lat, n_outer_lon), dtype=np.float32)

    # 4c. Define the RCM Core (W_RCM = 1)
    core_lat_start = lat_start + blend_w
    core_lat_end = lat_start + n_inner_lat - blend_w
    core_lon_start = lon_start + blend_w
    core_lon_end = lon_start + n_inner_lon - blend_w
    weight_rcm[core_lat_start:core_lat_end, core_lon_start:core_lon_end] = 1.0

    '''
    # Different types of blending strategies implemented, none currently used.
    # 4d. Apply Linear Blending (Tapering) on the four boundaries
    # LINEAR BLENDING ZONE DEFINITION:
    blend_w = BLENDING_WIDTH_POINTS
    lats_to_blend = n_inner_lat - 2 * blend_w
    lons_to_blend = n_inner_lon - 2 * blend_w

    # --- Create 1D linear ramp array ---
    # From 0 (inner edge) to 1 (outer edge). We reverse it later for the weight map.
    ramp = np.linspace(0.0, 1.0, blend_w)
    weight_decay = 1.0 - ramp # 1.0 to 0.0 (used for W_RCM)

    # NEW QUADRATIC BLENDING ZONE DEFINITION:
    # 4d. Apply Quadratic Blending (Tapering) on the four boundaries
    blend_w = BLENDING_WIDTH_POINTS
    lats_to_blend = n_inner_lat - 2 * blend_w
    lons_to_blend = n_inner_lon - 2 * blend_w

    # --- Create 1D distance-normalized array (x) ---
    # x goes from 0.0 (inner edge) to 1.0 (outer edge)
    x = np.linspace(0.0, 1.0, blend_w)

    # --- Quadratic Weight Function for W_RCM (1.0 to 0.0) ---
    # Function: W_RCM = 1.0 - x^2
    weight_decay = 1.0 - x**2 # This goes from 1.0 (inner) down to 0.0 (outer)

    # The 'ramp' that goes from 0.0 to 1.0 (used for the North/East boundaries)
    # Function: W_RCM = x^2
    ramp = x**2 # This goes from 0.0 (inner) up to 1.0 (outer)
    '''
    # NEW TANH BLENDING ZONE DEFINITION:
    # 4d. Apply Tanh Blending (Tapering) on the four boundaries
    if BLENDING_WIDTH_POINTS == 0:
        print("Blending width is set to 0. No blending will be applied.")   

    else:
        blend_w = BLENDING_WIDTH_POINTS
        lats_to_blend = n_inner_lat - 2 * blend_w
        lons_to_blend = n_inner_lon - 2 * blend_w

        # --- Create 1D Tanh weight array ---
        # weight_decay goes from 1.0 (inner edge) to 0.0 (outer edge)
        # Use k=5.0 for a balance between smoothness and compactness.
        weight_decay = tanh_blend_func(blend_w, k=2.5)

        # ramp goes from 0.0 (inner edge) to 1.0 (outer edge)
        ramp = 1.0 - weight_decay


        # --- Apply Blending to Edges of Inner Domain ---

        # 1. South (Bottom): Lat index from lat_start to lat_start + blend_w
        for i in range(blend_w):
            # Slice the entire width of the inner domain
            weight_rcm[lat_start + i, lon_slice_indices] = weight_decay[i]

        # 2. North (Top): Lat index from core_lat_end to core_lat_end + blend_w
        for i in range(blend_w):
            # Slice the entire width of the inner domain
            weight_rcm[core_lat_end + i, lon_slice_indices] = ramp[i]

        # 3. West (Left): Lon index from lon_start to lon_start + blend_w
        for j in range(blend_w):
            # Slice the blended lat range (excluding corners, which are already done/will be done later)
            weight_rcm[lat_slice_indices, lon_start + j] *= weight_decay[j]
            # Use *= to combine with existing North/South weights in the corner

        # 4. East (Right): Lon index from core_lon_end to core_lon_end + blend_w
        for j in range(blend_w):
            # Slice the blended lat range
            weight_rcm[lat_slice_indices, core_lon_end + j] *= ramp[j]
            # Use *= to combine with existing North/South weights in the corner

    # 4e. Perform the Smooth Merge (NumPy array operation)
    outer_np = ds_outer[MERGE_VAR_NAME].values.copy() 
    inner_np = ds_inner[MERGE_VAR_NAME].values.copy() 

    weight_rcm_3d = np.repeat(weight_rcm[np.newaxis, :, :], inner_time_size, axis=0)
    weight_era5_3d = 1.0 - weight_rcm_3d

    outer_slice_np = outer_np[:, lat_slice_indices, lon_slice_indices]
    weight_rcm_slice = weight_rcm_3d[:, lat_slice_indices, lon_slice_indices]
    weight_era5_slice = weight_era5_3d[:, lat_slice_indices, lon_slice_indices]

    blended_data = (weight_rcm_slice * inner_np) + (weight_era5_slice * outer_slice_np)
    outer_np[:, lat_slice_indices, lon_slice_indices] = blended_data
    ds_merged[MERGE_VAR_NAME].values = outer_np

    # --- 5. METADATA CLEANUP ---
    
    for v in ds_merged.variables:
        if 'coordinates' in ds_merged[v].attrs:
            attr_val = ds_merged[v].attrs['coordinates']
            # Remove valid_time and fix spacing
            new_attr = attr_val.replace('valid_time', '').replace('  ', ' ').strip()
            if new_attr:
                ds_merged[v].attrs['coordinates'] = new_attr
            else:
                del ds_merged[v].attrs['coordinates']

    ds_merged.attrs['history'] = f'Merged RCM {var_name} into ERA5 background. Cleaned time coordinates.'
    
    os.makedirs(os.path.dirname(file_merged_out), exist_ok=True)
    ds_merged.to_netcdf(file_merged_out)

    return ds_merged, ds_outer, ds_inner, lat_start, lon_slice_indices, lat_slice_indices

# Done for debugging, can also be removed.
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

    print(f" Discontinuity check plot saved for {var_name} in '{OUTPUT_DIR}' directory.")


# ######################################################################################
# --- MAIN EXECUTION LOOP ---
# ######################################################################################
# --- Choose which Variables to Plot ---
PLOT_VARS = ["ua200", "zg500", "ta850"]

print("Starting main blending execution...")
for year in tqdm(years, desc="Total Years"):
    for var_name in tqdm(var_list, desc=f"Year {year} Variables"):
        
        # --- 1. Define Paths ---

        #FILE_ERA5_OUTER = os.path.join(era5_folder, "remapped", f"ANT_{var_name}_{year}_remapped.nc")
        FILE_ERA5_OUTER = os.path.join(era5_folder, str(year), "remapped", f"ANT_{var_name}_{year}_remapped.nc") # THIS IS THE CORRECT ONE FOR ALL VARIBLES
        FILE_RCM_INNER = os.path.join(rcm_folder, "remapped", str(year), f"ANT_{model_type}_{var_name}_{year}_remapped.nc")
        #FILE_RCM_INNER = os.path.join(rcm_folder, "remapped", f"ANT_{model_type}_{var_name}_{year}_remapped.nc") 
        FILE_MERGED = os.path.join(rcm_folder, "nested", str(year), f"ANT_{model_type}_{var_name}_{year}_nested.nc") # Added year to output name
        #FILE_MERGED = os.path.join(rcm_folder, "nested", f"ANT_{model_type}_{var_name}_{year}_nested.nc") # Added year to output name

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