# This script performs the same analysis as EPE_feature_association_buffer.py, but instead of just calculating the overall fraction of EPEs associated with each feature across all years, 
# it calculates the fraction for each individual year. 
# The output is saves as .npz files containing yearly fractions for each feature and corresponding dataset. 
# This script should be run before "EPE_feature_association_buffer_differences_2.py".

import os
import re
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import cartopy.crs as ccrs
import time
import sys
from scipy.interpolate import griddata
import gc
from scipy.ndimage import distance_transform_edt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import label

start_time = time.time()

# Clear system-wide Python packages (optional)
sys.path = [p for p in sys.path if '/cluster/software/' not in p]

# Ensure conda env packages are first (guarded)
if 'CONDA_PREFIX' in os.environ:
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    sys.path.insert(0, os.path.join(os.environ['CONDA_PREFIX'], 'lib', pyver, 'site-packages'))

# Disable user site packages
os.environ['PYTHONNOUSERSITE'] = '1'

print("All imports successful")


####################################################################################################
# HELPER FUNCTIONS NEEDED FOR THE ANALYSIS
####################################################################################################

# Functions to compute buffer mask around a binary feature mask.
# Approximation: they assume regular grid spacing and uses distance_transform_edt for efficiency, as we have to buffer every single 6-hourly feature mask for 20 years.

def compute_buffer_mask(mask, buffer_km, dx_km=11.17): # grid spacing in km (approx for 0.11° at Antarctic latitudes)

    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    # Distance (in grid cells) from every cell to nearest PR cell
    dist_idx = distance_transform_edt(~mask)

    # Convert to km using constant spacing
    dist_km = dist_idx * dx_km

    return dist_km <= buffer_km

# Helper to buffer the entire 6h 3D array (time, lat, lon)
def buffer_3d_mask(mask_3d, dist_km, dx=11.17):
    
    buffered_3d = np.zeros_like(mask_3d, dtype=bool)
    for i in range(mask_3d.shape[0]):
        if np.any(mask_3d[i]):
            # Reusing your existing compute_buffer_mask logic
            buffered_3d[i] = compute_buffer_mask(mask_3d[i], dist_km, dx)
    
    return buffered_3d

# Function to expand 6-hourly feature masks to hourly, using a -2/+3 hour window around each 6-hourly timestamp. 
def expand_features_to_hourly(feature_mask_6h, time_6h, time_1h):

    feature_mask_6h = feature_mask_6h.astype(bool)

    ntime1 = len(time_1h)
    nlat, nlon = feature_mask_6h.shape[1:]
    
    feature_hourly = np.zeros((ntime1, nlat, nlon), dtype=bool)
    
    for i, t6 in enumerate(time_6h):
        # Find closest hourly index
        idx_center = np.argmin(np.abs(time_1h - t6))
        # Expand -2/+3 hours
        idx_start = max(0, idx_center - 2)
        idx_end   = min(ntime1, idx_center + 4)  # +4 because slicing excludes end
        feature_hourly[idx_start:idx_end] |= feature_mask_6h[i]

    
    return feature_hourly


def month_to_season(month):
    if month in [12, 1, 2]:
        return 'DJF'
    elif month in [3, 4, 5]:
        return 'MAM'
    elif month in [6, 7, 8]:
        return 'JJA'
    else:
        return 'SON'
    

# Build rectangle vertices (used for plotting the domain boxes)
def build_rectangle_vertices(rlon, rlat):
    rlon = np.asarray(rlon)
    rlat = np.asarray(rlat)

    # 1-D → reconstruct 2-D grid
    if rlon.ndim == 1 and rlat.ndim == 1:
        rlon, rlat = np.meshgrid(rlon, rlat)

    top    = np.column_stack([rlon[-1, :], rlat[-1, :]])
    bottom = np.column_stack([rlon[ 0, :], rlat[ 0, :]])
    left   = np.column_stack([rlon[:, 0],  rlat[:, 0]])
    right  = np.column_stack([rlon[:, -1], rlat[:, -1]])

    return np.vstack([bottom, right, top[::-1], left[::-1]])
####################################################################################################


print("🧠 Starting EPE Dual-Dataset Analysis...")

# --- CONFIG ---
model_name = "RACMO2" #HCLIM or MetUM or RACMO2
datasets = ["ERA5", model_name]
percentile = 999
years = np.arange(2001, 2021)
BUFFER_KM = 500

Out_folder = os.path.abspath(f'EPE_Significance_Maps_{model_name}_vs_ERA5_{percentile}')
os.makedirs(Out_folder, exist_ok=True)

# Results storage for the final maps
spatial_results = {ds: {} for ds in datasets}
total_epe_maps = {ds: None for ds in datasets}

# Load grid info
data_vars = xr.open_dataset('ERA5/2001/remapped/ANT_All_vars_2001_remapped.nc')[['lat','lon']]
lat2D, lon2D = data_vars['lat'].values, data_vars['lon'].values
data_vars.close()

n_years = len(years)
features = ['AR', 'CY', 'ACY', 'FR', 'JET', 'OTHER']

for data_type in datasets:

    save_filename = f'{Out_folder}/EPE_Yearly_Results_{data_type}_{percentile}.npz'
    
    # Check if we already have this data
    if os.path.exists(save_filename):
        print(f"\n📂 Yearly results for {data_type} already exist. Skipping to next dataset...")
        continue
    
    print(f"PROCESSING DATASET: {data_type}")

    # We store the FRACTION (%) for each year to compare them later
    yearly_fractions = {feat: np.zeros((n_years, *lat2D.shape), dtype=np.float32) for feat in features}
    
    # Initialize accumulators for this specific dataset
    accumulators = {feat: np.zeros(lat2D.shape, dtype=np.int32) for feat in features}
    epe_count_acc = np.zeros(lat2D.shape, dtype=np.int32)

    # Set paths
    moaap_AR_folder = os.path.abspath(f'moaap_output_{data_type}_test_AR_IVT_relaxed')
    moaap_CY_folder = os.path.abspath(f'moaap_output_{data_type}_test_CY')
    moaap_FR_folder = os.path.abspath(f'moaap_output_{data_type}_test_FR')
    moaap_JET_folder = os.path.abspath(f'moaap_output_{data_type}_test_JET_300')
    moaap_PR_folder = os.path.abspath(f'moaap_output_{data_type}_test_EPE_dry_thr0.1_{percentile}')

    for y_idx, year in enumerate(tqdm(years, desc=f"Years for {data_type}")):
        # Temporary counters for just THIS year
        year_acc = {feat: np.zeros(lat2D.shape, dtype=np.int32) for feat in features}
        year_epe_total = np.zeros(lat2D.shape, dtype=np.int32)

        # Load MOAAP object masks
        if data_type == "ERA5":
            epe_file = f'{moaap_PR_folder}/{year}/{year}_ERA5_ANT_{year}_EPE_ObjectMasks__dt-1h_MOAAP-masks.nc'
        else:
            epe_file = f'{moaap_PR_folder}/{year}/{year}_{model_name}_mask.nc'

        try:
            with xr.open_dataset(f'{moaap_AR_folder}/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc') as AR_ds, \
                 xr.open_dataset(f'{moaap_CY_folder}/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc') as CY_ds, \
                 xr.open_dataset(f'{moaap_FR_folder}/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc') as FR_ds, \
                 xr.open_dataset(f'{moaap_JET_folder}/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc') as JET_ds, \
                 xr.open_dataset(epe_file) as pr_ds:
                
                AR_f = AR_ds['AR_Objects'].values.astype(bool)
                CY_f = CY_ds['CY_z500_Objects'].values.astype(bool)
                ACY_f = CY_ds['ACY_z500_Objects'].values.astype(bool)
                FR_f = FR_ds['FR_Objects'].values.astype(bool)
                JET_f = JET_ds['JET_300_Objects'].values.astype(bool)
                EPE_objects = pr_ds['EPE_mask'].values
                t6, t1 = AR_ds['time'].values, pr_ds['time'].values
        except Exception as e:
            print(f"Error loading data for year {year} and dataset {data_type}: {e}")
            continue

        # Buffer and Expand
        AR_h = expand_features_to_hourly(buffer_3d_mask(AR_f, BUFFER_KM), t6, t1)
        CY_h = expand_features_to_hourly(buffer_3d_mask(CY_f, BUFFER_KM), t6, t1)
        ACY_h = expand_features_to_hourly(buffer_3d_mask(ACY_f, BUFFER_KM), t6, t1)
        FR_h = expand_features_to_hourly(buffer_3d_mask(FR_f, BUFFER_KM), t6, t1)
        JET_h = expand_features_to_hourly(buffer_3d_mask(JET_f, BUFFER_KM), t6, t1)

        # Pixel-wise association (same logic, but using year_acc)
        for t in range(len(t1)):
            if not np.any(EPE_objects[t] == 1): continue
            labeled, num_objects = label(EPE_objects[t] == 1, structure=np.ones((3,3)))

            for obj_id in range(1, num_objects+1):
                mask = (labeled == obj_id)
                year_epe_total += mask.astype(int)
                hit = False
                if np.any(mask & AR_h[t]):  year_acc['AR'] += mask.astype(int); hit = True
                if np.any(mask & CY_h[t]):  year_acc['CY'] += mask.astype(int); hit = True
                if np.any(mask & ACY_h[t]): year_acc['ACY'] += mask.astype(int); hit = True
                if np.any(mask & FR_h[t]):  year_acc['FR'] += mask.astype(int); hit = True
                if np.any(mask & JET_h[t]): year_acc['JET'] += mask.astype(int); hit = True
                if not hit: year_acc['OTHER'] += mask.astype(int)
        
        # At the end of each year, calculate the % for that year and store in 3D array
        for feat in features:
            # Calculate fraction: (Feature Count / Total EPEs) * 100
            yearly_fractions[feat][y_idx] = np.divide(
                year_acc[feat], year_epe_total, 
                out=np.zeros_like(lat2D, dtype=float), 
                where=year_epe_total != 0
            ) * 100

        gc.collect()

    print(f"💾 Saving YEARLY results for {data_type} to {save_filename}...")
    np.savez(save_filename, **yearly_fractions)

    print("✅ Data Generation Complete.")

