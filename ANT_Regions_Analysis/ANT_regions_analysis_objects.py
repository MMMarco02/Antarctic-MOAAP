'''
# OLD VERSION: Finds the top 100 events in each region without looking at EPE Objects (only for reference, not used in final analysis)
import xarray as xr
import numpy as np
import regionmask
import geopandas as gpd
import pandas as pd
import cartopy.crs as ccrs
from dask.diagnostics import ProgressBar
import os
import matplotlib.pyplot as plt

def expand_features_to_hourly(feature_mask_6h, time_6h, time_1h):
    # Expand a 6-hourly feature mask to hourly timesteps.
    # feature_mask_6h: np.array(time6, lat, lon)
    # time_6h: array of datetime64[ns] for 6-hourly data
    # time_1h: array of datetime64[ns] for hourly data
    # returns: np.array(time1, lat, lon) expanded

    # FIX: Ensure inputs are boolean
    feature_mask_6h = feature_mask_6h.astype(bool)

    ntime1 = len(time_1h)
    nlat, nlon = feature_mask_6h.shape[1:]
    feature_hourly = np.zeros((ntime1, nlat, nlon), dtype=bool)
    
    for i, t6 in enumerate(time_6h):
        # Find closest hourly index
        idx_center = np.argmin(np.abs(time_1h - t6))
        # Expand ±2 hours
        idx_start = max(0, idx_center - 2)
        idx_end = min(ntime1, idx_center + 4) # +4 because slicing excludes end
        feature_hourly[idx_start:idx_end] |= feature_mask_6h[i]

    return feature_hourly


def is_feature_nearby(peak_lat, peak_lon, feature_mask, grid_lats, grid_lons, radius_km=500):
    """
    Checks if any True pixel in feature_mask is within radius_km of the peak.
    """
    # 1. Haversine formula to find distance between peak and all grid points
    R = 6371.0 # Earth radius in km
    phi1, phi2 = np.radians(peak_lat), np.radians(grid_lats)
    dphi = np.radians(grid_lats - peak_lat)
    dlambda = np.radians(grid_lons - peak_lon)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    dist = 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    # 2. Filter mask to only pixels within the radius
    nearby_mask = dist <= radius_km
    
    # 3. Check if any feature pixel exists within that nearby area
    return np.any(feature_mask & nearby_mask)

def compute_distance_mask(peak_lat, peak_lon, grid_lats, grid_lons, radius_km=500):
    R = 6371.0
    phi1 = np.radians(peak_lat)
    phi2 = np.radians(grid_lats)
    dphi = np.radians(grid_lats - peak_lat)
    dlambda = np.radians(grid_lons - peak_lon)

    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    dist = 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return dist <= radius_km

# --- 1. SETUP & MASKING (Robust Bounding Box) ---
data_type = "MetUM" 

shp_path = "/cluster/work/users/mmarco02/moaap_project/IPCC-WGI-reference-regions-v4/IPCC-WGI-reference-regions-v4.shp"
df_all = gpd.read_file(shp_path)
df_regions = df_all.query("Acronym in ['EAN', 'WAN', 'SOO']")
ar6_regions = regionmask.from_geopandas(df_regions, names="Name", abbrevs="Acronym")

# Load the "Big Picture" grid
sample_ds = xr.open_dataset("/cluster/work/users/mmarco02/moaap_project/ERA5/2001/remapped/ANT_pr_2001_remapped.nc")

# Setup Coordinates
rotated_crs = ccrs.RotatedPole(pole_latitude=5.0, pole_longitude=20.0)
pc_crs = ccrs.PlateCarree()
rlons, rlats = np.meshgrid(sample_ds.rlon, sample_ds.rlat)
lon_lat_coords = pc_crs.transform_points(rotated_crs, rlons, rlats)
lons_2d = lon_lat_coords[:, :, 0]
lats_2d = lon_lat_coords[:, :, 1]

# --- NEW CLIPPING LOGIC ---
# Load the inner domain to get its EXACT boundaries
domain_ds = xr.open_dataset("/cluster/work/users/mmarco02/moaap_project/PolarRES_WP3_Antarctic_domain.nc")

# Get the min/max of the rotated coordinates from the domain file
rlon_min, rlon_max = domain_ds.rlon.min().item(), domain_ds.rlon.max().item()
rlat_min, rlat_max = domain_ds.rlat.min().item(), domain_ds.rlat.max().item()

# Create a boolean mask on the ERA5 grid
# We add .transpose('rlat', 'rlon') to force the (1061, 1319) shape
inner_domain_mask_da = (
    (sample_ds.rlon >= rlon_min) & (sample_ds.rlon <= rlon_max) &
    (sample_ds.rlat >= rlat_min) & (sample_ds.rlat <= rlat_max)
).transpose('rlat', 'rlon')

# Now extract the numpy values
inner_domain_mask_aligned = inner_domain_mask_da.values


# If your rlon/rlat are 1D in sample_ds, xarray will broadcast this to 2D automatically.
# If they are already 2D, this works as well.

# Generate and CLIP the masks
mask_3d = ar6_regions.mask_3D(lons_2d, lats_2d)
# Verification (Optional but recommended)
print(f"DEBUG: inner_mask shape: {inner_domain_mask_aligned.shape}")
print(f"DEBUG: mask_3d shape: {mask_3d.sel(region=ar6_regions.map_keys('EAN')).squeeze().shape}")

ean_mask_data = mask_3d.sel(region=ar6_regions.map_keys("EAN")).squeeze().data & inner_domain_mask_aligned
wan_mask_data = mask_3d.sel(region=ar6_regions.map_keys("WAN")).squeeze().data & inner_domain_mask_aligned
soo_mask_data = mask_3d.sel(region=ar6_regions.map_keys("SOO")).squeeze().data & inner_domain_mask_aligned

# --- 2. LOAD ALL YEARS ---
if data_type == "ERA5":
    file_pattern = "/cluster/work/users/mmarco02/moaap_project/ERA5/*/remapped/ANT_pr_*_remapped.nc"
    pr_name = 'tp'
else:
    file_pattern = f"/cluster/work/users/mmarco02/moaap_project/PolarRES/{data_type}/nested/*/ANT_{data_type}_pr_*_nested.nc"
    pr_name = 'pr'

ds_full = xr.open_mfdataset(file_pattern, combine='by_coords', chunks={'valid_time': 500})
ds_full = ds_full.assign_coords(lon=(("rlat", "rlon"), lons_2d), lat=(("rlat", "rlon"), lats_2d))

# --- 3. THE ANALYTICS FUNCTION ---
def find_top_events_declustered(data_type, ds, mask_np_array, region_name):
    print(f"Finding top events for {region_name}; data: {data_type}...")
    
    # SAFETY NET 1: Fast-fail if the mask is totally empty
    if not np.any(mask_np_array):
        print(f"⚠️ Warning: Mask for {region_name} has no valid points in the inner domain! Skipping.")
        return pd.DataFrame() # Return empty dataframe

    with ProgressBar():
        regional_max = ds[pr_name].where(mask_np_array).max(dim=['rlat', 'rlon']).compute()

    # SAFETY NET 2: Drop NaNs from the max series so we don't iterate over empty timesteps
    max_series = (regional_max * 1000).to_series().dropna().sort_values(ascending=False)
    selected = []

    for time, val in max_series.items():
        if len(selected) >= 100:
            break
            
        if all(abs((time - s['time']).total_seconds()) > 3*24*3600 for s in selected):
            time_name = "valid_time" if data_type == "ERA5" else "time"
            snapshot = ds[pr_name].sel(**{time_name: time}).compute()
            snapshot_masked = snapshot.where(mask_np_array)
            
            # SAFETY NET 3: Final check before running nanargmax
            if np.isnan(snapshot_masked.values).all():
                print(f"Skipping {time} for {region_name}: entirely NaNs.")
                continue
                
            idx = np.unravel_index(np.nanargmax(snapshot_masked.values), snapshot_masked.shape)
            event_lat = lats_2d[idx[0], idx[1]]
            event_lon = lons_2d[idx[0], idx[1]]
            
            selected.append({
                'time': time,
                'precip_mm': float(val),
                'lat': float(event_lat),
                'lon': float(event_lon)
            })
            
            if len(selected) <= 10:
                print(f"Event {len(selected)}: {time} - {val:.2f} mm at ({event_lat:.2f}, {event_lon:.2f})")

    return pd.DataFrame(selected)

# Define file names
ean_csv = f"top100_EAN_2001_2020_{data_type}.csv"
wan_csv = f"top100_WAN_2001_2020_{data_type}.csv"
soo_csv = f"top100_SOO_2001_2020_{data_type}.csv"

# --- 4. EXECUTION WITH CHECK ---
if os.path.exists(ean_csv):
    print("Found existing top 100 CSV files for EAN. Loading data...")
    events_EAN = pd.read_csv(ean_csv, parse_dates=['time'])
else:
    print("Top 100 files for EAN not found. Starting computation...")
    events_EAN = find_top_events_declustered(data_type, ds_full, ean_mask_data, "EAN")
    events_EAN.to_csv(ean_csv, index=False)

if os.path.exists(wan_csv):
    print("Found existing top 100 CSV files. Loading data...")
    events_WAN = pd.read_csv(wan_csv, parse_dates=['time'])
else:
    print("Top 100 files for WAN not found. Starting computation...")
    events_WAN = find_top_events_declustered(data_type, ds_full, wan_mask_data, "WAN")
    events_WAN.to_csv(wan_csv, index=False)

if os.path.exists(soo_csv):
    print("Found existing top 100 CSV files. Loading data...")
    events_SOO = pd.read_csv(soo_csv, parse_dates=['time'])
else:
    print("Top 100 files for SOO not found. Starting computation...")
    events_SOO = find_top_events_declustered(data_type, ds_full, soo_mask_data, "SOO")
    events_SOO.to_csv(soo_csv, index=False)

# --- 5. INITIALIZE RESULTS COLUMNS & TRACKING ---
final_files = {
    "EAN": f"top100_EAN_with_features_{data_type}.csv",
    "WAN": f"top100_WAN_with_features_{data_type}.csv",
    "SOO": f"top100_SOO_with_features_{data_type}.csv"
}

regions_to_process = [reg for reg, path in final_files.items() if not os.path.exists(path)]

if not regions_to_process:
    print("Step 2: All final feature-associated CSVs found.")
    events_EAN = pd.read_csv(final_files["EAN"], parse_dates=['time'])
    events_WAN = pd.read_csv(final_files["WAN"], parse_dates=['time'])
    events_SOO = pd.read_csv(final_files["SOO"], parse_dates=['time'])
else:
    for reg in regions_to_process:
        df = {"EAN": events_EAN, "WAN": events_WAN, "SOO": events_SOO}[reg]
        for col in ['AR', 'CY', 'ACY', 'FR', 'JET']:
            if col not in df.columns:
                df[col] = False

    # --- 6. YEARLY ASSOCIATION LOOP ---
    years = range(2001, 2021)
    moaap_AR_folder = os.path.abspath(f'moaap_output_{data_type}_test_AR_IVT_relaxed')
    moaap_CY_folder = os.path.abspath(f'moaap_output_{data_type}_test_CY')
    moaap_FR_folder = os.path.abspath(f'moaap_output_{data_type}_test_FR')
    moaap_JET_folder = os.path.abspath(f'moaap_output_{data_type}_test_JET_300')

    for year in years:
        year_events_EAN = events_EAN[pd.to_datetime(events_EAN['time']).dt.year == year]
        year_events_WAN = events_WAN[pd.to_datetime(events_WAN['time']).dt.year == year]
        year_events_SOO = events_SOO[pd.to_datetime(events_SOO['time']).dt.year == year]
        
        if len(year_events_EAN) == 0 and len(year_events_WAN) == 0 and len(year_events_SOO) == 0:
            print(f"No top events found in {year}. Skipping...")
            continue

        print(f"Processing features for year {year}...")
        time_name = "valid_time" if data_type == "ERA5" else "time"
        current_year_ds = ds_full.sel(**{time_name: slice(f"{year}-01-01", f"{year}-12-31")})
        time_1h_current = current_year_ds[time_name].values

        try:
            with xr.open_dataset(f'{moaap_AR_folder}/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc') as AR_ds, \
                 xr.open_dataset(f'{moaap_CY_folder}/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc') as CY_ds, \
                 xr.open_dataset(f'{moaap_FR_folder}/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc') as FR_ds, \
                 xr.open_dataset(f'{moaap_JET_folder}/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc') as JET_ds:

                time_6hr = AR_ds['time'].values
                AR_h = expand_features_to_hourly(AR_ds['AR_Objects'].values, time_6hr, time_1h_current)
                CY_h = expand_features_to_hourly(CY_ds['CY_z500_Objects'].values, time_6hr, time_1h_current)
                ACY_h = expand_features_to_hourly(CY_ds['ACY_z500_Objects'].values, time_6hr, time_1h_current)
                FR_h = expand_features_to_hourly(FR_ds['FR_Objects'].values, time_6hr, time_1h_current)
                JET_h = expand_features_to_hourly(JET_ds['JET_300_Objects'].values, time_6hr, time_1h_current)

                for df_name in regions_to_process:
                    df = {"EAN": events_EAN, "WAN": events_WAN, "SOO": events_SOO}[df_name]
                    year_indices = df[pd.to_datetime(df['time']).dt.year == year].index
                    for idx in year_indices:
                        event = df.loc[idx]
                        target_idx = np.argmin(np.abs(time_1h_current - pd.to_datetime(event['time']).to_datetime64()))
                        df.at[idx, 'AR'] = is_feature_nearby(event['lat'], event['lon'], AR_h[target_idx], lats_2d, lons_2d)
                        df.at[idx, 'CY'] = is_feature_nearby(event['lat'], event['lon'], CY_h[target_idx], lats_2d, lons_2d)
                        df.at[idx, 'ACY'] = is_feature_nearby(event['lat'], event['lon'], ACY_h[target_idx], lats_2d, lons_2d)
                        df.at[idx, 'FR'] = is_feature_nearby(event['lat'], event['lon'], FR_h[target_idx], lats_2d, lons_2d)
                        df.at[idx, 'JET'] = is_feature_nearby(event['lat'], event['lon'], JET_h[target_idx], lats_2d, lons_2d)

                del AR_h, CY_h, ACY_h, FR_h, JET_h

        except Exception as e:
            print(f"Error processing year {year}: {e}")
            continue

    for reg in regions_to_process:
        df = {"EAN": events_EAN, "WAN": events_WAN, "SOO": events_SOO}[reg]
        df.to_csv(final_files[reg], index=False)

# --- 9. PLOTTING SECTION ---
final_ean_csv = f"top100_EAN_with_features_{data_type}.csv"
final_wan_csv = f"top100_WAN_with_features_{data_type}.csv"
final_soo_csv = f"top100_SOO_with_features_{data_type}.csv"

if os.path.exists(final_ean_csv) and os.path.exists(final_wan_csv) and os.path.exists(final_soo_csv):
    df_ean = pd.read_csv(final_ean_csv)
    df_wan = pd.read_csv(final_wan_csv)
    df_soo = pd.read_csv(final_soo_csv)

    features = ['AR', 'CY', 'ACY', 'FR', 'JET']
    colors = ['red', 'black', 'orange', 'plum', 'purple']

    def get_frequencies(df, feature_list):
        return [df[feat].sum() / len(df) * 100 for feat in feature_list]

    ean_stats = get_frequencies(df_ean, features)
    wan_stats = get_frequencies(df_wan, features)
    soo_stats = get_frequencies(df_soo, features)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, stats, title in zip(axes, [ean_stats, wan_stats, soo_stats], ["East Antarctica (EAN)", "West Antarctica (WAN)", "Southern Ocean (SOO)"]):
        bars = ax.bar(features, stats, color=colors, edgecolor='black', alpha=0.8)
        ax.set_title(f"{title}", fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.0f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f"Antarctica_Extremes_Drivers_Regions_{data_type}.png", dpi=300)
    plt.show()
    print("Analysis and plotting complete.")
else:
    print("Final feature files not found.")


'''

import xarray as xr
import numpy as np
import regionmask
import geopandas as gpd
import pandas as pd
import cartopy.crs as ccrs
from dask.diagnostics import ProgressBar
import os
import matplotlib.pyplot as plt

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
# This function takes a 6-hourly feature mask and expands it to an hourly mask by assuming the feature is present for 2 hours before and 3 hours after the 6-hourly timestamp.
def expand_features_to_hourly(feature_mask_6h, time_6h, time_1h):
    feature_mask_6h = feature_mask_6h.astype(bool)
    ntime1 = len(time_1h)
    nlat, nlon = feature_mask_6h.shape[1:]
    feature_hourly = np.zeros((ntime1, nlat, nlon), dtype=bool)
    
    for i, t6 in enumerate(time_6h):
        idx_center = np.argmin(np.abs(time_1h - t6))
        idx_start = max(0, idx_center - 2)
        idx_end = min(ntime1, idx_center + 4) 
        feature_hourly[idx_start:idx_end] |= feature_mask_6h[i]

    return feature_hourly

# This function checks if a given feature is present within a specified radius (default 500 km) of the EPE event's peak location. 
def is_feature_nearby(peak_lat, peak_lon, feature_mask, grid_lats, grid_lons, radius_km=500):
    R = 6371.0 
    phi1, phi2 = np.radians(peak_lat), np.radians(grid_lats)
    dphi = np.radians(grid_lats - peak_lat)
    dlambda = np.radians(grid_lons - peak_lon)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    dist = 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    nearby_mask = dist <= radius_km
    return np.any(feature_mask & nearby_mask)

# =============================================================================
# 1. SETUP & MASKING
# =============================================================================
data_type = "ERA5"  # Change to "ERA5", "HCLIM", "MetUM", or "RACMO2"

shp_path = "/cluster/work/users/mmarco02/moaap_project/IPCC-WGI-reference-regions-v4/IPCC-WGI-reference-regions-v4.shp" # Path to the shapefile containing the IPCC AR6 reference regions (make sure this path is correct on your system)
df_all = gpd.read_file(shp_path)
df_regions = df_all.query("Acronym in ['EAN', 'WAN', 'SOO']")
ar6_regions = regionmask.from_geopandas(df_regions, names="Name", abbrevs="Acronym")

# Load a sample dataset to get the grid information (assuming all datasets share the same grid after remapping/nesting)
sample_ds = xr.open_dataset("/cluster/work/users/mmarco02/moaap_project/ERA5/2001/remapped/ANT_pr_2001_remapped.nc")

rotated_crs = ccrs.RotatedPole(pole_latitude=5.0, pole_longitude=20.0)
pc_crs = ccrs.PlateCarree()
rlons, rlats = np.meshgrid(sample_ds.rlon, sample_ds.rlat)
lon_lat_coords = pc_crs.transform_points(rotated_crs, rlons, rlats)
lons_2d = lon_lat_coords[:, :, 0]
lats_2d = lon_lat_coords[:, :, 1]

# Load the Antarctic domain mask to ensure we only analyze points within the defined Antarctic region
domain_ds = xr.open_dataset("/cluster/work/users/mmarco02/moaap_project/PolarRES_WP3_Antarctic_domain.nc")
rlon_min, rlon_max = domain_ds.rlon.min().item(), domain_ds.rlon.max().item()
rlat_min, rlat_max = domain_ds.rlat.min().item(), domain_ds.rlat.max().item()

inner_domain_mask_da = (
    (sample_ds.rlon >= rlon_min) & (sample_ds.rlon <= rlon_max) &
    (sample_ds.rlat >= rlat_min) & (sample_ds.rlat <= rlat_max)
).transpose('rlat', 'rlon')

inner_domain_mask_aligned = inner_domain_mask_da.values
mask_3d = ar6_regions.mask_3D(lons_2d, lats_2d)

# Apply the inner domain mask to each region's mask to ensure we only consider points within the Antarctic domain
ean_mask_data = mask_3d.sel(region=ar6_regions.map_keys("EAN")).squeeze().data & inner_domain_mask_aligned
wan_mask_data = mask_3d.sel(region=ar6_regions.map_keys("WAN")).squeeze().data & inner_domain_mask_aligned
soo_mask_data = mask_3d.sel(region=ar6_regions.map_keys("SOO")).squeeze().data & inner_domain_mask_aligned

# =============================================================================
# 2. LOAD ALL YEARS (Precipitation & EPE Objects)
# =============================================================================
# --- Precipitation Data ---
# Note: Make sure the file patterns and variable names match your actual data structure. The code assumes that ERA5 has a different naming convention for the precipitation variable and time dimension compared to the other datasets.
if data_type == "ERA5":
    file_pattern = "/cluster/work/users/mmarco02/moaap_project/ERA5/*/remapped/ANT_pr_*_remapped.nc"
    pr_name = 'tp'
    time_dim_pr = 'valid_time'
elif data_type == "RACMO2":
    file_pattern = f"/cluster/work/users/mmarco02/moaap_project/PolarRES/{data_type}/nested/*/ANT_{data_type}_pr_*_nested_con.nc"
    pr_name = 'pr'
    time_dim_pr = 'time'
else:
    file_pattern = f"/cluster/work/users/mmarco02/moaap_project/PolarRES/{data_type}/nested/*/ANT_{data_type}_pr_*_nested.nc"
    pr_name = 'pr'
    time_dim_pr = 'time'

ds_full = xr.open_mfdataset(file_pattern, combine='by_coords', chunks={time_dim_pr: 500})
ds_full = ds_full.assign_coords(lon=(("rlat", "rlon"), lons_2d), lat=(("rlat", "rlon"), lats_2d))

# --- EPE Objects Data ---
# Note: We use the 99th EPE objects as these will for sure include the top 100 events we are looking for.
if data_type == "ERA5":
    epe_pattern = f"moaap_output_ERA5_test_EPE_dry_thr0.1_99/*/*_EPE_ObjectMasks__dt-1h_MOAAP-masks.nc"
else:
    epe_pattern = f"moaap_output_{data_type}_test_EPE_dry_thr0.1_99/*/*_{data_type}_mask.nc"

ds_epe_raw = xr.open_mfdataset(epe_pattern, combine='by_coords', chunks={'time': 500})

# Standardize dimensions to perfectly match the precip data
rename_dict = {}
if 'y' in ds_epe_raw.dims: rename_dict['y'] = 'rlat'
if 'x' in ds_epe_raw.dims: rename_dict['x'] = 'rlon'
if 'time' in ds_epe_raw.dims and data_type == "ERA5": rename_dict['time'] = 'valid_time'

ds_epe = ds_epe_raw.rename(rename_dict)
epe_mask_da = ds_epe['EPE_mask'].astype(bool) 

# =============================================================================
# 3. THE ANALYTICS FUNCTION (Object-Based)
# =============================================================================
from scipy.ndimage import label, center_of_mass, sum as nd_sum

def find_top_epe_events_labeled(data_type, ds_pr, epe_da, mask_np_array, region_name):
    print(f"Finding top labeled EPE objects for {region_name}...")
    
    time_name = "valid_time" if data_type == "ERA5" else "time"
    all_candidates = []

    # Process year-by-year for memory safety
    for year in range(2001, 2021):
        print(f"  -> Processing year {year}...")
        pr_year = (ds_pr[pr_name].sel(**{time_name: slice(f"{year}-01-01", f"{year}-12-31")}) * 1000).compute()
        epe_year = epe_da.sel(**{time_name: slice(f"{year}-01-01", f"{year}-12-31")}).compute()
        
        # Apply the static region mask to the dynamic EPE mask
        # This ensures we only look at objects (or parts of objects) inside the region
        regional_epe = epe_year & mask_np_array
        
        for t in range(len(regional_epe[time_name])):
            mask_slice = regional_epe.values[t]
            pr_slice = pr_year.values[t]
            
            if not np.any(mask_slice):
                continue
                
            # 1. Label individual objects in this timestep
            labeled_array, num_features = label(mask_slice)
            
            if num_features == 0:
                continue
                
            # 2. Calculate the sum of precip for each individual object ID
            obj_ids = np.arange(1, num_features + 1)
            obj_sums = nd_sum(pr_slice, labeled_array, index=obj_ids)
            
            # 3. Find the biggest object at this specific hour
            max_idx = np.argmax(obj_sums)
            best_id = obj_ids[max_idx]
            best_sum = obj_sums[max_idx]
            
            # 4. Calculate centroid for ONLY that specific object
            single_obj_mask = (labeled_array == best_id)
            c_y, c_x = center_of_mass(single_obj_mask)
            
            all_candidates.append({
                'time': regional_epe[time_name].values[t],
                'total_precip_obj': float(best_sum),
                'max_intensity_mm': float(np.max(pr_slice[single_obj_mask])), # Peak mm/hr in this specific object
                'lat': float(lats_2d[int(c_y), int(c_x)]),
                'lon': float(lons_2d[int(c_y), int(c_x)])
            })

    # Convert to DataFrame, sort globally, and decluster by 3 days
    df = pd.DataFrame(all_candidates)
    if df.empty: return df
    
    df = df.sort_values(by='total_precip_obj', ascending=False).reset_index(drop=True)
    
    selected = []
    for _, row in df.iterrows():
        if len(selected) >= 100: break
        if all(abs((pd.to_datetime(row['time']) - pd.to_datetime(s['time'])).total_seconds()) > 3*24*3600 for s in selected):
            selected.append(row.to_dict())
            
    return pd.DataFrame(selected)

# =============================================================================
# 3. THE ANALYTICS FUNCTION (Peak Intensity & Peak Location)
# =============================================================================
from scipy.ndimage import label

def find_top_epe_events_peak_based(data_type, ds_pr, epe_da, mask_np_array, region_name):
    print(f"Finding top EPE objects based on PEAK intensity for {region_name}...")
    
    time_name = "valid_time" if data_type == "ERA5" else "time"
    all_candidates = []

    for year in range(2001, 2021):
        print(f"  -> Processing year {year}...")
        pr_year = (ds_pr[pr_name].sel(**{time_name: slice(f"{year}-01-01", f"{year}-12-31")}) * 1000).compute()
        epe_year = epe_da.sel(**{time_name: slice(f"{year}-01-01", f"{year}-12-31")}).compute()
        
        regional_epe = epe_year & mask_np_array
        
        for t in range(len(regional_epe[time_name])):
            mask_slice = regional_epe.values[t]
            pr_slice = pr_year.values[t]
            
            if not np.any(mask_slice):
                continue
                
            # Label objects so we only look at valid EPE clusters
            labeled_array, num_features = label(mask_slice)
            if num_features == 0: continue
            
            # Mask the precip data to only include valid EPE objects
            pr_objects_only = np.where(labeled_array > 0, pr_slice, np.nan)
            
            if np.all(np.isnan(pr_objects_only)): continue
            
            # Find the global maximum intensity for this hour within any EPE object
            max_val = np.nanmax(pr_objects_only)
            
            # Get coordinates of that peak pixel
            # (Note: if multiple pixels have the same max, we take the first one)
            y_idx, x_idx = np.unravel_index(np.nanargmax(pr_objects_only), pr_objects_only.shape)
            
            all_candidates.append({
                'time': regional_epe[time_name].values[t],
                'max_intensity_mm': float(max_val),
                'lat': float(lats_2d[y_idx, x_idx]),
                'lon': float(lons_2d[y_idx, x_idx])
            })

    df = pd.DataFrame(all_candidates)
    if df.empty: return df
    
    # SORT BY PEAK INTENSITY
    df = df.sort_values(by='max_intensity_mm', ascending=False).reset_index(drop=True)
    
    # Decluster by 3 days
    selected = []
    for _, row in df.iterrows():
        if len(selected) >= 100: break
        if all(abs((pd.to_datetime(row['time']) - pd.to_datetime(s['time'])).total_seconds()) > 3*24*3600 for s in selected):
            selected.append(row.to_dict())
            
    return pd.DataFrame(selected)

# Define file names ('Objects' suffix to remind that the top 100 events are based on the peak intensity and location within the EPE objects, not just the raw precip values at the grid points)
ean_csv = f"top100_EAN_2001_2020_{data_type}_Objects.csv"
wan_csv = f"top100_WAN_2001_2020_{data_type}_Objects.csv"
soo_csv = f"top100_SOO_2001_2020_{data_type}_Objects.csv"

# =============================================================================
# 4. EXECUTION WITH CHECK
# =============================================================================
if os.path.exists(ean_csv):
    print("Found existing top 100 EPE CSV for EAN. Loading data...")
    events_EAN = pd.read_csv(ean_csv, parse_dates=['time'])
else:
    events_EAN = find_top_epe_events_peak_based(data_type, ds_full, epe_mask_da, ean_mask_data, "EAN")
    events_EAN.to_csv(ean_csv, index=False)

if os.path.exists(wan_csv):
    print("Found existing top 100 EPE CSV for WAN. Loading data...")
    events_WAN = pd.read_csv(wan_csv, parse_dates=['time'])
else:
    events_WAN = find_top_epe_events_peak_based(data_type, ds_full, epe_mask_da, wan_mask_data, "WAN")
    events_WAN.to_csv(wan_csv, index=False)

if os.path.exists(soo_csv):
    print("Found existing top 100 EPE CSV for SOO. Loading data...")
    events_SOO = pd.read_csv(soo_csv, parse_dates=['time'])
else:
    events_SOO = find_top_epe_events_peak_based(data_type, ds_full, epe_mask_da, soo_mask_data, "SOO")
    events_SOO.to_csv(soo_csv, index=False)

# =============================================================================
# 5. FEATURE ASSOCIATION LOOP
# =============================================================================
final_files = {
    "EAN": f"top100_EAN_with_features_{data_type}_Objects.csv",
    "WAN": f"top100_WAN_with_features_{data_type}_Objects.csv",
    "SOO": f"top100_SOO_with_features_{data_type}_Objects.csv"
}

regions_to_process = [reg for reg, path in final_files.items() if not os.path.exists(path)]

# If all final files exist, we can skip the feature association loop and directly load the data for plotting. 
if not regions_to_process:
    print("All final feature-associated EPE CSVs found.")
    events_EAN = pd.read_csv(final_files["EAN"], parse_dates=['time'])
    events_WAN = pd.read_csv(final_files["WAN"], parse_dates=['time'])
    events_SOO = pd.read_csv(final_files["SOO"], parse_dates=['time'])
else:
    for reg in regions_to_process:
        df = {"EAN": events_EAN, "WAN": events_WAN, "SOO": events_SOO}[reg]
        for col in ['AR', 'CY', 'ACY', 'FR', 'JET']:
            if col not in df.columns:
                df[col] = False

    years = range(2001, 2021)
    moaap_AR_folder = os.path.abspath(f'moaap_output_{data_type}_test_AR_IVT_relaxed')
    moaap_CY_folder = os.path.abspath(f'moaap_output_{data_type}_test_CY')
    moaap_FR_folder = os.path.abspath(f'moaap_output_{data_type}_test_FR')
    moaap_JET_folder = os.path.abspath(f'moaap_output_{data_type}_test_JET_300')

    for year in years:
        year_events_EAN = events_EAN[pd.to_datetime(events_EAN['time']).dt.year == year]
        year_events_WAN = events_WAN[pd.to_datetime(events_WAN['time']).dt.year == year]
        year_events_SOO = events_SOO[pd.to_datetime(events_SOO['time']).dt.year == year]
        
        if len(year_events_EAN) == 0 and len(year_events_WAN) == 0 and len(year_events_SOO) == 0:
            continue

        print(f"Processing meteorological features for year {year}...")
        time_name = "valid_time" if data_type == "ERA5" else "time"
        current_year_ds = ds_full.sel(**{time_name: slice(f"{year}-01-01", f"{year}-12-31")})
        time_1h_current = current_year_ds[time_name].values

        try:
            with xr.open_dataset(f'{moaap_AR_folder}/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc') as AR_ds, \
                 xr.open_dataset(f'{moaap_CY_folder}/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc') as CY_ds, \
                 xr.open_dataset(f'{moaap_FR_folder}/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc') as FR_ds, \
                 xr.open_dataset(f'{moaap_JET_folder}/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc') as JET_ds:

                time_6hr = AR_ds['time'].values
                AR_h = expand_features_to_hourly(AR_ds['AR_Objects'].values, time_6hr, time_1h_current)
                CY_h = expand_features_to_hourly(CY_ds['CY_z500_Objects'].values, time_6hr, time_1h_current)
                ACY_h = expand_features_to_hourly(CY_ds['ACY_z500_Objects'].values, time_6hr, time_1h_current)
                FR_h = expand_features_to_hourly(FR_ds['FR_Objects'].values, time_6hr, time_1h_current)
                JET_h = expand_features_to_hourly(JET_ds['JET_300_Objects'].values, time_6hr, time_1h_current)

                for df_name in regions_to_process:
                    df = {"EAN": events_EAN, "WAN": events_WAN, "SOO": events_SOO}[df_name]
                    year_indices = df[pd.to_datetime(df['time']).dt.year == year].index
                    for idx in year_indices:
                        event = df.loc[idx]
                        target_idx = np.argmin(np.abs(time_1h_current - pd.to_datetime(event['time']).to_datetime64()))
                        df.at[idx, 'AR'] = is_feature_nearby(event['lat'], event['lon'], AR_h[target_idx], lats_2d, lons_2d)
                        df.at[idx, 'CY'] = is_feature_nearby(event['lat'], event['lon'], CY_h[target_idx], lats_2d, lons_2d)
                        df.at[idx, 'ACY'] = is_feature_nearby(event['lat'], event['lon'], ACY_h[target_idx], lats_2d, lons_2d)
                        df.at[idx, 'FR'] = is_feature_nearby(event['lat'], event['lon'], FR_h[target_idx], lats_2d, lons_2d)
                        df.at[idx, 'JET'] = is_feature_nearby(event['lat'], event['lon'], JET_h[target_idx], lats_2d, lons_2d)

                del AR_h, CY_h, ACY_h, FR_h, JET_h

        except Exception as e:
            print(f"Error processing year {year}: {e}")
            continue

    for reg in regions_to_process:
        df = {"EAN": events_EAN, "WAN": events_WAN, "SOO": events_SOO}[reg]
        df.to_csv(final_files[reg], index=False)

# =============================================================================
# 6. PLOTTING SECTION
# =============================================================================
if os.path.exists(final_files["EAN"]) and os.path.exists(final_files["WAN"]) and os.path.exists(final_files["SOO"]):
    df_ean = pd.read_csv(final_files["EAN"])
    df_wan = pd.read_csv(final_files["WAN"])
    df_soo = pd.read_csv(final_files["SOO"])

    features = ['AR', 'CY', 'ACY', 'FR', 'JET']
    colors = ['red', 'black', 'orange', 'plum', 'purple']

    def get_frequencies(df, feature_list):
        return [df[feat].sum() / len(df) * 100 for feat in feature_list]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, stats, title in zip(axes, 
                                [get_frequencies(df_ean, features), get_frequencies(df_wan, features), get_frequencies(df_soo, features)], 
                                ["East Antarctica (EAN)", "West Antarctica (WAN)", "Southern Ocean (SOO)"]):
        bars = ax.bar(features, stats, color=colors, edgecolor='black', alpha=0.8)
        ax.set_title(f"{title}", fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.0f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f"Antarctica_Extremes_Drivers_Regions_{data_type}_Objects.png", dpi=300)
    plt.show()
    print("Analysis and plotting complete.")
else:
    print("Final feature files not found.")