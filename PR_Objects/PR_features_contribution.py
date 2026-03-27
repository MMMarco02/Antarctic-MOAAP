
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
from scipy.ndimage import distance_transform_edt, gaussian_filter
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import label

start_time = time.time()

print("All imports successful")


####################################################################################################
# HELPER FUNCTIONS NEEDED FOR THE ANALYSIS
####################################################################################################

# This function sums hourly precipitation into 6-hour bins centered at 00, 06, 12, and 18 UTC.
# Specifically:
    #00 = sum(22,23,00,01,02,03)
    #06 = sum(04,05,06,07,08,09)
    #12 = sum(10,11,12,13,14,15)
    #18 = sum(16,17,18,19,20,21)

def sum_to_6h_centered(pr_hourly, time_dim='valid_time'):

    pr = pr_hourly.copy()
    pr = pr.sortby(time_dim)

    # Start / end
    start = pr[time_dim][0].values.astype("datetime64[h]").astype("datetime64[ns]")
    end = pr[time_dim][-1].values

    # Build a complete hourly index
    full_hours = pd.date_range(start=pd.Timestamp(start), end=pd.Timestamp(end), freq="h")

    # Reindex to full hours
    pr = pr.reindex({time_dim: full_hours}, fill_value=0)

    # Prepare output
    sums = []
    times = []

    centers = [0, 6, 12, 18]

    for day in pd.date_range(start=pd.Timestamp(start).normalize(),
                             end=pd.Timestamp(end).normalize(), freq="D"):

        for c in centers:
            t_start = day + pd.Timedelta(hours=c-2)
            t_end   = day + pd.Timedelta(hours=c+3)

            # Select with valid_time
            window = pr.sel({time_dim: slice(t_start, t_end)})

            sums.append(window.sum(dim=time_dim))
            times.append(day + pd.Timedelta(hours=c))

    # Concat along valid_time
    pr6 = xr.concat(sums, dim=time_dim)
    pr6 = pr6.assign_coords({time_dim: (time_dim, times)})

    return pr6

# This function computes a Gaussian-smoothed influence map from a binary feature mask. 
# The output values will be between 0 (no influence) and 1 (center of influence).
def compute_gaussian_influence(mask, sigma_km, dx_km=11.1):

    if not np.any(mask):
        return np.zeros_like(mask, dtype=np.float32)

    # Convert sigma (in km) to sigma (in grid cells)
    sigma_grid_cells = sigma_km / dx_km

    # Apply the Gaussian filter.
    influence_map = gaussian_filter(
        input=mask.astype(np.float32), 
        sigma=sigma_grid_cells, 
        mode='constant', 
        cval=0.0
    )
    
    return influence_map

# This function computes a binary hard-buffer mask around a feature, using a constant buffer distance. It returns True for grid cells that are within the specified buffer distance (in km) from any part of the feature, 
# and False otherwise.
def compute_buffer_mask(mask, buffer_km, dx_km=11.1):

    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    # distance (grid cells) to nearest object cell
    dist_idx = distance_transform_edt(~mask)

    # convert to km
    dist_km = dist_idx * dx_km

    # return buffer
    return dist_km <= buffer_km

    

# Build rectangle vertices (used for plotting domain outlines)
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


print("🧠 Starting PR-features contribution analysis...")

# --- USER CONFIG ---
data_type = "ERA5" # ERA5 or HCLIM or MetUM or RACMO2
# Outut folder
OutputFolder = os.path.abspath(f'PR_features_analysis_output_{data_type}_2001_2020_amount_contribution') + '/'
os.makedirs(OutputFolder, exist_ok=True)


# Features & pr_objects folders 
moaap_AR_folder = os.path.abspath(f'moaap_output_{data_type}_test_AR_IVT_relaxed')
moaap_CY_folder = os.path.abspath(f'moaap_output_{data_type}_test_CY')
moaap_FR_folder = os.path.abspath(f'moaap_output_{data_type}_test_FR')
#moaap_JET_folder = os.path.abspath(f'moaap_output_{data_type}_test_JET')
moaap_JET_folder = os.path.abspath(f'moaap_output_{data_type}_test_JET_300')
moaap_PR_folder = os.path.abspath(f'moaap_output_{data_type}_test_PR_all_objects_summed') 
contrib_AR = contrib_CY = contrib_ACY = contrib_FR = contrib_JET = contrib_OTHER = None
total_precip_from_PR = None

# Buffering parameters needed
data_vars = xr.open_dataset(f'ERA5/2001/remapped/ANT_All_vars_2001_remapped.nc')[['lat','lon']] # same for all
lat2D = data_vars['lat'].values
lon2D = data_vars['lon'].values
data_vars.close()

years = np.arange(2001, 2021)  # 2001-2020

for year in tqdm(years, desc="Processing years"):
    print(f"\n Processing year {year}...")
    
    # 1. Determine the conditional path first
    if data_type == "ERA5":
        pr_path = f'ERA5/{year}/remapped/ANT_pr_{year}_remapped.nc'
        pr_name = 'tp'
        time_name = 'valid_time'
    elif data_type == "HCLIM":
        pr_path = f'PolarRES/HCLIM/nested/{year}/ANT_HCLIM_pr_{year}_nested.nc'
        pr_name = 'pr'
        time_name = 'time'
    elif data_type == "MetUM":
        pr_path = f'PolarRES/MetUM/nested/{year}/ANT_MetUM_pr_{year}_nested.nc'
        pr_name = 'pr'
        time_name = 'time'
    elif data_type == "RACMO2":
        pr_path = f'PolarRES/RACMO2/nested/{year}/ANT_RACMO2_pr_{year}_nested_con.nc'
        pr_name = 'pr'
        time_name = 'time'
    else:
        print(f"  ❌ Precipitation data type not recognized, skipping year {year}.")
        continue # Or handle error

    try:
        # 2. Open everything in one clean block
        with xr.open_dataset(f'{moaap_AR_folder}/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc') as AR_ds, \
             xr.open_dataset(f'{moaap_CY_folder}/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc') as CY_ds, \
             xr.open_dataset(f'{moaap_FR_folder}/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc') as FR_ds, \
             xr.open_dataset(f'{moaap_JET_folder}/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc') as JET_ds, \
             xr.open_dataset(f'{moaap_PR_folder}/{year}/{year}_{data_type}_ANT_{year}_PR_ObjectMasks__dt-6h_MOAAP-masks.nc') as pr_ds, \
             xr.open_dataset(pr_path) as pr_data:


            # --- Robust Harmonization ---
            # 1. Define the master template. 
            master_coords = {
                time_name: pr_ds['time'],  # Use 'time' as the key since feature files use 'time'
                'lat': (('rlat', 'rlon'), lat2D),
                'lon': (('rlat', 'rlon'), lon2D)
            }

            # 2. Assign and then drop any 'old' rlat/rlon that might conflict
            AR_ds = AR_ds.assign_coords(master_coords)
            CY_ds = CY_ds.assign_coords(master_coords)
            FR_ds = FR_ds.assign_coords(master_coords)
            JET_ds = JET_ds.assign_coords(master_coords)

            ds_all = xr.merge([
            AR_ds[['AR_Objects']],
            CY_ds[['CY_z500_Objects','ACY_z500_Objects']],
            FR_ds[['FR_Objects']],
            #JET_ds[['JETObjects']],
            JET_ds[['JET_300_Objects']],
            pr_ds[['PR_mask']]
            ], join="inner", compat="override")   # ensures exact same times
            
            # Extract data
            AR_features = AR_ds['AR_Objects'].values
            CY_features = CY_ds['CY_z500_Objects'].values
            ACY_features = CY_ds['ACY_z500_Objects'].values
            FR_features = FR_ds['FR_Objects'].values
            #JET_features = JET_ds['JET_Objects'].values
            JET_features = JET_ds['JET_300_Objects'].values
            pr_values = pr_data[pr_name].values 
            PR_objects = pr_ds['PR_mask'].values

            time_6hr = AR_ds['time'].values
    except FileNotFoundError:
        print(f"  ⚠️ Missing files for year {year}, skipping.")
        continue

    # Sum precipitation to 6-hourly centered
    pr_data_6h = sum_to_6h_centered(pr_data[[pr_name]], time_dim=time_name)
    # Rename the dimension to 'time' so it matches AR_ds/CY_ds etc.
    pr_data_6h = pr_data_6h.rename({time_name: 'time'})
    # Now reindex using 'time'
    pr_data_6h = pr_data_6h.reindex(time=time_6hr, method="nearest", tolerance="1H")
    pr_values_6h = pr_data_6h[pr_name].values  # in m/6h

    print(f"Summed precipitation to 6-hourly intervals for year {year}.")


    # --- Initialize accumulators ---
    if contrib_AR is None:
        shape = PR_objects.shape[1:]
        contrib_AR    = np.zeros(shape, dtype=np.float32)
        contrib_CY    = np.zeros(shape, dtype=np.float32)
        contrib_ACY   = np.zeros(shape, dtype=np.float32)
        contrib_FR    = np.zeros(shape, dtype=np.float32)
        contrib_JET   = np.zeros(shape, dtype=np.float32)
        contrib_OTHER = np.zeros(shape, dtype=np.float32)
        total_precip_from_PR = np.zeros(shape, dtype=np.float32)


    # --- ASSOCIATION CONFIGURATION ---
    # Set this to True to use the hard threshold/buffer method. 
    # Set to False to use the original Gaussian influence method.
    USE_BUFFER_METHOD = True
    
    if USE_BUFFER_METHOD:
        print("  Associating PR with features using HARD BUFFERS (Binary Overlap)...")
    else:
        print("  Associating PR with features using Gaussian-weighted grid-cells...")

    nt = len(time_6hr)

    # --- CONFIGURATION FOR GAUSSIAN FILTER ---
    # Define the Gaussian Sigma (standard deviation) in km. 
    # This determines the extent of the "influence" of the feature.
    
    # SIGMA VALUE FOR GAUSSIAN-WEIGHTED METHOD
    SIGMA_BROAD_KM = 100 
    SIGMA_FRONTS_KM = 200
    
    # BUFFER DISTANCE FOR HARD BUFFER METHOD
    BUFFER_KM = 500
    dx_km = 11.17 # Grid spacing (assumed constant, polarCORDEX ~0.11° → ~11.17 km)

    for t in tqdm(range(nt), desc=f"6-Hourly timesteps {year}", leave=False):

        # --- Step 1: Compute Gaussian Influence Maps for Features ---
        
        # Binary masks (0/1) for the core feature area
        AR_mask_t = AR_features[t] > 0
        CY_mask_t = CY_features[t] > 0
        ACY_mask_t = ACY_features[t] > 0
        FR_mask_t = FR_features[t] > 0
        JET_mask_t = JET_features[t] > 0


        # --- PR Data for this timestep ---
        # Use PR_objects mask to define the grid cells we are analysing
        PR_object_cells_t = PR_objects[t] > 0
        # Precip value only where PR object exists
        precip_data = pr_values_6h[t] * PR_object_cells_t 
        
        # Accumulate total PR from PR objects (Same for both methods)
        total_precip_from_PR += precip_data

        # ---------------------------------------------------------
        # OPTION A: HARD-BUFFER METHOD 
        # ---------------------------------------------------------
        if USE_BUFFER_METHOD:

            # 1. Compute Binary Buffers (True if within distance, False otherwise)
            AR_buff  = compute_buffer_mask(AR_mask_t, BUFFER_KM, dx_km)
            CY_buff  = compute_buffer_mask(CY_mask_t, BUFFER_KM, dx_km)
            ACY_buff = compute_buffer_mask(ACY_mask_t, BUFFER_KM, dx_km)
            JET_buff = compute_buffer_mask(JET_mask_t, BUFFER_KM, dx_km)
            FR_buff  = compute_buffer_mask(FR_mask_t, BUFFER_KM, dx_km)

            # 2. Assign Contribution 
            #    If pixel is in buffer, it contributes full precip amount.
            #    (Non-exclusivity: one pixel can contribute to AR and CY simultaneously)
            contrib_AR  += precip_data * AR_buff
            contrib_CY  += precip_data * CY_buff
            contrib_ACY += precip_data * ACY_buff
            contrib_FR  += precip_data * FR_buff
            contrib_JET += precip_data * JET_buff

            # 3. Calculate OTHER Contribution
            #    Precipitation that is inside a PR object but NOT covered by ANY feature buffer
            any_feature_mask = (AR_buff | CY_buff | ACY_buff | FR_buff | JET_buff)
            
            # The mask for 'OTHER' is: Is a PR object AND is NOT in any feature buffer
            OTHER_mask = PR_object_cells_t & (~any_feature_mask)
            
            contrib_OTHER += precip_data * OTHER_mask

        # ---------------------------------------------------------
        # OPTION B: ORIGINAL GAUSSIAN METHOD (Weighted contribution) --> NOT USED ANYMORE
        # ---------------------------------------------------------
        else:
            # 1. Calculate the smooth influence maps (values 0.0 to 1.0)
            AR_influence  = compute_gaussian_influence(AR_mask_t, DIST_BROAD_KM, dx_km)
            CY_influence  = compute_gaussian_influence(CY_mask_t, DIST_BROAD_KM, dx_km)
            ACY_influence = compute_gaussian_influence(ACY_mask_t, DIST_BROAD_KM, dx_km)
            JET_influence = compute_gaussian_influence(JET_mask_t, DIST_BROAD_KM, dx_km)
            FR_influence  = compute_gaussian_influence(FR_mask_t, DIST_FRONTS_KM, dx_km)

            # 2. Contribution = Precipitation * Influence_Weight
            contrib_AR  += precip_data * AR_influence
            contrib_CY  += precip_data * CY_influence
            contrib_ACY += precip_data * ACY_influence
            contrib_FR  += precip_data * FR_influence
            contrib_JET += precip_data * JET_influence

            # 3. Calculate OTHER Contribution
            total_feature_influence = AR_influence + CY_influence + ACY_influence + FR_influence + JET_influence
            OTHER_mask = PR_object_cells_t & (total_feature_influence < 1e-3)
            contrib_OTHER += precip_data * OTHER_mask
    
    print(f"  ✅ Year {year} processing complete.")

    # Cleanup variables to free memory
    del AR_features, CY_features, ACY_features, FR_features, JET_features, PR_objects, pr_values_6h, pr_data_6h
    del AR_mask_t, CY_mask_t, ACY_mask_t, FR_mask_t, JET_mask_t, PR_object_cells_t, precip_data
    
    if USE_BUFFER_METHOD:
        del AR_buff, CY_buff, ACY_buff, FR_buff, JET_buff, any_feature_mask
    else:
        del AR_influence, CY_influence, ACY_influence, FR_influence, JET_influence, total_feature_influence
        
    gc.collect()


# -------------------
# SAVE AFTER ALL YEARS
# -------------------
print("\n Saving final multi-year contribution maps...")

ds_out = xr.Dataset(
    {
        "AR_contrib":    (("lat","lon"), contrib_AR),
        "CY_contrib":    (("lat","lon"), contrib_CY),
        "ACY_contrib":   (("lat","lon"), contrib_ACY),
        "FR_contrib":    (("lat","lon"), contrib_FR),
        "JET_contrib":   (("lat","lon"), contrib_JET),
        "OTHER_contrib": (("lat","lon"), contrib_OTHER),
        "PR_total_precip": (("lat","lon"), total_precip_from_PR),
    },
    coords={"lat": (("lat","lon"), lat2D), "lon": (("lat","lon"), lon2D)}
)

ds_out.to_netcdf(f"{OutputFolder}/PR_feature_precip_contribution_2001_2020_{data_type}.nc")
print("Saved final contribution maps.")

# Print statistics - FOR DEBUGGING
tot = np.sum(total_precip_from_PR)
if tot > 0:
    print(f"  📊 Overall statistics for {data_type}:")
    print(f"    Total PR objects: {tot:.2f} m")
    print(f"AR = {np.sum(contrib_AR)/tot*100:.1f}%")
    print(f"CY = {np.sum(contrib_CY)/tot*100:.1f}%")
    print(f"ACY = {np.sum(contrib_ACY)/tot*100:.1f}%")
    print(f"FR = {np.sum(contrib_FR)/tot*100:.1f}%")
    print(f"JET 300 = {np.sum(contrib_JET)/tot*100:.1f}%")
    print(f"OTHER = {np.sum(contrib_OTHER)/tot*100:.1f}%")

else:
    print(f"  Year {year}: No PR objects found")




##################################################################################################
# PLOTTING OF BUFFER-BASED FRACTIONS 
##################################################################################################
print("\n🎨 Creating fraction plots for buffered PR-object associations...")

# --- Load domain info ---
domain_file = xr.open_dataset('PolarRES_WP3_Antarctic_domain_expanded.nc')
inner_domain_file = xr.open_dataset('PolarRES_WP3_Antarctic_domain.nc')

rlat = domain_file['rlat'].values
rlon = domain_file['rlon'].values
rotated_pole = domain_file['rotated_pole'].attrs
rlat_inner = inner_domain_file['rlat'].values
rlon_inner = inner_domain_file['rlon'].values

inner_domain_file.close()
domain_file.close()

rlon2D, rlat2D = np.meshgrid(rlon, rlat)

verts_outer = build_rectangle_vertices(rlon, rlat)
verts_inner = build_rectangle_vertices(rlon_inner, rlat_inner)
rotated_crs = ccrs.RotatedPole(
    pole_longitude=rotated_pole['grid_north_pole_longitude'],
    pole_latitude=rotated_pole['grid_north_pole_latitude']
)

# --- Load all yearly .npz files ---
df = xr.open_dataset(f"{OutputFolder}/PR_feature_precip_contribution_2001_2020_{data_type}.nc")

AR_tot = CY_tot = ACY_tot = FR_tot = JET_tot = OTHER_tot = IVT_tot = 0


AR_tot = np.nansum(df['AR_contrib'].values)
CY_tot = np.nansum(df['CY_contrib'].values)
ACY_tot = np.nansum(df['ACY_contrib'].values)
FR_tot = np.nansum(df['FR_contrib'].values)
JET_tot = np.nansum(df['JET_contrib'].values)
OTHER_tot = np.nansum(df['OTHER_contrib'].values)
tot = np.nansum(df['PR_total_precip'].values)

print(f"Loaded.")
print(f"Total PR_Objects across loaded years: {tot}")



##################################################################################################
# PLOTTING — FRACTION MAPS FOR EACH FEATURE
##################################################################################################
print("\n🎨 Creating fraction maps...")

df = xr.open_dataset(f"{OutputFolder}/PR_feature_precip_contribution_2001_2020_{data_type}.nc")

PR_tot = df["PR_total_precip"].values

# Safe fractions (%)
AR_frac = np.divide(df["AR_contrib"].values, PR_tot, out=np.zeros_like(PR_tot), where=PR_tot!=0) * 100
CY_frac = np.divide(df["CY_contrib"].values, PR_tot, out=np.zeros_like(PR_tot), where=PR_tot!=0) * 100
ACY_frac = np.divide(df["ACY_contrib"].values, PR_tot, out=np.zeros_like(PR_tot), where=PR_tot!=0) * 100
FR_frac = np.divide(df["FR_contrib"].values, PR_tot, out=np.zeros_like(PR_tot), where=PR_tot!=0) * 100
JET_frac = np.divide(df["JET_contrib"].values, PR_tot, out=np.zeros_like(PR_tot), where=PR_tot!=0) * 100
OTHER_frac = np.divide(df["OTHER_contrib"].values, PR_tot, out=np.zeros_like(PR_tot), where=PR_tot!=0) * 100

feature_names = ["ARs", "CYs", "ACYs", "FRs", "JETs", "OTHER"]
feature_titles = ["(a) ARs", "(b) CYs", "(c) ACYs", "(d) FRs", "(e) JETs", "(f) OTHER"] 
fraction_fields = [AR_frac, CY_frac, ACY_frac, FR_frac, JET_frac, OTHER_frac]

# Load domain for plotting
domain_file = xr.open_dataset('/cluster/work/users/mmarco02/moaap_project/PolarRES_WP3_Antarctic_domain_expanded.nc')
rlat = domain_file['rlat'].values
rlon = domain_file['rlon'].values
rotated_pole = domain_file['rotated_pole'].attrs
domain_file.close()


fig, axes = plt.subplots(
    2, 3, 
    figsize=(30, 16), 
    subplot_kw={'projection': ccrs.Orthographic(0, -90)}
)

axes = axes.flatten()

# Colormap (0 to 50%) as used in Prein et al. (2023)
colors = [
    "#ffffff",  # 0-10% white
    "#b3e5fc",  # 10-20% light blue
    "#03a9f4",  # 20-30% blue
    "#4caf50",  # 30-40% green
    "#ffeb3b",  # 40-50% yellow
    "#ff9800",  # 50-60% orange
    "#f44336",  # 60-70% red
    "#b71c1c",  # 70-80% dark red
    "#9c27b0",  # 80-90% purple
    "#ff00ff",  # 90-100% fucsia/magenta
]

cmap = ListedColormap(colors)
bounds = np.arange(0, 110, 10)  # 0,10,20,...,100
norm = BoundaryNorm(bounds, ncolors=len(colors))

for ax, name, field in zip(axes, feature_names, fraction_fields):
    ax.plot(verts_outer[:,0], verts_outer[:,1], color="#EABB0F", linewidth=2, transform=rotated_crs)
    ax.plot(verts_inner[:,0], verts_inner[:,1], color="k", linewidth=2, transform=rotated_crs)
    ax.set_extent([-180, 180, -90, -20], crs=ccrs.PlateCarree())
    ax.coastlines()
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                    linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False


    mesh = ax.pcolormesh(
        rlon2D, rlat2D, field,
        cmap=cmap, norm=norm,
        transform=rotated_crs
    )

    ax.set_title(f'{feature_titles[axes.tolist().index(ax)]}', fontsize=20, fontweight='bold')

# 2. REDUCE WHITESPACE
# This minimizes the gap between plots, making the plots themselves larger
#plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0.02, right=0.98)

cbar = fig.colorbar(
    mesh,
    ax=axes,
    orientation='horizontal',
    fraction=0.05,
    pad=0.05,
    aspect=20
)

cbar.set_label('PR contribution (%)', fontsize=20)
cbar.ax.tick_params(labelsize=14)
plt.savefig(f"{OutputFolder}/PR_contribution_features_maps_2001_2020_{data_type}.png", dpi=300, bbox_inches='tight')
plt.close()



print(" Finished fraction maps.")