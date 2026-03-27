
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

# --- Environment setup (not fundamental, can be removed) ---

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

    # Ensure inputs are boolean
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
    

# Build rectangle vertices (for plotting domain boundaries)
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


print("🧠 Starting EPE analysis with buffered EPE objects...")

# --- USER CONFIG ---
data_type = "ERA5" # ERA5 or HCLIM or MetUM or RACMO2
percentile = 999 # 999 or 99
# Outut folder
OutputFolder = os.path.abspath(f'EPE_analysis_output_{data_type}_2001_2020_epe_objects_buffered_500_{percentile}') + '/'
os.makedirs(OutputFolder, exist_ok=True)


# Features & pr_objects folders 
moaap_AR_folder = os.path.abspath(f'moaap_output_{data_type}_test_AR_IVT_relaxed')
moaap_CY_folder = os.path.abspath(f'moaap_output_{data_type}_test_CY')
moaap_FR_folder = os.path.abspath(f'moaap_output_{data_type}_test_FR')
#moaap_JET_folder = os.path.abspath(f'moaap_output_{data_type}_test_JET') # previously used when working with JET at 200hPa
moaap_JET_folder = os.path.abspath(f'moaap_output_{data_type}_test_JET_300')
moaap_PR_folder = os.path.abspath(f'moaap_output_{data_type}_test_EPE_dry_thr0.1_{percentile}') 

# Initialize accumulation arrays
AR_total_overlap = CY_total_overlap = ACY_total_overlap = FR_total_overlap = JET_total_overlap = OTHER_total_overlap = None
EPE_total_count = None

# Buffering parameters needed
data_vars = xr.open_dataset(f'ERA5/2001/remapped/ANT_All_vars_2001_remapped.nc')[['lat','lon']] # same for all
lat2D = data_vars['lat'].values
lon2D = data_vars['lon'].values
data_vars.close()
BUFFER_KM = 500  # Buffer distance in km

years = np.arange(2001, 2021)  # 2001-2020

# --- CHECK IF ANALYSIS ALREADY EXISTS ---
# We check for both association and spatial files for every year
expected_files = []
for year in years:
    expected_files.append(f"{OutputFolder}/EPE_feature_association_{year}.npz")
    expected_files.append(f"{OutputFolder}/EPE_feature_spatial_{year}.npz")

analysis_done = all(os.path.exists(f) for f in expected_files)

if analysis_done:
    print("📦 All NPZ files found. Skipping heavy computation...")
    print("📂 Loading and accumulating spatial overlap maps...")

    first = True
    for year in tqdm(years, desc="Loading data"):
        d = np.load(f"{OutputFolder}/EPE_feature_spatial_{year}.npz")
        if first:
            # Using copy() is good practice to avoid reference issues
            AR_total_overlap  = d['AR'].copy()
            CY_total_overlap  = d['CY'].copy()
            ACY_total_overlap = d['ACY'].copy()
            FR_total_overlap  = d['FR'].copy()
            JET_total_overlap = d['JET'].copy()
            OTHER_total_overlap = d['OTHER'].copy()
            EPE_total_count   = d['TOTAL'].copy()
            first = False
        else:
            AR_total_overlap  += d['AR']
            CY_total_overlap  += d['CY']
            ACY_total_overlap += d['ACY']
            FR_total_overlap  += d['FR']
            JET_total_overlap += d['JET']
            OTHER_total_overlap += d['OTHER']
            EPE_total_count   += d['TOTAL']

else:
    print("NPZ files missing or incomplete. Running full analysis...")


    for year in tqdm(years, desc="Processing years"):
        print(f"\n🧠 Processing year {year}...")

        if data_type == "ERA5":
            epe_file = f'{moaap_PR_folder}/{year}/{year}_ERA5_ANT_{year}_EPE_ObjectMasks__dt-1h_MOAAP-masks.nc'
        else :
            epe_file = f'{moaap_PR_folder}/{year}/{year}_{data_type}_mask.nc'

        try:
            with xr.open_dataset(f'{moaap_AR_folder}/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc') as AR_ds, \
                xr.open_dataset(f'{moaap_CY_folder}/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc') as CY_ds, \
                xr.open_dataset(f'{moaap_FR_folder}/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc') as FR_ds, \
                xr.open_dataset(f'{moaap_JET_folder}/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc') as JET_ds, \
                xr.open_dataset(epe_file) as pr_ds:
                
                # Extract data
                # Change these lines in your loop:
                AR_features = AR_ds['AR_Objects'].values.astype(bool)
                CY_features = CY_ds['CY_z500_Objects'].values.astype(bool)
                ACY_features = CY_ds['ACY_z500_Objects'].values.astype(bool)
                FR_features = FR_ds['FR_Objects'].values.astype(bool)
                JET_features = JET_ds['JET_300_Objects'].values.astype(bool)
                EPE_objects = pr_ds['EPE_mask'].values
                time_6hr = AR_ds['time'].values
                time_1hr = pr_ds['time'].values
        except FileNotFoundError:
            print(f" Missing files for year {year}, skipping.")
            continue


        # 1. Buffer the 6-hourly features BEFORE expanding
        print(f"  Buffering 6-hourly features ({BUFFER_KM}km)...")
        

        AR_buffered_6h = buffer_3d_mask(AR_features, BUFFER_KM)
        CY_buffered_6h = buffer_3d_mask(CY_features, BUFFER_KM)
        ACY_buffered_6h = buffer_3d_mask(ACY_features, BUFFER_KM)
        FR_buffered_6h = buffer_3d_mask(FR_features, BUFFER_KM)
        JET_buffered_6h = buffer_3d_mask(JET_features, BUFFER_KM)

        # 2. Now expand the ALREADY BUFFERED features to hourly
        print("  Expanding buffered features to hourly...")
        AR_hourly = expand_features_to_hourly(AR_buffered_6h, time_6hr, time_1hr)
        CY_hourly = expand_features_to_hourly(CY_buffered_6h, time_6hr, time_1hr)
        ACY_hourly = expand_features_to_hourly(ACY_buffered_6h, time_6hr, time_1hr)
        FR_hourly = expand_features_to_hourly(FR_buffered_6h, time_6hr, time_1hr)
        JET_hourly = expand_features_to_hourly(JET_buffered_6h, time_6hr, time_1hr)


        # Initialize accumulation arrays (will be set after first year)
        if AR_total_overlap is None:
            shape = AR_hourly.shape[1:]  # (lat, lon)
            AR_total_overlap = np.zeros(shape, dtype=np.int32)
            CY_total_overlap = np.zeros(shape, dtype=np.int32)
            ACY_total_overlap = np.zeros(shape, dtype=np.int32)
            FR_total_overlap = np.zeros(shape, dtype=np.int32)
            JET_total_overlap = np.zeros(shape, dtype=np.int32)
            OTHER_total_overlap = np.zeros(shape, dtype=np.int32)  
            EPE_total_count = np.zeros(shape, dtype=np.int32)


        # --- BUFFER-BASED ASSOCIATION ---
        print("  Associating EPE_objects with features using buffer...")

        AR_hits = CY_hits = ACY_hits = FR_hits = JET_hits = OTHER_hits = total_EPE_objects = IVT_hits = 0
        nt = len(time_1hr)

        # --- SIMPLIFIED ASSOCIATION ---
        for t in tqdm(range(nt), desc=f"Hourly timesteps {year}", leave=False):
            pr_mask_binary = EPE_objects[t] == 1
            if not np.any(pr_mask_binary): continue

            labeled, num_objects = label(pr_mask_binary, structure=np.ones((3,3)))

            for obj_id in range(1, num_objects+1):
                epe_mask = (labeled == obj_id)
                total_EPE_objects += 1
                EPE_total_count += epe_mask.astype(int)
                associated = False

                # We now check if the EPE overlaps with the BUFFERED feature
                if np.any(epe_mask & AR_hourly[t]):
                    AR_hits += 1
                    AR_total_overlap += epe_mask.astype(int)
                    associated = True
                if np.any(epe_mask & CY_hourly[t]):
                    CY_hits += 1
                    CY_total_overlap += epe_mask.astype(int)
                    associated = True
                if np.any(epe_mask & ACY_hourly[t]):
                    ACY_hits += 1
                    ACY_total_overlap += epe_mask.astype(int)
                    associated = True
                if np.any(epe_mask & FR_hourly[t]):
                    FR_hits += 1
                    FR_total_overlap += epe_mask.astype(int)
                    associated = True
                if np.any(epe_mask & JET_hourly[t]):
                    JET_hits += 1
                    JET_total_overlap += epe_mask.astype(int)
                    associated = True
                
                if not associated:
                    OTHER_hits += 1
                    OTHER_total_overlap += epe_mask.astype(int)

        # Save spatial overlap counts
        np.savez_compressed(f"{OutputFolder}/EPE_feature_spatial_{year}.npz",
                            AR=AR_total_overlap, CY=CY_total_overlap, ACY=ACY_total_overlap,
                            FR=FR_total_overlap, JET=JET_total_overlap, OTHER=OTHER_total_overlap,
                            TOTAL=EPE_total_count
    )

        # Save buffer-based association counts
        np.savez_compressed(f"{OutputFolder}/EPE_feature_association_{year}.npz",
                            AR_hits=AR_hits, CY_hits=CY_hits, ACY_hits=ACY_hits,
                            FR_hits=FR_hits, JET_hits=JET_hits, OTHER_hits=OTHER_hits,
                            total_EPE_objects=total_EPE_objects)

        # Print statistics - FIXED
        if total_EPE_objects > 0:
            print(f"DATA TYPE: {data_type}, PERCENTILE: {percentile}")
            print(f"  📊 Year {year} statistics:")
            print(f"    Total EPE objects: {total_EPE_objects}")
            print(f"    AR-associated: {AR_hits} ({AR_hits/total_EPE_objects*100:.1f}%)")
            print(f"    CY-associated: {CY_hits} ({CY_hits/total_EPE_objects*100:.1f}%)")
            print(f"    ACY-associated: {ACY_hits} ({ACY_hits/total_EPE_objects*100:.1f}%)")
            print(f"    FR-associated: {FR_hits} ({FR_hits/total_EPE_objects*100:.1f}%)")
            print(f"    JET300-associated: {JET_hits} ({JET_hits/total_EPE_objects*100:.1f}%)")
            print(f"    OTHER-associated: {OTHER_hits} ({OTHER_hits/total_EPE_objects*100:.1f}%)")
        else:
            print(f"  📊 Year {year}: No EPE objects found")

        print(f"  Year {year} summary: Total EPE: {total_EPE_objects}, AR:{AR_hits}, CY:{CY_hits}, ACY:{ACY_hits}, FR:{FR_hits}, JET300:{JET_hits}, OTHER:{OTHER_hits}")

        # Cleanup
        del AR_features, CY_features, ACY_features, FR_features, JET_features, EPE_objects
        del AR_hourly, CY_hourly, ACY_hourly, FR_hourly, JET_hourly
        #del CY_features, ACY_features, FR_features, JET_features, EPE_objects, IVT_features
        #del CY_hourly, ACY_hourly, FR_hourly, JET_hourly, IVT_hourly
        gc.collect()

    print("✅ EPE-feature association complete.")


##################################################################################################
# PLOTTING OF BUFFER-BASED FRACTIONS (FOR THE YEARS PROCESSED ABOVE)
##################################################################################################
print("\n🎨 Creating fraction plots for buffered EPE-object associations...")

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
files = sorted(glob.glob(f"{OutputFolder}/EPE_feature_association_*.npz"))

if len(files) == 0:
    raise RuntimeError("❌ No NPZ files found. Cannot create plots.")

AR_tot = CY_tot = ACY_tot = FR_tot = JET_tot = OTHER_tot = TOTAL_EPE = IVT_tot = 0

for f in files:
    d = np.load(f)
    AR_tot    += d['AR_hits']
    CY_tot    += d['CY_hits']
    ACY_tot   += d['ACY_hits']
    FR_tot    += d['FR_hits']
    JET_tot   += d['JET_hits']
    OTHER_tot += d['OTHER_hits']
    TOTAL_EPE += d['total_EPE_objects']

print(f"Loaded {len(files)} years.")
print(f"Total EPEs across loaded years: {TOTAL_EPE}")

# --- Assemble into lists ---
features = ['AR', 'CY', 'ACY', 'FR', 'JET300', 'OTHER']
#features = ['IVT', 'CY', 'ACY', 'FR', 'JET', 'OTHER']
hit_counts = [AR_tot, CY_tot, ACY_tot, FR_tot, JET_tot, OTHER_tot]
#hit_counts = [IVT_tot, CY_tot, ACY_tot, FR_tot, JET_tot, OTHER_tot]
overlap_data = [AR_total_overlap, CY_total_overlap, ACY_total_overlap, 
                FR_total_overlap, JET_total_overlap, OTHER_total_overlap]
#overlap_data = [IVT_total_overlap, CY_total_overlap, ACY_total_overlap,
#               FR_total_overlap, JET_total_overlap, OTHER_total_overlap]

# --- Compute fractions ---
fractions = [(h / TOTAL_EPE * 100) if TOTAL_EPE > 0 else 0 for h in hit_counts]

##################################################################################################
# MAKE BAR PLOT OF FRACTIONS
##################################################################################################

# Plot 1: Fractions
fig, axs = plt.subplots(2, 3, figsize=(24, 12), subplot_kw={'projection': ccrs.Orthographic(0, -90)})
axs = axs.flatten()

fraction_data = []
for overlap in overlap_data[:-1]:  # Exclude ANY for fractions
    fraction = np.divide(overlap, EPE_total_count, 
                        out=np.zeros_like(overlap, dtype=float), 
                        where=EPE_total_count!=0) * 100
    fraction_data.append(fraction)

# Add ANY fraction separately
other_fraction = np.divide(OTHER_total_overlap, EPE_total_count,
                        out=np.zeros_like(OTHER_total_overlap, dtype=float),
                        where=EPE_total_count!=0) * 100
fraction_data.append(other_fraction)

# cmap = 'magma_r'

# UPDATED PART: Define discrete colormap and normalization
    # Define discrete bins
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



for ax, fraction, feature in zip(axs, fraction_data, features):
    ax.plot(verts_outer[:,0], verts_outer[:,1], color="#EABB0F", linewidth=2, transform=rotated_crs)
    ax.plot(verts_inner[:,0], verts_inner[:,1], color="k", linewidth=2, transform=rotated_crs)
    ax.set_extent([-180, 180, -90, -10], crs=ccrs.PlateCarree())
    ax.coastlines()
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    #mesh = ax.pcolormesh(rlon2D, rlat2D, fraction, shading="auto", cmap=cmap, vmin=0, vmax=100, transform=rotated_crs) # PREVIOUS

    mesh = ax.pcolormesh(rlon2D, rlat2D, fraction, shading="auto",
                        cmap=cmap, norm=norm, transform=rotated_crs) 

    # ALSO NEW PART: Add contour lines for better visualization
    #levels = [25, 50, 75]  # % thresholds
    #cs = ax.contour(rlon2D, rlat2D, fraction, levels=levels, colors='white', linewidths=1.5, transform=rotated_crs)
    #ax.clabel(cs, fmt='%d%%', fontsize=10)
    ################
    ax.set_title(f'{data_type} Fraction of EPEs associated with {feature} (2001-2020)')

fig.colorbar(mesh, ax=axs, orientation='horizontal', fraction=0.05, pad=0.05, label='Fraction of EPEs (%)')
plt.savefig(f'{OutputFolder}/EPE_feature_overlap_fractions_2001_2020_{data_type}.png', dpi=300, bbox_inches='tight')
plt.close()

# Print final statistics
total_epe_all_years = np.sum(EPE_total_count)
print(f"\n📊 FINAL STATISTICS (2001-2020):")
print(f"Total EPE occurrences: {total_epe_all_years}")
for feature, overlap in zip(features, overlap_data):
    feature_total = np.sum(overlap)
    fraction = feature_total / total_epe_all_years * 100
    print(f"{feature}: {feature_total} EPEs ({fraction:.1f}%)")

end_time = time.time()
print(f"\n⏱️ Total elapsed time: {(end_time - start_time)/60:.2f} minutes")
print("EPE analysis completed successfully!")
