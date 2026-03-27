# Features Climatology Script

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

start_time = time.time()

# --- ENVIRONMENT SETUP (not fundamental, can be removed) ---

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
####################################################################################################

# --- USER CONFIG ---
data_type = "ERA5" # ERA5 or HCLIM or MetUM or RACMO2
feature = "AR" # choose amongst "AR", "CY", "ACY", "JET", "FR"

if feature == "AR":
    OutputFolder = os.path.abspath(f'moaap_output_{data_type}_test_AR_IVT_relaxed') + '/'
    letter = 'a'
    objects = "AR_Objects"
elif feature == "CY":
    OutputFolder = os.path.abspath(f'moaap_output_{data_type}_test_CY') + '/'
    letter = 'b'
    objects = "CY_z500_Objects"
elif feature == "ACY":
    OutputFolder = os.path.abspath(f'moaap_output_{data_type}_test_CY') + '/'
    letter = 'c'
    objects = "ACY_z500_Objects"
elif feature == "FR":
    OutputFolder = os.path.abspath(f'moaap_output_{data_type}_test_FR') + '/'
    letter = 'd'
    objects = "FR_Objects"
elif feature == "JET":
    OutputFolder = os.path.abspath(f'moaap_output_{data_type}_test_JET_300') + '/'
    letter = 'e'
    objects = "JET_300_Objects"


else:
    raise ValueError(f"Unknown feature: {feature}")
    
    
    
os.makedirs(OutputFolder, exist_ok=True)

# Representative variables/grid file (used for plotting coordinates)
rep_file = 'ERA5/2001/remapped/ANT_All_vars_2001_remapped.nc' # this is the same for all datasets: they all share the expanded domain
if not os.path.exists(rep_file):
    raise FileNotFoundError(f"Representative file not found: {rep_file}")
ds_rep = xr.open_dataset(rep_file)
rlon = ds_rep['rlon'].values
rlat = ds_rep['rlat'].values

rlon2D, rlat2D = np.meshgrid(rlon, rlat)
ds_rep.close()

# Years to process
START_YEAR = 2001
END_YEAR = 2020

# Helper: build rectangle vertices (used for plotting domain boundaries)
def build_rectangle_vertices(rlon, rlat):
    rlon = np.asarray(rlon)
    rlat = np.asarray(rlat)
    if rlon.ndim == 1 and rlat.ndim == 1:
        rlon, rlat = np.meshgrid(rlon, rlat)
    top    = np.column_stack([rlon[-1, :], rlat[-1, :]])
    bottom = np.column_stack([rlon[ 0, :], rlat[ 0, :]])
    left   = np.column_stack([rlon[:, 0],  rlat[:, 0]])
    right  = np.column_stack([rlon[:, -1], rlat[:, -1]])
    return np.vstack([bottom, right, top[::-1], left[::-1]])

# Precompute polygons and rotated CRS
verts_outer = build_rectangle_vertices(rlon, rlat)
rotated_crs = ccrs.RotatedPole(pole_longitude=20.0, pole_latitude=5.0)
# inner domain vertices (open once)
inner_domain_file = "PolarRES_WP3_Antarctic_domain.nc"
if not os.path.exists(inner_domain_file):
    raise FileNotFoundError(f"Inner domain file not found: {inner_domain_file}")
_inner = xr.open_dataset(inner_domain_file)
rlon_inner = _inner['rlon'].values
rlat_inner = _inner['rlat'].values
verts_inner = build_rectangle_vertices(rlon_inner, rlat_inner)
_inner.close()

# --- Prepare accumulation arrays ---
ny, nx = len(rlat), len(rlon)
freq_accum = np.zeros((ny, nx), dtype=np.float64)          # counts of timesteps with event
hours_accum = np.zeros((ny, nx), dtype=np.float64) 
season_accum = {season: np.zeros((ny, nx), dtype=np.float64) for season in ['DJF','MAM','JJA','SON']}
days_in_dataset_accum = np.zeros((ny, nx), dtype=np.float64)
total_time_steps = 0
total_time_hours = 0.0    # accumulate hours across all files


# Find files
files = sorted(glob.glob(os.path.join(OutputFolder, "**", "*MOAAP-masks.nc"), recursive=True))
selected_files = []
for f in files:
    m = re.search(r'((19|20)\d{2})', f)
    if m:
        year = int(m.group(1))
        if START_YEAR <= year <= END_YEAR:
            selected_files.append((year,f))
selected_files.sort(key=lambda x: x[0])

print(f"Processing {len(selected_files)} years: {[y for y,_ in selected_files]}")
if len(selected_files) == 0:
    raise RuntimeError("No MOAAP files found in selected range.")


# --- Main loop ---
for year, f in tqdm(selected_files, desc="Years"):
    print(f"Year {year}: {os.path.basename(f)}")
    ds = xr.open_dataset(f)
    if objects not in ds:
        ds.close()
        raise KeyError(f"{objects} variable not found in {f}")

    feature_objects = ds[objects]    # (time,y,x) integer IDs

    time_vals = feature_objects['time'].values.astype('datetime64[s]')
    if len(time_vals) < 2:
        ds.close()
        raise ValueError(f"Not enough time steps in {f}")
    dt_seconds = np.median(np.diff(time_vals).astype('timedelta64[s]').astype('int64'))
    dt_hours = float(dt_seconds) / 3600.0
    n_steps_year = int(feature_objects.sizes['time'])

    # --- Annual frequency (counts & days) ---
    ar_bool = (feature_objects > 0).astype(np.float32)         # boolean array
    counts = ar_bool.sum(dim='time').values               # number of timesteps with event
    freq_accum += counts
    hours_accum += counts * dt_hours    # ADD THIS
    days_in_dataset_accum += counts * dt_hours / 24.0

    # update totals
    total_time_steps += n_steps_year
    total_time_hours += n_steps_year * dt_hours

    # --- Seasonal frequency: accumulate event-hours->days per season ---
    season_group = feature_objects.groupby('time.season')
    for s, da in season_group:
        season_accum[s] += (da > 0).sum(dim='time').values * dt_hours / 24.0

    # --- Lifetimes ---
    #obj_data = feature_objects.values  # integer IDs (time,y,x)
    #ids = np.unique(obj_data)
    #ids = ids[ids > 0]
    #for oid in ids:
    #    present_by_t = np.any(obj_data == oid, axis=(1,2))
    #    n_steps_obj = np.sum(present_by_t)
    #    all_lifetimes.append(n_steps_obj * dt_hours)

    # --- Peak IVT per object (if IVT present) ---
    #if ivt is not None:
    #    ivt_data = ivt.values
    #    for oid in ids:
    #        mask = (obj_data == oid)
    #        if not mask.any():
    #            continue
    #        peak_ivt = np.nanmax(ivt_data[mask])
    #        if np.isfinite(peak_ivt):
    #            all_peak_ivt.append(peak_ivt)

    # cleanup
    ds.close()
    del ds, feature_objects, ar_bool
    gc.collect()

# --- Final normalizations ---
if total_time_hours <= 0:
    raise RuntimeError("No dataset hours found (total_time_hours <= 0)")

period_days = total_time_hours / 24.0
# days per year (normalize dataset-days to 365-day year)
days_per_year = days_in_dataset_accum * (365.0 / period_days)
# percent of time with event (time-weighted): hours_accum / total_time_hours * 100
freq_percent_time = hours_accum / total_time_hours * 100.0
freq_percent_steps = freq_accum / total_time_steps * 100.0

# Save annual frequency / days (named coords)
xr.DataArray(freq_percent_time.astype(np.float32),
             coords={'y': rlat, 'x': rlon},
             dims=['y','x']).to_netcdf(os.path.join(OutputFolder,f'{feature}_frequency_percent_time.nc'))
xr.DataArray(freq_percent_steps.astype(np.float32),
             coords={'y': rlat, 'x': rlon},
             dims=['y','x']).to_netcdf(os.path.join(OutputFolder,f'{feature}_frequency_percent_steps.nc'))
xr.DataArray(days_per_year.astype(np.float32),
             coords={'y': rlat, 'x': rlon},
             dims=['y','x']).to_netcdf(os.path.join(OutputFolder,f'{feature}_frequency_days_per_year.nc'))


# --- Annual occurrence plot ---
fig, ax = plt.subplots(figsize=(9,7), subplot_kw={'projection': ccrs.Orthographic(0,-90)})
ax.set_extent([-180,180,-90,-10], crs=ccrs.PlateCarree())
ax.coastlines()

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlocator = plt.FixedLocator(np.arange(-180, 181, 60))  # longitude every 60°
gl.ylocator = plt.FixedLocator(np.arange(-90, -9, 15))    # latitude every 10°
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}

ax.plot(verts_outer[:,0], verts_outer[:,1], color="#EABB0F", linewidth=2, transform=rotated_crs)
ax.plot(verts_inner[:,0], verts_inner[:,1], color="k", linewidth=2, transform=rotated_crs)

vmin, vmax = 0.0, float(np.nanmax(days_per_year))
pcm = ax.pcolormesh(rlon2D, rlat2D, days_per_year, shading='auto', cmap='magma_r', transform=rotated_crs, vmin=vmin, vmax=vmax)
plt.colorbar(pcm, ax=ax, shrink=0.6, label=f'{feature} occurrence (days/year)')

# Second colorbar for percent of time
# create a normalized ScalarMappable with same cmap and vmin/vmax in percentage
norm = mpl.colors.Normalize(vmin=vmin/365*100, vmax=vmax/365*100)
sm = mpl.cm.ScalarMappable(cmap='magma_r', norm=norm)
sm.set_array([])  # needed for colorbar
cbar2 = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.12, label=f'{feature} occurrence (%)')
# cbar2.set_ticks(np.arange(0, 101, 10))
cbar2.ax.tick_params(labelsize=8)
#################################################################################

ax.set_title(f'{feature} Annual Occurrence {START_YEAR}-{END_YEAR} {data_type}')
fig.savefig(os.path.join(OutputFolder,f'{feature}_Annual_DaysPerYear_{START_YEAR}_{END_YEAR}_{data_type}.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)


# --- Seasonal frequency & plots (Dynamic Colorbar) ---
print("Generating seasonal plots (normalized by season length)...")

# 1. Define season lengths
season_lengths = {'DJF': 90.25, 'MAM': 92.0, 'JJA': 92.0, 'SON': 91.0}

# 2. Pre-calculate seasonal data AND find the global maximum
season_pct_map = {}
all_values = [] # Collect all pixel values to find the max

for s in ['DJF', 'MAM', 'JJA', 'SON']:
    # Calculate average days per season
    num_years = (END_YEAR - START_YEAR) + 1
    avg_days_per_season = season_accum[s] / num_years
    
    # Calculate percentage
    pct_of_season = (avg_days_per_season / season_lengths[s]) * 100.0
    season_pct_map[s] = pct_of_season
    
    all_values.append(pct_of_season)
    
    # Save NetCDF
    xr.DataArray(pct_of_season.astype(np.float32), coords={'y': rlat, 'x': rlon}, dims=['y','x']) \
      .to_netcdf(os.path.join(OutputFolder, f'{feature}_SeasonalPercent_{s}_{START_YEAR}_{END_YEAR}.nc'))

# --- DETERMINE DYNAMIC VMAX ---
global_max = np.nanmax([np.nanmax(v) for v in all_values])

# Round up to a "nice" number for the colorbar
if global_max < 1.0:
    vmax_rounded = np.ceil(global_max * 10) / 10 
elif global_max < 10.0:
    vmax_rounded = np.ceil(global_max)          
else:
    vmax_rounded = np.ceil(global_max / 5) * 5  

print(f"Global max frequency found: {global_max:.2f}%. Setting vmax to {vmax_rounded}%.")

# 3. Plotting Pairs
plot_pairs = [('DJF', 'JJA'), ('MAM', 'SON')]

for pair in plot_pairs:
    s1, s2 = pair
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={'projection': ccrs.Orthographic(0, -90)})
    
    pcm = None 
    
    for ax, s in zip(axes, pair):
        ax.set_extent([-180, 180, -90, -10], crs=ccrs.PlateCarree())
        ax.coastlines()

        # Gridlines
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False 
        gl.xlabel_style = {'size': 9}
        gl.ylabel_style = {'size': 9}

        # Domains
        ax.plot(verts_outer[:, 0], verts_outer[:, 1], color="#EABB0F", linewidth=2, transform=rotated_crs, zorder=3)
        ax.plot(verts_inner[:, 0], verts_inner[:, 1], color="#F22424", linewidth=2, transform=rotated_crs, zorder=6)

        # Plot Data
        pcm = ax.pcolormesh(rlon2D, rlat2D, season_pct_map[s], shading='auto', 
                            cmap='magma_r', transform=rotated_crs, 
                            vmin=0, vmax=vmax_rounded) 
        
        ax.set_title(f'{s}', fontsize=16)

    # --- ADDED: Side Label ---
    axes[0].text(-0.1, 0.55, f"({letter}) {feature}s", transform=axes[0].transAxes, 
                 fontsize=20, fontweight='bold', va='center', ha='center', rotation=90)

    # Shared Colorbar
    fig.subplots_adjust(right=0.85, wspace=0.20)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7]) 
    cbar = fig.colorbar(pcm, cax=cbar_ax, label=f'{feature} frequency (%)')
    
    # Supertitle removed as requested
    # fig.suptitle(f'{feature} Seasonal Frequency {START_YEAR}-{END_YEAR} ({data_type})', fontsize=18, y=0.92)

    save_path = os.path.join(OutputFolder, f'{feature}_Seasonal_Corrected_{s1}_{s2}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


print("Done. Elapsed time: {:.1f} s".format(time.time()-start_time))

'''
# --- Lifetime histogram (excpet for FRs) ---
lifetimes_hours = np.array(all_lifetimes)
if lifetimes_hours.size>0:
    median_life = np.median(lifetimes_hours)
    p90_life = np.percentile(lifetimes_hours,90)
    max_life = np.max(lifetimes_hours)
    print(f"Median lifetime: {median_life:.2f} hours")
    print(f"90th percentile lifetime: {p90_life:.2f} hours")
    print(f"Max lifetime: {max_life:.2f} hours")
    # plot
    fig,ax = plt.subplots(figsize=(8,5))
    bin_width = 6
    bins = np.arange(0,lifetimes_hours.max()+bin_width,bin_width)
    ax.hist(lifetimes_hours, bins=bins, density=True, edgecolor='black', alpha=0.7)
    ax.set_xticks(bins)
    ax.set_xlabel("Lifetime (hours)")
    ax.set_ylabel("Probability Density")
    ax.set_title(f"AR Lifetime Distribution {START_YEAR}-{END_YEAR} {data_type}")
    ax.tick_params(axis='x', labelsize=8)  # smaller font
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')  # rotate x-axis labels
    fig.savefig(os.path.join(OutputFolder,f"AR_Lifetime_PDF_{START_YEAR}_{END_YEAR}_{data_type}.png"), dpi=150)
    plt.close(fig)



# --- Peak IVT histogram (for ARs)---
peak_ivts = np.array(all_peak_ivt)
fig, ax = plt.subplots(figsize=(8,5))
ax.hist(peak_ivts, bins=40, density=True, color='royalblue', alpha=0.7)
ax.set_xlabel('Peak IVT [kg m⁻¹ s⁻¹]')
ax.set_ylabel('Probability Density')
ax.set_title('Distribution of Peak IVT for AR Objects {}-{} {}'.format(START_YEAR, END_YEAR, data_type))
plt.tight_layout()
plt.savefig(os.path.join(OutputFolder,f'AR_peakIVT_distribution_{START_YEAR}_{END_YEAR}_{data_type}.png'), dpi=150)
plt.close()
'''

