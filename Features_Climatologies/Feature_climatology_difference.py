import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import gc
from scipy import stats
from tqdm import tqdm

# --- 0. USER CONFIG ---
START_YEAR = 2001
END_YEAR = 2020
datasets = ["ERA5", "HCLIM", "MetUM", "RACMO2"] 
RCM_MODELS = ["HCLIM", "MetUM", "RACMO2"] 
SIGNIFICANCE_TEST = 'wilcoxon' 
epsilon = 1e-4 

feature = "AR" # choose amongst "AR", "CY", "ACY", "JET", "FR"


# --- 1. DOMAIN BOUNDARY PRE-PROCESSING ---
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

# Load representative grid and boundaries
rep_file = 'ERA5/2001/remapped/ANT_All_vars_2001_remapped.nc' # same grid for all datasets, so just need one representative file
ds_rep = xr.open_dataset(rep_file)
rlon, rlat = ds_rep['rlon'].values, ds_rep['rlat'].values
rlon2D, rlat2D = np.meshgrid(rlon, rlat)
ny, nx = len(rlat), len(rlon)
verts_outer = build_rectangle_vertices(rlon, rlat)
ds_rep.close()

# Inner domain boundaries
inner_domain_file = "PolarRES_WP3_Antarctic_domain.nc"
_inner = xr.open_dataset(inner_domain_file)
rlon_inner, rlat_inner = _inner['rlon'].values, _inner['rlat'].values
verts_inner = build_rectangle_vertices(rlon_inner, rlat_inner)
_inner.close()

rotated_crs = ccrs.RotatedPole(pole_longitude=20.0, pole_latitude=5.0)
plot_projection = ccrs.Orthographic(0, -90)

yearly_data_store = {}

# --- 2. Processing Loop (Data Loading) ---
for data_type in datasets:
    print(f"\n--- Loading {data_type} ---")
    
    # Choose folder and confoguration based on dataset and feature
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
    
    selected_files = []
    for year in range(START_YEAR, END_YEAR + 1):
        year_dir = os.path.join(OutputFolder, str(year))
        if os.path.isdir(year_dir):
            pattern = os.path.join(year_dir, "*_MOAAP-masks.nc")
            found = glob.glob(pattern)
            if found:
                selected_files.append((year, found[0]))
    selected_files.sort()

    years_list = [] 
    for year, f in tqdm(selected_files, desc=f"Reading {data_type}"):
        ds = xr.open_dataset(f, engine='netcdf4', chunks=None)
        feature_objects = ds[objects]
        
        time_vals = feature_objects['time'].values.astype('datetime64[s]')
        if len(time_vals) > 1:
            dt_seconds = np.median(np.diff(time_vals).astype('timedelta64[s]').astype('int64'))
            dt_hours = float(dt_seconds) / 3600.0
        else:
            dt_hours = 0 
        
        counts = (feature_objects > 0).sum(dim='time').values
        days_this_year = (counts * dt_hours) / 24.0
        years_list.append(days_this_year)
        
        ds.close()
        del ds
        gc.collect()

    if len(years_list) > 0:
        yearly_data_store[data_type] = np.array(years_list)
    else:
        yearly_data_store[data_type] = np.zeros((END_YEAR - START_YEAR + 1, ny, nx))

# --- 3. CALCULATION PHASE ---
data_era5 = yearly_data_store["ERA5"]
mean_era5 = np.mean(data_era5, axis=0)
era5_freq_pct = (mean_era5 / 365.25) * 100 

comparisons = {}

for model_name in RCM_MODELS:
    print(f"\nAnalyzing {model_name} vs ERA5...")
    data_rcm = yearly_data_store[model_name]
    mean_rcm = np.mean(data_rcm, axis=0)
    diff = mean_rcm - mean_era5
    
    rel_diff_grid = np.divide(diff, mean_era5, 
                              out=np.full_like(diff, np.nan), 
                              where=mean_era5 > 0) * 100 

    p_values = np.ones((ny, nx)) 
    
    if SIGNIFICANCE_TEST == 't-test':
        _, p_values = stats.ttest_rel(data_rcm, data_era5, axis=0, nan_policy='omit')
    elif SIGNIFICANCE_TEST == 'wilcoxon':
        for y in tqdm(range(ny), desc=f"Wilcoxon {model_name}"):
            for x in range(nx):
                rcm_series = data_rcm[:, y, x]
                e_series = data_era5[:, y, x]
                if not np.array_equal(rcm_series, e_series): 
                    try:
                        _, p = stats.wilcoxon(rcm_series, e_series, mode='approx')
                        p_values[y, x] = p
                    except ValueError:
                        p_values[y, x] = 1.0 
                else:
                    p_values[y, x] = 1.0

    abs_diff = np.abs(diff)
    is_noise = abs_diff < epsilon
    rel_diff_grid[is_noise] = 0.0
    p_values[is_noise] = 1.0 
    sig_mask = p_values > 0.05
    
    comparisons[model_name] = {
        "rel_diff": rel_diff_grid,
        "sig_mask": sig_mask
    }

print("\nCalculations complete. Plotting...")

# --- 4. PLOTTING SECTION ---
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(26, 7),
                         subplot_kw={'projection': plot_projection})
# Adjusted left=0.08 to accommodate the larger, rotated label
#plt.subplots_adjust(wspace=0.35, right=0.88, left=0.08, top=0.85, bottom=0.15)
plt.subplots_adjust(wspace=0.42, right=0.88, left=0.06, top=0.85, bottom=0.15)
axes = axes.flatten()

def setup_map(ax):
    ax.set_extent([-180, 180, -90, -10], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='50m', color='black', linewidth=0.7)
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                      linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = plt.FixedLocator(np.arange(-180, 181, 60))
    gl.ylocator = plt.FixedLocator(np.arange(-90, -9, 10))
    gl.xlabel_style = {'size': 9} # Slightly larger grid labels
    gl.ylabel_style = {'size': 9}
    
    ax.plot(verts_outer[:,0], verts_outer[:,1], color="#EABB0F", 
            linewidth=1.5, transform=rotated_crs, zorder=5)
    ax.plot(verts_inner[:,0], verts_inner[:,1], color="k", 
            linewidth=1.5, transform=rotated_crs, zorder=5)

# PANEL 1: ERA5
ax1 = axes[0]
setup_map(ax1)
clim_plot = ax1.pcolormesh(rlon2D, rlat2D, era5_freq_pct, shading='auto', 
                           cmap='magma_r', transform=rotated_crs, vmin=0)

# CHANGE: Larger title
ax1.set_title("ERA5", fontsize=18, weight='bold', pad=15)

# CHANGE: Rotated label, closer to plot (x=-0.1), larger font (20)
ax1.text(-0.1, 0.55, f"({letter}) {feature}s", transform=ax1.transAxes, 
         fontsize=20, fontweight='bold', va='center', ha='center', rotation=90)

# PANEL 2, 3 & 4: RCMs
v_limit_pct = 50 
diff_cmap = 'RdBu_r'
plots_diff = []

for i, model in enumerate(RCM_MODELS):
    ax = axes[i+1]
    setup_map(ax)
    data = comparisons[model]
    
    pcm = ax.pcolormesh(rlon2D, rlat2D, data['rel_diff'], shading='auto', 
                        cmap=diff_cmap, transform=rotated_crs, 
                        vmin=-v_limit_pct, vmax=v_limit_pct)
    plots_diff.append(pcm)
    
    ax.contourf(rlon2D, rlat2D, data['sig_mask'], levels=[0.5, 1.5], 
                transform=rotated_crs, hatches=['////'], colors='none')
    # CHANGE: Larger title
    if model == "RACMO2":
        ax.set_title(f"RACMO", fontsize=18, weight='bold', pad=15)
    else:
        ax.set_title(f"{model}", fontsize=18, weight='bold', pad=15)

# MANUAL COLORBARS
pos_ax1 = ax1.get_position()
pos_ax3 = axes[3].get_position()

# ERA5 Colorbar
#cbar_ax1 = fig.add_axes([pos_ax1.x1 + 0.02, pos_ax1.y0, 0.012, pos_ax1.height])
cbar_ax1 = fig.add_axes([pos_ax1.x1 + 0.02, pos_ax1.y0, 0.012, pos_ax1.height])
cb1 = plt.colorbar(clim_plot, cax=cbar_ax1, orientation='vertical', extend='max')
# CHANGE: Larger font for label and ticks
cb1.set_label(f'{feature} Frequency (%)', fontsize=14)
cb1.ax.tick_params(labelsize=12)

# Shared Diff Colorbar
cbar_ax2 = fig.add_axes([pos_ax3.x1 + 0.03, pos_ax3.y0, 0.012, pos_ax3.height])
cb2 = plt.colorbar(plots_diff[-1], cax=cbar_ax2, orientation='vertical', extend='both')
# CHANGE: Larger font for label and ticks
cb2.set_label('Relative Difference (%)', fontsize=14)
cb2.ax.tick_params(labelsize=12)

save_path = f'{feature}_Final_Plot_{START_YEAR}_{END_YEAR}.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f" All done! Plot saved as {save_path}")