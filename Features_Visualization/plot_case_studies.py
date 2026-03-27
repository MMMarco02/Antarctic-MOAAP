'''
# This is an old version, not used, kept for reference.
import os
import sys
import time
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import PowerNorm
from scipy.ndimage import distance_transform_edt

# --- SYSTEM SETUP ---
sys.path = [p for p in sys.path if '/cluster/software/' not in p]
if 'CONDA_PREFIX' in os.environ:
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    sys.path.insert(0, os.path.join(os.environ['CONDA_PREFIX'], 'lib', pyver, 'site-packages'))
os.environ['PYTHONNOUSERSITE'] = '1'

# ==========================================
# 1. USER CONFIGURATION & CASE SELECTION
# ==========================================
data_type = "ERA5"
percentile = 99

# Define your regions
case_studies_cfg = {
    'Dronning_Maud_Land': {
        # Expanded to see more of the Southern Ocean storm track
        'extent': [-50, 80, -90, -40], 
        'central_lon': 15,
        'year': 2015, 
        'idx_t1': 7458,
        'idx_t2': 7476 
    },
    'Antarctic_Peninsula': {
        # Wider view of the SE Pacific and Drake Passage
        'extent': [-120, -25, -85, -45],
        'central_lon': -72.5,
        'year': 2016,
        'idx_t1': 4704,
        'idx_t2': 4743
    },
    'South_Shetland_Islands': {
        # Expanded to show the pathway through the Drake Passage clearly
        'extent': [-105, -30, -82, -42], 
        'central_lon': -67.5,
        'year': 2013,
        'idx_t1': 132, 
        'idx_t2': 162
    }
}

# --- CHOOSE YOUR CASE AND TIME INDICES HERE ---
selected_case = 'South_Shetland_Islands'  # 'Dronning_Maud_Land', 'Antarctic_Peninsula', or 'South_Shetland_Islands'
idx_t1 = case_studies_cfg[selected_case]['idx_t1']
idx_t2 = case_studies_cfg[selected_case]['idx_t2']
year = case_studies_cfg[selected_case]['year']

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def expand_features_to_hourly(feature_mask_6h, time_6h, time_1h):
    feature_mask_6h = feature_mask_6h.astype(bool)
    ntime1 = len(time_1h)
    nlat, nlon = feature_mask_6h.shape[1:]
    feature_hourly = np.zeros((ntime1, nlat, nlon), dtype=bool)
    for i, t6 in enumerate(time_6h):
        idx_center = np.argmin(np.abs(time_1h - t6))
        idx_start = max(0, idx_center - 2)
        idx_end   = min(ntime1, idx_center + 4)
        feature_hourly[idx_start:idx_end] |= feature_mask_6h[i]
    return feature_hourly

# ==========================================
# 3. DATA LOADING & PROCESSING
# ==========================================
print(f"Loading data for {year}...")

# Load MOAAP Features
moaap_AR_ds  = xr.open_dataset(f'moaap_output_{data_type}_test_AR_IVT_relaxed/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc')
moaap_CY_ds  = xr.open_dataset(f'moaap_output_{data_type}_test_CY/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc')
moaap_FR_ds  = xr.open_dataset(f'moaap_output_{data_type}_test_FR/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc')
moaap_JET_ds = xr.open_dataset(f'moaap_output_{data_type}_test_JET_300/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc')
moaap_PR_ds  = xr.open_dataset(f'moaap_output_{data_type}_test_EPE_dry_thr0.1_{percentile}/{year}/{year}_ERA5_ANT_{year}_EPE_ObjectMasks__dt-1h_MOAAP-masks.nc')

# Grid and Projection Info
data_vars = xr.open_dataset(f'ERA5/{year}/remapped/ANT_All_vars_{year}_remapped.nc')
rlon = data_vars['rlon'].values
rlat = data_vars['rlat'].values
rotated_pole = data_vars['rotated_pole']
rotated_crs = ccrs.RotatedPole(pole_longitude=rotated_pole.grid_north_pole_longitude,
                               pole_latitude=rotated_pole.grid_north_pole_latitude)

# Inner Domain Grid (for plotting inner domain outline)
inner_domain_file = xr.open_dataset("PolarRES_WP3_Antarctic_domain.nc")
rlon_inner = inner_domain_file['rlon'].values
rlat_inner = inner_domain_file['rlat'].values

# Precipitation Data
precip_ds = xr.open_dataset(f'ERA5/{year}/remapped/ANT_pr_{year}_remapped.nc')
PR_hourly = precip_ds['tp'].values
pr_min, pr_max = 0, float(PR_hourly.max()) * 1000

# Times
time_6h = pd.to_datetime(moaap_AR_ds['time'].values)
time_1h = pd.to_datetime(moaap_PR_ds['time'].values)

print("Expanding 6h features to 1h...")
feature_dict = {
    'CY_z500_Objects':  expand_features_to_hourly(moaap_CY_ds['CY_z500_Objects'].values, time_6h, time_1h),
    'ACY_z500_Objects': expand_features_to_hourly(moaap_CY_ds['ACY_z500_Objects'].values, time_6h, time_1h),
    'AR_Objects':       expand_features_to_hourly(moaap_AR_ds['AR_Objects'].values, time_6h, time_1h),
    'JET_300_Objects':      expand_features_to_hourly(moaap_JET_ds['JET_300_Objects'].values, time_6h, time_1h),
    'FR_Objects':       expand_features_to_hourly(moaap_FR_ds['FR_Objects'].values, time_6h, time_1h),
    'EPE_Objects':      moaap_PR_ds['EPE_mask'].values
}

feature_colors = {
    'CY_z500_Objects': 'k',
    'ACY_z500_Objects': '#ff7f00',
    'AR_Objects': '#A21212',
    'JET_300_Objects': '#6a3d9a',
    'FR_Objects': '#20740d',
    'EPE_Objects': "white"
}

# ==========================================
# 4. PLOTTING LOGIC
# ==========================================

# --- PREPARE GLOBAL DOMAIN OUTLINES ONCE ---
top_edge = np.column_stack([rlon, np.full_like(rlon, rlat[-1])])
bottom_edge = np.column_stack([rlon, np.full_like(rlon, rlat[0])])
left_edge = np.column_stack([np.full_like(rlat, rlon[0]), rlat])
right_edge = np.column_stack([np.full_like(rlat, rlon[-1]), rlat])
verts = np.vstack([bottom_edge, right_edge, top_edge[::-1], left_edge[::-1]])

# Inner Domain (using your variables rlon_inner, rlat_inner)
top_inner = np.column_stack([rlon_inner, np.full_like(rlon_inner, rlat_inner[-1])])
bottom_inner = np.column_stack([rlon_inner, np.full_like(rlon_inner, rlat_inner[0])])
left_inner = np.column_stack([np.full_like(rlat_inner, rlon_inner[0]), rlat_inner])
right_inner = np.column_stack([np.full_like(rlat_inner, rlon_inner[-1]), rlat_inner])
verts_inner = np.vstack([bottom_inner, right_inner, top_inner[::-1], left_inner[::-1]])
# ==========================================

# ==========================================
# 5. FUNCTION: PRECIPITATION PLOTS
# ==========================================
def plot_zoomed_comparison(indices, roi_name):
    roi_config = case_studies_cfg[roi_name]
    proj = ccrs.SouthPolarStereo(central_longitude=roi_config['central_lon'])
    local_max = max(PR_hourly[indices[0]].max(), PR_hourly[indices[1]].max()) * 1000
    v_max = min(local_max, 8) 
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), subplot_kw={'projection': proj}, constrained_layout=True)

    fig.set_constrained_layout_pads(h_pad=0.2, w_pad=0.2, hspace=0.1, wspace=0.1)

    for i, t_idx in enumerate(indices):
        ax = axes[i]
        ax.set_extent(roi_config['extent'], crs=ccrs.PlateCarree())
        ax.coastlines(resolution='50m', color='#787878', linewidth=1.2, zorder=2)
        
        gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, linestyle='--', alpha=0.3)
        gl.top_labels = True; gl.bottom_labels = False; gl.right_labels = False
        gl.left_labels = True if i == 0 else False 

        pcm = ax.pcolormesh(rlon, rlat, PR_hourly[t_idx]*1000, norm=PowerNorm(gamma=0.4, vmin=0, vmax=v_max), 
                            cmap='YlGnBu', transform=rotated_crs, zorder=1)
        
        ax.plot(verts[:, 0], verts[:, 1], color="#EABB0F", linewidth=3, alpha=0.4, transform=rotated_crs, zorder=4)

        for fname, feature_arr in feature_dict.items():
            mask = feature_arr[t_idx]
            if np.any(mask):
                ax.contour(rlon, rlat, mask, transform=rotated_crs, colors=feature_colors[fname], levels=[0.5], zorder=3)

        ax.set_title(f"{str(time_1h[t_idx])[:16]}", fontsize=12, pad=10)

    cb = fig.colorbar(pcm, ax=axes, orientation='vertical', fraction=0.015, pad=0.02)
    cb.set_label('Precipitation (mm/h)')
    
    legend_handles = [mpl.lines.Line2D([], [], color=color, label=fname.replace('_Objects', '')) for fname, color in feature_colors.items()]
    fig.legend(handles=legend_handles, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.05))
    plt.savefig(f"Thesis_Precip_{roi_name}.png", bbox_inches='tight', dpi=300)
    plt.show()

# ==========================================
# 6. FUNCTION: ANOMALY PLOTS
# ==========================================
def plot_anomaly_comparison(t_idx, roi_name):
    # Load anomaly data only when needed
    ds_z_anom = xr.open_dataset(f'moaap_output_{data_type}_test_CY/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc')
    ds_slp_anom = xr.open_dataset(f'moaap_output_{data_type}_test_CY_sfc/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc')
    
    roi_config = case_studies_cfg[roi_name]
    proj = ccrs.SouthPolarStereo(central_longitude=roi_config['central_lon'])
    idx_6h = np.argmin(np.abs(time_6h - time_1h[t_idx]))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), subplot_kw={'projection': proj}, constrained_layout=True)
    
    plot_vars = [
        {'data': ds_z_anom['Z500_Anomaly'].values[idx_6h], 'title': 'Z500 Anomaly', 'levels': np.arange(-300, 301, 20), 'unit': 'm'},
        {'data': ds_slp_anom['SLP_Anomaly'].values[idx_6h], 'title': 'SLP Anomaly', 'levels': np.arange(-30, 31, 2), 'unit': 'hPa'}
    ]

    for i, p_info in enumerate(plot_vars):
        ax = axes[i]
        ax.set_extent(roi_config['extent'], crs=ccrs.PlateCarree())
        
        # Shading
        im = ax.contourf(rlon, rlat, p_info['data'], levels=p_info['levels'], 
                         cmap='RdBu_r', extend='both', transform=rotated_crs, zorder=1)
        
        # IMPROVED ZERO-CONTOUR: Using a grey dashed line to distinguish from coastlines
        ax.contour(rlon, rlat, p_info['data'], levels=[0], 
                   colors='#555555', linewidths=1.5, linestyles='--', 
                   transform=rotated_crs, zorder=2)
        
        # COASTLINES: Plotted on top, solid and dark to stand out from the dashed zero-line
        ax.coastlines(resolution='50m', color='black', linewidth=1.0, zorder=3)
        
        # Domain Outline - using a distinct color
        ax.plot(verts[:, 0], verts[:, 1], color="#EABB0F", linewidth=2, 
                linestyle='-', transform=rotated_crs, zorder=4)
        
        ax.set_title(f"{p_info['title']}", fontsize=14,)
        fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, label=f'Anomaly ({p_info["unit"]})')

    plt.savefig(f"Thesis_Anomaly_{roi_name}.png", bbox_inches='tight', dpi=300)
    plt.show()

# --- EXECUTION ---
plot_zoomed_comparison([idx_t1, idx_t2], selected_case)

if selected_case == 'South_Shetland_Islands':
    plot_anomaly_comparison(idx_t1, selected_case)

print("Plots generated successfully!")
'''

# This snippet generates the case study plots used in MM's MSc Thesis, showing precipitation and feature overlays for selected events in the Antarctic region. 
# It also includes an anomaly comparison for the South Shetland Islands case. The code is structured to be modular and easily adaptable for different cases or additional features.
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import PowerNorm


# ==========================================
# 1. USER CONFIGURATION
# ==========================================
data_type = "ERA5" # ERA5 or HCLIM or MetUM or RACMO2
percentile = 99 # which EPE Objects you want to use?
selected_case = 'South_Shetland_Islands'  # 'Dronning_Maud_Land', 'Antarctic_Peninsula', or 'South_Shetland_Islands'

case_studies_cfg = {
    'Dronning_Maud_Land': {
        'extent': [-50, 80, -90, -40], 'central_lon': 15, 'year': 2015, 
        'idx_t1': 7458, 'idx_t2': 7476,
        'marker_loc': [10.0, -70.5]
    }, # Simon, Sibin, John K. Turner, Thamban Meloth, Pranab Deb, Irina V. Gorodetskaya, and 
        # Matthew Lazzara (2024). “An extreme precipitation event over Dronning Maud Land, East 
        # Antarctica: A case study of an atmospheric river event using the Polar WRF Model”. In:
        # Science of the Total Environment 987, p. 174281. doi: 10.1016/j.scitotenv.2024.174281.

    'Antarctic_Peninsula': {
        'extent': [-140, -10, -90, -40], 'central_lon': -72.5, 'year': 2016,
        'idx_t1': 4704, 'idx_t2': 4743,
        'marker_loc': [-73.0, -73.0]
    },  # González-Herrero, Sergi, Francisco Vasallo, Joan Bech, Irina Gorodetskaya, Benito Elvira, and
        # Ana Justel (2023). “Extreme precipitation records in Antarctica”. In: International Journal of
        # Climatology 43.7, pp. 3125–3138. doi: 10.1002/joc.8020.

    'South_Shetland_Islands': {
        'extent': [-105, -30, -90, -40], 'central_lon': -67.5, 'year': 2013,
        'idx_t1': 132, 'idx_t2': 162,
        'marker_loc': [-54.0, -64.5]
    }   # Torres, Christian, Deniz Bozkurt, Michael Matějka, Kamil Láska, Sibin Simon, Ricardo Jaña,
        # and Jorge Arigony-Neto (2025). “Contrasting impacts of two mesoscale cyclones on the South
        # Shetland Islands’ glaciers, northern Antarctic Peninsula”. In: Quarterly Journal of the Royal
        # Meteorological Society 151.772, e5052. doi: 10.1002/qj.5052.
}

config = case_studies_cfg[selected_case]
idx_t1, idx_t2, year = config['idx_t1'], config['idx_t2'], config['year']

# ==========================================
# 2. DATA LOADING
# ==========================================
print(f"Loading data for {year}...")

# Grid and Projection Info
data_vars = xr.open_dataset(f'ERA5/{year}/remapped/ANT_All_vars_{year}_remapped.nc')
rlon = data_vars['rlon'].values
rlat = data_vars['rlat'].values
rotated_pole = data_vars['rotated_pole']
rotated_crs = ccrs.RotatedPole(pole_longitude=rotated_pole.grid_north_pole_longitude,
                               pole_latitude=rotated_pole.grid_north_pole_latitude)

# Load Precip Data 
precip_ds = xr.open_dataset(f'ERA5/{year}/remapped/ANT_pr_{year}_remapped.nc')
PR_hourly = precip_ds['tp'].values

# Load MOAAP Features
moaap_AR_ds  = xr.open_dataset(f'moaap_output_{data_type}_test_AR_IVT_relaxed/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc')
moaap_CY_ds  = xr.open_dataset(f'moaap_output_{data_type}_test_CY/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc')
moaap_FR_ds  = xr.open_dataset(f'moaap_output_{data_type}_test_FR/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc')
moaap_JET_ds = xr.open_dataset(f'moaap_output_{data_type}_test_JET_300/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc')

if data_type == "ERA5":
    moaap_PR_ds  = xr.open_dataset(f'moaap_output_{data_type}_test_EPE_dry_thr0.1_{percentile}/{year}/{year}_{data_type}_ANT_{year}_EPE_ObjectMasks__dt-1h_MOAAP-masks.nc')
else:
    moaap_PR_ds  = xr.open_dataset(f'moaap_output_{data_type}_test_EPE_dry_thr0.1_{percentile}/{year}/{year}_{data_type}_masks.nc')

# Times
time_6h = pd.to_datetime(moaap_AR_ds['time'].values)
time_1h = pd.to_datetime(moaap_PR_ds['time'].values)

# This function takes the 6-hourly feature masks and expands them to hourly resolution by marking a window of hours around each 6-hourly timestamp as active if the feature is present.
def expand_features_to_hourly(feature_mask_6h, time_6h, time_1h):
    feature_mask_6h = feature_mask_6h.astype(bool)
    ntime1 = len(time_1h)
    nlat, nlon = feature_mask_6h.shape[1:]
    feature_hourly = np.zeros((ntime1, nlat, nlon), dtype=bool)
    for i, t6 in enumerate(time_6h):
        idx_center = np.argmin(np.abs(time_1h - t6))
        idx_start = max(0, idx_center - 2)
        idx_end   = min(ntime1, idx_center + 4)
        feature_hourly[idx_start:idx_end] |= feature_mask_6h[i]
    return feature_hourly

feature_dict = {
    'CY_z500_Objects':  expand_features_to_hourly(moaap_CY_ds['CY_z500_Objects'].values, time_6h, time_1h),
    'ACY_z500_Objects': expand_features_to_hourly(moaap_CY_ds['ACY_z500_Objects'].values, time_6h, time_1h),
    'AR_Objects':       expand_features_to_hourly(moaap_AR_ds['AR_Objects'].values, time_6h, time_1h),
    'JET_300_Objects':      expand_features_to_hourly(moaap_JET_ds['JET_300_Objects'].values, time_6h, time_1h),
    'FR_Objects':       expand_features_to_hourly(moaap_FR_ds['FR_Objects'].values, time_6h, time_1h),
    'EPE_Objects':      moaap_PR_ds['EPE_mask'].values
}

feature_colors = {
    'CY_z500_Objects': 'k', 'ACY_z500_Objects': '#ff7f00', 'AR_Objects': '#A21212',
    'JET_300_Objects': '#6a3d9a', 'FR_Objects': '#20740d', 'EPE_Objects': "white"
}

# Outer Domain Outlines
inner_domain_file = xr.open_dataset("PolarRES_WP3_Antarctic_domain.nc")
rlon_inner, rlat_inner = inner_domain_file['rlon'].values, inner_domain_file['rlat'].values

top_edge = np.column_stack([rlon, np.full_like(rlon, rlat[-1])])
bottom_edge = np.column_stack([rlon, np.full_like(rlon, rlat[0])])
left_edge = np.column_stack([np.full_like(rlat, rlon[0]), rlat])
right_edge = np.column_stack([np.full_like(rlat, rlon[-1]), rlat])
verts = np.vstack([bottom_edge, right_edge, top_edge[::-1], left_edge[::-1]])

# --- Inner Domain ---
top_inner = np.column_stack([rlon_inner, np.full_like(rlon_inner, rlat_inner[-1])])
bottom_inner = np.column_stack([rlon_inner, np.full_like(rlon_inner, rlat_inner[0])])
left_inner = np.column_stack([np.full_like(rlat_inner, rlon_inner[0]), rlat_inner])
right_inner = np.column_stack([np.full_like(rlat_inner, rlon_inner[-1]), rlat_inner])
verts_inner = np.vstack([bottom_inner, right_inner, top_inner[::-1], left_inner[::-1]])

# ==========================================
# 3. PRECIPITATION PLOT FUNCTION
# ==========================================
def plot_zoomed_comparison(indices, roi_name):
    roi_config = case_studies_cfg[roi_name]
    proj = ccrs.SouthPolarStereo(central_longitude=roi_config['central_lon'])
    
    # Calculate max based on ALL provided indices
    v_max = 10.0 # Default max 
    n_panels = len(indices)

    # Scale width precisely: 2 panels should be 14 wide (not 16) to reduce gap
    fig_width = 8 if n_panels == 1 else 14
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, 6), 
                             subplot_kw={'projection': proj}, constrained_layout=True)
    
    if n_panels == 1: axes = [axes]

    # THESIS FIX: hspace/wspace control the gap between panels. 
    fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.1, hspace=0, wspace=0.2)

    for i, t_idx in enumerate(indices):
        ax = axes[i]
        ax.set_extent(roi_config['extent'], crs=ccrs.PlateCarree())
        ax.coastlines(resolution='50m', color='#787878', linewidth=1.2, zorder=2)
        
        gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, linestyle='--', alpha=0.3)
        gl.xlabel_style = {"size":10}; gl.ylabel_style = {"size":10}
        gl.top_labels = True; gl.bottom_labels = False; gl.right_labels = False
        gl.left_labels = True if i == 0 else False 

        pcm = ax.pcolormesh(rlon, rlat, PR_hourly[t_idx]*1000, norm=PowerNorm(gamma=0.4, vmin=0, vmax=v_max), 
                            cmap='YlGnBu', transform=rotated_crs, zorder=1)
        
        # 1. Outer boundary (Solid Yellow)
        ax.plot(verts[:, 0], verts[:, 1], color="#EABB0F", linewidth=2.5, alpha=0.6, transform=rotated_crs, zorder=4)

        # 2. Inner boundary (Dotted Black/Grey)
        ax.plot(verts_inner[:, 0], verts_inner[:, 1], color="black", linestyle=':', linewidth=1.5, transform=rotated_crs, zorder=4)

        for fname, feature_arr in feature_dict.items():
            mask = feature_arr[t_idx]
            if np.any(mask):
                ax.contour(rlon, rlat, mask, transform=rotated_crs, colors=feature_colors[fname], levels=[0.5], zorder=3)

        # --- ADD MARKER FOR LOCATION OF EXTREME ---
        ax.plot(roi_config['marker_loc'][0], roi_config['marker_loc'][1], 
                marker='*', color='red', markersize=14, markeredgecolor='black', markeredgewidth=0.8,
                transform=ccrs.PlateCarree(), zorder=6)

        ax.set_title(f"{str(time_1h[t_idx])[:16]}", fontsize=14, pad=10)

    # ADJUST COLORBAR LOCATION FOR SINGLE PANEL
    if n_panels == 1:
        cb = fig.colorbar(pcm, ax=axes, orientation='vertical', fraction=0.03, pad=0.03)
    else:
        cb = fig.colorbar(pcm, ax=axes, orientation='vertical', fraction=0.015, pad=0.02)
        
    cb.set_label('Precipitation (mm/h)', fontsize=14)
    cb.ax.tick_params(labelsize=12)
    
    legend_handles = [mpl.lines.Line2D([], [], color=color, label=fname.replace('_Objects', '')) for fname, color in feature_colors.items()]
    fig.legend(handles=legend_handles, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.05))
    
    plt.savefig(f"Thesis_Precip_{roi_name}_{year}.png", bbox_inches='tight', dpi=300)
    plt.show()

# ==========================================
# 4. ANOMALY PLOT FUNCTION (4-PANEL)
# ==========================================
# Note: For this part, you need to have run the cyclone surface detection in MOAAP for the year 2013.
def plot_anomaly_4panel(indices, roi_name):
    ds_z_anom = xr.open_dataset(f'moaap_output_{data_type}_test_CY/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc')
    ds_slp_anom = xr.open_dataset(f'moaap_output_{data_type}_test_CY_sfc/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc')
    
    roi_config = case_studies_cfg[roi_name]
    proj = ccrs.SouthPolarStereo(central_longitude=roi_config['central_lon'])
    
    # Use 14 wide to match the precip plot width
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), 
                             subplot_kw={'projection': proj}, constrained_layout=True)
    
    # Match the wspace here to 0.02
    fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05, hspace=0.05, wspace=0.02)
    
    rows_info = [
        {'data': ds_z_anom['Z500_Anomaly'].values, 'title_prefix': 'Z500 Anomaly', 'levels': np.arange(-300, 301, 20), 'unit': 'm'},
        {'data': ds_slp_anom['SLP_Anomaly'].values, 'title_prefix': 'SLP Anomaly', 'levels': np.arange(-30, 31, 2), 'unit': 'hPa'}
    ]
    
    mappables = [None, None] 

    # Panel labels
    panel_labels = [['(a)', '(b)'], ['(c)', '(d)']]
    
    for row_idx, r_info in enumerate(rows_info):
        for col_idx, t_idx in enumerate(indices):
            
            dt_current = time_1h[t_idx]
            idx_6h = np.argmin(np.abs(time_6h - dt_current))
            data_slice = r_info['data'][idx_6h]
            
            ax = axes[row_idx, col_idx]
            ax.set_extent(roi_config['extent'], crs=ccrs.PlateCarree())
            
            # Plot Data
            im = ax.contourf(rlon, rlat, data_slice, levels=r_info['levels'], 
                             cmap='RdBu_r', extend='both', transform=rotated_crs, zorder=1)
            
            if col_idx == 1: 
                mappables[row_idx] = im

            # Zero Line
            ax.contour(rlon, rlat, data_slice, levels=[0], 
                       colors='#555555', linewidths=1.0, linestyles='--', 
                       transform=rotated_crs, zorder=2)
            
            # Coastlines & Domain
            ax.coastlines(resolution='50m', color='black', linewidth=0.8, zorder=3)
            ax.plot(verts[:, 0], verts[:, 1], color="#EABB0F", linestyle='-', linewidth=1.5, transform=rotated_crs, zorder=4)
            
            # --- ADD MARKER (optional) ---
            #ax.plot(roi_config['marker_loc'][0], roi_config['marker_loc'][1], 
            #        marker='*', color='red', markersize=14, markeredgecolor='black', markeredgewidth=0.8,
            #        transform=ccrs.PlateCarree(), zorder=5)

            # Title
            timestamp_str = str(time_6h[idx_6h])[:16]
            ax.set_title(f"{panel_labels[row_idx][col_idx]} {r_info['title_prefix']} - {timestamp_str}", fontsize=16, pad=6)

        # Shared Colorbar for this Row
        cb = fig.colorbar(mappables[row_idx], ax=axes[row_idx, :], location='right', 
                          shrink=0.9, pad=0.02, aspect=30)
        cb.set_label(f'Anomaly ({r_info["unit"]})', fontsize=14)
        cb.ax.tick_params(labelsize=12)

    plt.savefig(f"Thesis_Anomaly_4Panel_{roi_name}.png", bbox_inches='tight', dpi=300)
    plt.show()

# --- EXECUTION ---

# Logic: If South Shetland, show t1 AND t2. Otherwise, show ONLY t2.
if selected_case == 'South_Shetland_Islands':
    plot_indices = [idx_t1, idx_t2]
else:
    plot_indices = [idx_t2]

# 1. Precip Plot
plot_zoomed_comparison(plot_indices, selected_case)

# 2. Anomaly Plot (Only if SSI)
if selected_case == 'South_Shetland_Islands':
    plot_anomaly_4panel(plot_indices, selected_case)

print("Done!")