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
from matplotlib.colors import LogNorm, PowerNorm
from PIL import Image


start_time = time.time()

print("All imports successful")


####################################################################################################
# HELPER FUNCTIONS NEEDED FOR THE ANALYSIS
####################################################################################################

# Function to expand 6-hourly feature masks to hourly timesteps, assuming their persistence over 6-hour windows. This is a simple approach and can be refined if needed.
def expand_features_to_hourly(feature_mask_6h, time_6h, time_1h):


    feature_mask_6h = feature_mask_6h.astype(bool)

    ntime1 = len(time_1h)
    nlat, nlon = feature_mask_6h.shape[1:]
    
    feature_hourly = np.zeros((ntime1, nlat, nlon), dtype=bool)
    
    for i, t6 in enumerate(time_6h):
        # Find closest hourly index
        idx_center = np.argmin(np.abs(time_1h - t6))
        # Expand ±2 hours
        idx_start = max(0, idx_center - 2)
        idx_end   = min(ntime1, idx_center + 4)  # +3 because slicing excludes end
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
    

#  Build rectangle vertices (used for plotting domain outlines)
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


def make_gif(frame_folder, data_type):
    frames = [Image.open(image) for image in np.sort(glob.glob(f"{frame_folder}/*.jpg"))]
    frame_one = frames[0]
    frame_one.save(f"{data_type}_features_with_{percentile}_epes_{plot_type}_{plot_start}_{plot_end}_{year}.gif", format="GIF", append_images=frames,
               save_all=True, duration=1000, loop=0)
####################################################################################################


# --- USER CONFIG ---

year = 2004 # choose sample year to analyze
data_type = "RACMO2"  # "ERA5" or "HCLIM" or "MetUM" or "RACMO2"
percentile = 999 # 99 or 999

# Read features' files
moaap_AR_ds = xr.open_dataset(f'moaap_output_{data_type}_test_AR_IVT_relaxed/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc')
moaap_CY_ds = xr.open_dataset(f'moaap_output_{data_type}_test_CY/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc')
moaap_FR_ds = xr.open_dataset(f'moaap_output_{data_type}_test_FR/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc')
#moaap_JET_ds = xr.open_dataset(f'moaap_output_{data_type}_test_JET/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc')
moaap_JET_ds = xr.open_dataset(f'moaap_output_{data_type}_test_JET_300/{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc') # now with jets at 300 hPa instead of 200 hPa


if data_type == "ERA5":
    moaap_PR_ds = xr.open_dataset(f'moaap_output_ERA5_test_EPE_dry_thr0.1_{percentile}/{year}/{year}_ERA5_ANT_{year}_EPE_ObjectMasks__dt-1h_MOAAP-masks.nc')
else:
    moaap_PR_ds = xr.open_dataset(f'moaap_output_{data_type}_test_EPE_dry_thr0.1_{percentile}/{year}/{year}_{data_type}_mask.nc')

# Read features' variables
AR_6h = moaap_AR_ds['AR_Objects'].values # (time6h, lat, lon)
CY_6h = moaap_CY_ds['CY_z500_Objects'].values
ACY_6h = moaap_CY_ds['ACY_z500_Objects'].values
FR_6h = moaap_FR_ds['FR_Objects'].values
#JET_6h  = moaap_JET_ds['JET_Objects'].values
JET_6h  = moaap_JET_ds['JET_300_Objects'].values
EPE_1h = moaap_PR_ds['EPE_mask'].values # (time1h, lat, lon)

# Read time variables
time_6h = pd.to_datetime(moaap_AR_ds['time'].values)
time_1h = pd.to_datetime(moaap_PR_ds['time'].values)

# Expand 6-hourly features to hourly
AR_1h  = expand_features_to_hourly(AR_6h,  time_6h, time_1h)
CY_1h  = expand_features_to_hourly(CY_6h,  time_6h, time_1h)
ACY_1h = expand_features_to_hourly(ACY_6h, time_6h, time_1h)
FR_1h  = expand_features_to_hourly(FR_6h,  time_6h, time_1h)
JET_1h = expand_features_to_hourly(JET_6h, time_6h, time_1h)



############################################################################################################################
# --- PLOTTING PARAMETERS ---
plot_type = "contours"  # "contours" or "masks"
plot_start = 2950  # first 6-hourly timestep to plot
plot_end   = 3020 # last 6-hourly timestep to plot

# If you want, add a marker to this location (e.g. a specific research station, glacier, or island)
#marker_lat = -63.98015390418824
#marker_lon = -59.44246207935862
############################################################################################################################

if plot_type == "contours": # originally I planned to do both contours and masks, but masks were abandoned at a later stage.

    print("Plotting contours frames...")
    os.makedirs(f'{data_type}_frames_{year}', exist_ok=True)

    # --- Setup (use whatever) ---
    data_vars = xr.open_dataset(f'ERA5/{year}/remapped/ANT_All_vars_{year}_remapped.nc')
    rlon = data_vars['rlon']  
    rlat = data_vars['rlat'] 
    rotated_pole = data_vars['rotated_pole']
    rotated_crs = ccrs.RotatedPole(pole_longitude=rotated_pole.grid_north_pole_longitude,
                                   pole_latitude=rotated_pole.grid_north_pole_latitude)


    # Hourly precipitation data
    if data_type in ["HCLIM", "MetUM"]: 
        total_precip = xr.open_dataset(f'PolarRES/{data_type}/nested/{year}/ANT_{data_type}_pr_{year}_nested.nc')  # total precipitation variable
        PR_hourly = total_precip['pr']  # [time, y, x]

        # For consistent colorbar
        pr_min = float(PR_hourly.min()) * 1000
        pr_max = float(PR_hourly.max()) * 1000
    
    elif data_type == "RACMO2":
        total_precip = xr.open_dataset(f'PolarRES/{data_type}/nested/{year}/ANT_{data_type}_pr_{year}_nested_con.nc')  # total precipitation variable
        PR_hourly = total_precip['pr']  # [time, y, x]

        # For consistent colorbar
        pr_min = float(PR_hourly.min()) * 1000
        pr_max = float(PR_hourly.max()) * 1000

    elif data_type == "ERA5":
        total_precip = xr.open_dataset(f'ERA5/{year}/remapped/ANT_pr_{year}_remapped.nc')  # total precipitation variable
        PR_hourly = total_precip['tp']  # [time, y, x]


        # For consistent colorbar
        pr_min = float(PR_hourly.min()) * 1000
        pr_max = float(PR_hourly.max()) * 1000

    # Sea ice data
    #sea_ice_ds = xr.open_dataset(f'ERA5/{year}/remapped/ANT_sea_ice_cover_{year}_remapped.nc')
    #sic_all = sea_ice_ds['siconc'].values

    # Inner domain for highlight
    inner_domain_file = xr.open_dataset("PolarRES_WP3_Antarctic_domain.nc")
    rlon_inner = inner_domain_file['rlon']
    rlat_inner = inner_domain_file['rlat']



    # Features dictionary
    feature_dict = {
    'CY_z500_Objects':  CY_1h,
    'ACY_z500_Objects': ACY_1h,
    'AR_Objects':       AR_1h,
    'JET_300_Objects':      JET_1h,
    'FR_Objects':       FR_1h,
    'EPE_Objects':      EPE_1h
    }
    
    feature_colors = {
        'CY_z500_Objects': 'k',
        'ACY_z500_Objects': '#ff7f00',
        'AR_Objects': '#A21212',
        'JET_300_Objects': '#6a3d9a',
        'FR_Objects': '#20740d',
        'EPE_Objects': "white"
    }


    # --- Loop over hourly timesteps ---

    for t_hr in range(plot_start, plot_end + 1):
        dt_hr = time_1h[t_hr]
        pr = PR_hourly[t_hr, :, :]     

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.Orthographic(0, -90)})
        ax.set_extent([-180, 180, -90, -10], crs=ccrs.PlateCarree())
        ax.coastlines(color='#969696')

        # Gridlines
        gl = ax.gridlines(draw_labels=True, xlocs=[-180, -120, -60, 0, 60, 120, 180],
                        ylocs=[-90, -75, -60, -45, -30], linewidth=0.5, color='gray', alpha=0.7)
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}

        # Domain outline
        top_edge = np.column_stack([rlon, np.full_like(rlon, rlat[-1])])
        bottom_edge = np.column_stack([rlon, np.full_like(rlon, rlat[0])])
        left_edge = np.column_stack([np.full_like(rlat, rlon[0]), rlat])
        right_edge = np.column_stack([np.full_like(rlat, rlon[-1]), rlat])
        verts = np.vstack([bottom_edge, right_edge, top_edge[::-1], left_edge[::-1]])
        ax.fill(verts[:,0], verts[:,1], facecolor="#DFEA0F", alpha=0.0, edgecolor="#EABB0F", linewidth=6, transform=rotated_crs)

        # Inner domain outline
        top_inner = np.column_stack([rlon_inner, np.full_like(rlon_inner, rlat_inner[-1])])
        bottom_inner = np.column_stack([rlon_inner, np.full_like(rlon_inner, rlat_inner[0])])
        left_inner = np.column_stack([np.full_like(rlat_inner, rlon_inner[0]), rlat_inner])
        right_inner = np.column_stack([np.full_like(rlat_inner, rlon_inner[-1]), rlat_inner])
        verts_inner = np.vstack([bottom_inner, right_inner, top_inner[::-1], left_inner[::-1]])
        ax.plot(verts_inner[:, 0], verts_inner[:, 1], color="red", linewidth=2, transform=rotated_crs, zorder=6)

        # --- Plot hourly precipitation as base layer ---
        pcm = ax.pcolormesh(
            rlon, rlat, pr*1000,
            norm=PowerNorm(gamma=0.4),
            cmap='YlGnBu',
            transform=rotated_crs,
            zorder=0
        )

        cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Precipitation (mm/h)') # to modify when HCLIM data available


        # --- Plot sea ice cover from ERA5 as contours (optional) ---
        #t6_idx = np.argmin(np.abs(time_6h - time_1h[t_hr]))
        #sic = sic_all[t6_idx, :, :]  # fractional 0-1
        #ax.contour(rlon, rlat, sic, levels=[0.005], colors='yellow', linewidths=2.0, transform=rotated_crs, zorder=10)
        #sic = sic_all[t_hr,:,:]  # fractional 0-1
        #ax.contour(rlon, rlat, sic, levels=[0.05], colors='royalblue', linewidths=3.0, transform=rotated_crs, zorder=10)
        # ax.contour(rlon, rlat, sic, levels=[0.15], colors='white', linewidths=1.5, transform=rotated_crs, zorder=11)


        # --- Overlay repeated 6-hourly features ---
        for fname, feature_arr in feature_dict.items():
            mask = feature_arr[t_hr]    # already boolean
            if fname == 'EPE_Objects':
                lw = 3.0  # thicker
                ls = 'solid'
            else:
                lw = 2.0
                ls = 'solid'
            ax.contour(
                rlon, rlat, mask,
                transform=rotated_crs,
                colors=feature_colors[fname],
                levels=[0.5],
                linewidths=lw,
                linestyles=ls,
            )


        # --- Add marker for specific location (optional) ---
        #ax.plot(marker_lon, marker_lat, marker='X', color='magenta', markersize=10, transform=ccrs.PlateCarree(), zorder=20, label='Marker Location')


        # --- Legend ---
        legend_handles = []
        for fname, color in feature_colors.items():
            legend_handles.append(
                mpl.lines.Line2D([], [], color=color, label=fname.replace('_Objects',''))
            )
        ax.legend(handles=legend_handles, loc='lower left', fontsize=8)

        ax.set_title(f'{data_type}-MOAAP Features Contours at {str(dt_hr)[:16]}', fontsize=14)

        fig.savefig(f'{data_type}_frames_{year}/{str(t_hr).zfill(3)}_all_features_{plot_type}.jpg', bbox_inches='tight', dpi=100)
        plt.close(fig)


# if __name__ == "__main__":
make_gif(f'{data_type}_frames_{year}/', data_type)
print("GIF created successfully!")

# --- Optional: delete frames after GIF creation ---
frame_files = glob.glob(f'{data_type}_frames_{year}/*.jpg')
for f in frame_files:
    os.remove(f)

print(f"Deleted {len(frame_files)} frame files after GIF creation.")
