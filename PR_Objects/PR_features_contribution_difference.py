import os
import gc
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy import stats
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
import time

print("✅ Imported all necessary modules.")


# =========================================================================================
# --- HELPER FUNCTIONS ---
# =========================================================================================

# This function sums hourly precipitation into 6-hour bins centered at 00, 06, 12, and 18 UTC.
# Specifically:
    #00 = sum(22,23,00,01,02,03)
    #06 = sum(04,05,06,07,08,09)
    #12 = sum(10,11,12,13,14,15)
    #18 = sum(16,17,18,19,20,21)

def sum_to_6h_centered(pr_hourly, time_dim='time'):

    pr = pr_hourly.sortby(time_dim)
    
    # rolling sum, right-edge aligned
    pr_rolled = pr.rolling({time_dim: 6}, center=False).sum()
    
    # select times ending at hour %6==3 (so after shift, centers at 0,6,12,18)
    pr_selection = pr_rolled.isel({time_dim: pr_rolled[time_dim].dt.hour % 6 == 3})
    
    # shift labels back to center
    pr_selection = pr_selection.assign_coords({time_dim: pr_selection[time_dim] - pd.Timedelta(hours=3)})
    
    return pr_selection


# This function computes a binary hard-buffer mask around a feature, using a constant buffer distance. It returns True for grid cells that are within the specified buffer distance (in km) from any part of the feature, 
# and False otherwise.
def compute_buffer_mask(mask, buffer_km, dx_km=11.7):
    """Computes binary buffer mask around features."""
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    dist_idx = distance_transform_edt(~mask)
    return (dist_idx * dx_km) <= buffer_km

# Build rectangle vertices (used for plotting domain outlines)
def build_rectangle_vertices(rlon, rlat):
    """Builds box vertices for plotting domains."""
    rlon = np.asarray(rlon)
    rlat = np.asarray(rlat)
    if rlon.ndim == 1: rlon, rlat = np.meshgrid(rlon, rlat)
    top    = np.column_stack([rlon[-1, :], rlat[-1, :]])
    bottom = np.column_stack([rlon[ 0, :], rlat[ 0, :]])
    left   = np.column_stack([rlon[:, 0],  rlat[:, 0]])
    right  = np.column_stack([rlon[:, -1], rlat[:, -1]])
    return np.vstack([bottom, right, top[::-1], left[::-1]])



# =========================================================================================
# --- USER CONFIGURATION ---
# =========================================================================================
START_YEAR = 2001
END_YEAR = 2020
years = np.arange(START_YEAR, END_YEAR + 1)
model_name = "RACMO2" # HCLIM or MetUM or RACMO2
datasets = ["ERA5", model_name]

PLOT_OUTPUT_DIR = f"PR_Amount_Contribution_Difference_Maps_Significance_ERA5_{model_name}"
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

# Analysis Parameters
BUFFER_KM = 500
dx_km = 11.17
SIGNIFICANCE_TEST = 'wilcoxon' # "wilcoxon" or "t-test"
P_THRESHOLD = 0.05
EPSILON_NOISE = 0.00  # not used but configurable if we wanted to mask out very small differences as "noise"

# Paths
domain_path = 'PolarRES_WP3_Antarctic_domain_expanded.nc'
inner_domain_path = "PolarRES_WP3_Antarctic_domain.nc"

feature_names = ["AR", "CY", "ACY", "FR", "JET", "OTHER"]
feature_titles = ["(a) ARs", "(b) CYs", "(c) ACYs", "(d) FRs", "(e) JETs", "(f) OTHER"] 

# =========================================================================================
# --- MAIN PROCESSING LOOP ---
# =========================================================================================

# 1. Load Grid Info
dom = xr.open_dataset(domain_path)
rlon, rlat = dom.rlon.values, dom.rlat.values
rlon2D, rlat2D = np.meshgrid(rlon, rlat)
ny, nx = rlat2D.shape
rotated_pole = dom.rotated_pole.attrs
rotated_crs = ccrs.RotatedPole(pole_longitude=rotated_pole['grid_north_pole_longitude'],
                               pole_latitude=rotated_pole['grid_north_pole_latitude'])
dom.close()

data_store = {}

for data_type in datasets:
    save_filename = f'Processed_Fractions_{data_type}_{START_YEAR}_{END_YEAR}.npz'

    if os.path.exists(save_filename):
        loaded = np.load(save_filename, allow_pickle=True)
        data_store[data_type] = loaded['data_dict'].item()
        continue

    dataset_storage = {f: np.zeros((len(years), ny, nx), dtype=np.float32)
                       for f in feature_names}

    for y_idx, year in enumerate(tqdm(years, desc=f"{data_type}")):

        pr_path = (
            f'ERA5/{year}/remapped/ANT_pr_{year}_remapped.nc'
            if data_type == "ERA5"
            else f'PolarRES/{model_name}/nested/{year}/ANT_{model_name}_pr_{year}_nested_con.nc' # .con is only for RACMO2, HCLIM and MetUM still use original w/o it
        )

        pr_name  = 'tp' if data_type == "ERA5" else 'pr'
        time_dim = 'valid_time' if data_type == "ERA5" else 'time'

        try:
            print(f"\n--- [DEBUG] Starting {data_type} for Year {year} ---")
            # --- Target grid & time ---
            mask_path = (
                f'moaap_output_{data_type}_test_PR_all_objects_summed/'
                f'{year}/{year}_{data_type}_ANT_{year}_PR_ObjectMasks__dt-6h_MOAAP-masks.nc'
            )

            t0 = time.time()
            pr_ds = xr.open_dataset(mask_path)
            target_time = pr_ds.time.values
            target_coords = pr_ds.coords
            target_dims = ('time', 'y', 'x')
            print(f"[DEBUG] Mask file opened. Time steps found: {len(target_time)} ({time.time()-t0:.2f}s)")

            # --- Raw precipitation ---
            pr_raw = xr.open_dataset(pr_path, chunks={time_dim: 120})[[pr_name]]

            # --- CENTERED 6h aggregation ---
            print(f"[DEBUG] Starting 6h rolling sum (Lazy Graph construction)...")
            t_sum = time.time()
            pr_6h = sum_to_6h_centered(pr_raw, time_dim)
            print(f"[DEBUG] 6h rolling sum graph built ({time.time()-t_sum:.2f}s)")

            if time_dim != 'time':
                pr_6h = pr_6h.rename({time_dim: 'time'})



            # Ensure pr_6h and target_time have the same length
            common_times = np.intersect1d(pr_6h.time.values, target_time)
            pr_6h = pr_6h.sel(time=common_times)
            target_time = common_times
            print(f"[DEBUG] Times aligned. Final size: {len(target_time)}")

            print("Aligned times, new length:", len(pr_6h.time))


            # --- STRICT time alignment ---
            assert np.all(pd.to_datetime(pr_6h.time.values) ==
                        pd.to_datetime(target_time)), \
                f"Time mismatch in {year} ({data_type})"

            # --- Manual merge ---
            print(f"[DEBUG] Merging 3D feature masks (Lazy)...")
            def get_vals(path, var):
                with xr.open_dataset(path) as ds:
                    return ds[var].values

            ar_path = (
                f'moaap_output_{data_type}_test_AR_IVT_relaxed/'
                f'{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc'
            )

            cy_path = (
                f'moaap_output_{data_type}_test_CY/'
                f'{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc'
            )

            fr_path = (
                f'moaap_output_{data_type}_test_FR/'
                f'{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc'
            )

            jet_path = (
                f'moaap_output_{data_type}_test_JET_300/'
                f'{year}/{year}01_{data_type}_ANT_{year}_ObjectMasks__dt-6h_MOAAP-masks.nc'
            )

            ds_merge = xr.Dataset(
                data_vars={
                    'AR_Objects':  (target_dims, get_vals(ar_path,  'AR_Objects')),
                    'CY_z500_Objects':  (target_dims, get_vals(cy_path,  'CY_z500_Objects')),
                    'ACY_z500_Objects': (target_dims, get_vals(cy_path,  'ACY_z500_Objects')),
                    'FR_Objects':  (target_dims, get_vals(fr_path,  'FR_Objects')),
                    'JET_300_Objects': (target_dims, get_vals(jet_path, 'JET_300_Objects')),
                    'PR_mask':     (target_dims, pr_ds['PR_mask'].values),
                    pr_name:       (target_dims, pr_6h[pr_name].values)
                },
                coords=target_coords
            )

            pr_ds.close()
            pr_raw.close()

            total_pr_year = np.zeros((ny, nx), dtype=np.float32)
            feat_precip_accum = {f: np.zeros((ny, nx), dtype=np.float32)
                                 for f in feature_names}

            for month in range(1, 13):
                t_month = time.time()
                ds_m = ds_merge.sel(time=ds_merge.time.dt.month == month)
                if ds_m.time.size == 0:
                    continue

                print(f"[DEBUG] Processing Month {month:02d} (Steps: {ds_m.time.size})...")

                t_load = time.time()
                pr_vals = ds_m[pr_name].values
                pr_mask = ds_m['PR_mask'].values > 0
                print(f"  [DEBUG] Data loaded/computed for Month {month} in {time.time()-t_load:.2f}s")
                precip_masked = pr_vals * pr_mask

                total_pr_year += np.nansum(precip_masked, axis=0)

                any_feat = np.zeros_like(pr_mask, dtype=bool)

                for f_key, var in [
                    ('AR','AR_Objects'), ('CY','CY_z500_Objects'),
                    ('ACY','ACY_z500_Objects'), ('FR','FR_Objects'),
                    ('JET','JET_300_Objects')
                ]:
                    mask_3d = ds_m[var].values > 0
                    buf_3d = np.zeros_like(mask_3d, dtype=bool)

                    for t in range(mask_3d.shape[0]):
                        if mask_3d[t].any():
                            buf_3d[t] = compute_buffer_mask(mask_3d[t], BUFFER_KM, dx_km)

                    feat_precip_accum[f_key] += np.nansum(precip_masked * buf_3d, axis=0)
                    any_feat |= buf_3d

                feat_precip_accum['OTHER'] += np.nansum(
                    precip_masked * (~any_feat), axis=0
                )

            for f in feature_names:
                dataset_storage[f][y_idx] = np.divide(
                    feat_precip_accum[f], total_pr_year,
                    out=np.zeros_like(total_pr_year),
                    where=total_pr_year > 0
                ) * 100

        except Exception as e:
            print(f"❌ {year} failed:", e)
            raise

    np.savez(save_filename, data_dict=dataset_storage, rlon2D=rlon2D, rlat2D=rlat2D)
    data_store[data_type] = dataset_storage

# =========================================================================================
# --- SIGNIFICANCE TESTING & PLOTTING ---
# =========================================================================================

print("\n🎨 Generating Difference Maps with Significance...")

dom_i = xr.open_dataset(inner_domain_path)
rlon_i, rlat_i = dom_i.rlon.values, dom_i.rlat.values
verts_inner = build_rectangle_vertices(rlon_i, rlat_i)
verts_outer = build_rectangle_vertices(rlon, rlat)
dom_i.close()

fig, axes = plt.subplots(2, 3, figsize=(30,16), subplot_kw={'projection': ccrs.Orthographic(0, -90)}) #(24,13) originally
axes = axes.flatten()

levels = np.array([-25, -20, -15, -10, -5, -2.5, 2.5, 5, 10, 15, 20, 25])
cmap = plt.cm.RdBu_r
norm = BoundaryNorm(levels, ncolors=cmap.N, extend='both')

for i, feat in enumerate(feature_names):
    ax = axes[i]
    print(f"  ... analyzing {feat}")
    
    h_data_3d = data_store[model_name][feat]
    e_data_3d = data_store["ERA5"][feat]
    
    mean_h = np.nanmean(h_data_3d, axis=0)
    mean_e = np.nanmean(e_data_3d, axis=0)
    diff_field = mean_h - mean_e
    
    print(f"    {feat}: Max Diff = {np.nanmax(diff_field):.2f}%, Min Diff = {np.nanmin(diff_field):.2f}%")
    
    p_vals = np.ones((ny, nx))
    
    # Pixel-wise significance
    for y in range(ny):
        for x in range(nx):
            sh = h_data_3d[:, y, x]
            se = e_data_3d[:, y, x]

            # Paired differences
            d = sh - se
            d = d[~np.isnan(d)]
            d = d[d != 0]

            # Minimal safety for Wilcoxon
            if len(d) >= 8:
                try:
                    _, p = stats.wilcoxon(d, mode='approx')
                    p_vals[y, x] = p
                except:
                    p_vals[y, x] = 1.0
            else:
                p_vals[y, x] = 1.0

    # Apply masks
    diff_field[np.abs(diff_field) < EPSILON_NOISE] = 0.0
    insig_mask = p_vals > P_THRESHOLD

    # Plotting
    ax.set_extent([-180, 180, -90, -20], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='50m', linewidth=0.8)
    ax.plot(verts_outer[:,0], verts_outer[:,1], color="#EABB0F", linewidth=2, transform=rotated_crs)
    ax.plot(verts_inner[:,0], verts_inner[:,1], color="k", linewidth=1.5, transform=rotated_crs)
    
    mesh = ax.pcolormesh(rlon2D, rlat2D, diff_field, cmap=cmap, norm=norm, transform=rotated_crs)
    
    # Hatching for non-significance
    ax.contourf(rlon2D, rlat2D, insig_mask, levels=[0.5, 1.5], transform=rotated_crs, 
                hatches=['////'], colors='none', alpha=0)
    
    ax.set_title(f'{feature_titles[i]}', fontsize=20, fontweight='bold')


# Shared Colorbar
#cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.02])
cbar = fig.colorbar(mesh, ax=axes, orientation='horizontal', extend='both', fraction=0.05, pad=0.05, aspect=20)
cbar.set_label('Difference in Precipitation Contribution (% points)', fontsize=20)
cbar.ax.tick_params(labelsize=14)

#plt.subplots_adjust(bottom=0.15, hspace=0.15, wspace=0.1)
save_path = f"{PLOT_OUTPUT_DIR}/PR_Amount_Contribution_Difference_Map_Significance_{SIGNIFICANCE_TEST}_{START_YEAR}_{END_YEAR}_ERA5_{model_name}.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()



print(f"Analysis Complete. Plot saved to: {save_path}")