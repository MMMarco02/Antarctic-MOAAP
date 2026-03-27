import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import os

# --- CONFIGURATION ---
mode        = "_Objects"
data_types  = ["ERA5", "HCLIM", "MetUM", "RACMO2"]
regions     = ["EAN", "WAN", "SOO"]
region_titles = {
    "EAN": "East Antarctica (EAN)",
    "WAN": "West Antarctica (WAN)",
    "SOO": "Southern Ocean (SOO)"
}
colors = {
    "ERA5":   "#E63946",
    "HCLIM":  "#457B9D",
    "MetUM":  "#2D6A4F",
    "RACMO2": "#F4A261"
}
display_names = {
    "ERA5":   "ERA5",
    "HCLIM":  "HCLIM",
    "MetUM":  "MetUM",
    "RACMO2": "RACMO" 
}

# Rotated pole CRS (consistent with PolarCORDEX "rotated_pole" grid)
POLE_LAT    = 5.0
POLE_LON    = 20.0
rotated_crs = ccrs.RotatedPole(pole_latitude=POLE_LAT, pole_longitude=POLE_LON)

# Region extents in PlateCarree 
region_extents = {
    "EAN": (  0,  180, -90, -55),
    "WAN": (-180,   0, -90, -55),
    "SOO": (-180, 180, -75, -45),
}

# --- LOAD POLARCORDEX (INNER) DOMAIN BOUNDARY ---
domain_file = "/cluster/work/users/mmarco02/moaap_project/PolarRES_WP3_Antarctic_domain.nc"
ds_domain   = xr.open_dataset(domain_file)
rlon_inner  = ds_domain['rlon'].values
rlat_inner  = ds_domain['rlat'].values
ds_domain.close()

# Build closed boundary path from the 4 edges
top_i   = np.column_stack([rlon_inner,                          np.full_like(rlon_inner, rlat_inner[-1])])
bot_i   = np.column_stack([rlon_inner,                          np.full_like(rlon_inner, rlat_inner[0])])
left_i  = np.column_stack([np.full_like(rlat_inner, rlon_inner[0]),  rlat_inner])
right_i = np.column_stack([np.full_like(rlat_inner, rlon_inner[-1]), rlat_inner])
inner_boundary_path = np.vstack([bot_i, right_i, top_i[::-1], left_i[::-1]])

# --- LOAD DATA ---
def load_dataframe(data_type, region):
    csv_path = f"top100_{region}_with_features_{data_type}{mode}.csv"
    if not os.path.exists(csv_path):
        print(f"⚠️  File not found: {csv_path}")
        return None
    return pd.read_csv(csv_path)

# --- FIGURE SETUP ---
projection = ccrs.SouthPolarStereo()
fig, axes  = plt.subplots(
    4, 3,
    figsize=(15, 18),
    subplot_kw=dict(projection=projection)
)

# --- PLOTTING ---
for row, dt in enumerate(data_types):
    for col, reg in enumerate(regions):
        ax  = axes[row, col]
        ext = region_extents[reg]

        # Map extent and features
        ax.set_extent([ext[0], ext[1], ext[2], ext[3]], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN,     facecolor='#d6eaf8', zorder=0)
        ax.add_feature(cfeature.LAND,      facecolor='#f0ede8', zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6,        zorder=2)
        ax.gridlines(
            draw_labels=False,
            linewidth=0.4,
            color='grey',
            alpha=0.5,
            linestyle='--'
        )

        # Inner PolarCORDEX domain boundary
        ax.plot(
            inner_boundary_path[:, 0], inner_boundary_path[:, 1],
            color='black', linewidth=1.2,
            transform=rotated_crs,
            zorder=4,
            label='PolarCORDEX inner domain'
        )

        # Event locations
        df = load_dataframe(dt, reg)
        if df is not None:
            ax.scatter(
                df['lon'], df['lat'],
                transform=ccrs.PlateCarree(),
                color=colors[dt],
                s=18,
                alpha=0.7,
                edgecolors='white',
                linewidths=0.3,
                zorder=5
            )

        # Labels
        if row == 0:
            ax.set_title(region_titles[reg], fontsize=12,
                         fontweight='bold', pad=10)
        if col == 0:
            ax.text(
                -0.08, 0.5, display_names[dt],
                transform=ax.transAxes,
                fontsize=12, fontweight='bold',
                va='center', ha='right', rotation=90
            )

# --- SPACING ---
plt.subplots_adjust(wspace=0.1, hspace=0.15,
                    left=0.08, right=0.97,
                    top=0.95, bottom=0.08)

# --- LEGEND ---
legend_handles = [
    mpatches.Patch(color=colors[dt], label=display_names[dt], alpha=0.85)
    for dt in data_types
]
legend_handles.append(
    plt.Line2D([0], [0], color='black', linewidth=1.2,
               label='PolarCORDEX inner domain')
)
fig.legend(
    handles=legend_handles,
    loc='lower center',
    ncol=5,
    fontsize=11,
    frameon=True,
    framealpha=0.9,
    edgecolor='grey',
    bbox_to_anchor=(0.5, 0.01)
)

# --- SAVE ---
output_fig = f"Antarctica_top100_Events_Locations{mode}.png"
plt.savefig(output_fig, dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ Saved: {output_fig}")