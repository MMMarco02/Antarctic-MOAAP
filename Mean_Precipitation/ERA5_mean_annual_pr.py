# Script to compute the mean total annual precipitation grid-by-grid over a period and generate geographical plot.
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import glob
import os
import matplotlib as mpl
from matplotlib.colors import PowerNorm

# --- Configuration ---
# Define the base directory where the yearly files are located
base_dir = os.path.abspath("ERA5/")
# Define the time period for the mean calculation
start_year = 2001
end_year = 2020
years = range(start_year, end_year + 1)
# Define the pattern for input files
file_pattern = "remapped/ANT_pr_{year}_remapped.nc"
# Output file name for the plot
output_plot_file = f"ERA5_mean_annual_precipitation_{start_year}-{end_year}.png"

# --- Rotated Pole Parameters (from "rotated_pole" of PolarCORDEX file) ---
POLE_LAT = 5.0
POLE_LON = 20.0


def compute_and_plot_mean_annual_pr(base_dir, years, file_pattern, output_plot_file, pole_lat, pole_lon):

    print(f"Starting computation for period: {years[0]}-{years[-1]}")

    # 1. DATA LOADING AND PRE-PROCESSING
    file_paths = [
        os.path.join(base_dir, str(year), file_pattern.format(year=year))
        for year in years
    ]

    try:
        # Load the files. Specify the variable name 'tp'
        ds = xr.open_mfdataset(file_paths, combine="nested", concat_dim="valid_time", chunks='auto')
        # Rename the time dimension to 'time' for convenience in xarray functions
        ds = ds.rename_dims({'valid_time': 'time'}).rename({'valid_time': 'time'})
        tp_hourly = ds['tp'] # Total precipitation (accumulated over 1 hour) in meters (m)
    except Exception as e:
        print(f"Error loading files or accessing 'tp' variable: {e}")
        return

    # 2. COMPUTATION: Total Annual Precipitation
    print("Computing total annual precipitation for each year...")

    # The 'tp' variable is total precipitation in 'm' accumulated over the time step (1 hour).
    # To get total annual precipitation in mm/year:
    # 1. Convert units from 'm' to 'mm': * 1000
    tp_mm_hourly = tp_hourly * 1000 # Now in mm

    # 2. Group by year and sum the hourly accumulated values
    # The sum of hourly accumulated precipitation gives the total precipitation for the year.
    pr_annual_sum = tp_mm_hourly.resample(time="YE").sum()

    # 3. COMPUTATION: Mean Total Annual Precipitation
    print("Computing mean total annual precipitation over the period...")
    # Calculate the mean across the 'time' dimension (which is now yearly)
    mean_annual_pr = pr_annual_sum.mean(dim="time")

    # 4. EXTRACT COORDINATES AND DEFINE PROJECTION
    rlon = mean_annual_pr['rlon'].values
    rlat = mean_annual_pr['rlat'].values
    pr_data = mean_annual_pr.values
    
    # Define the Rotated Pole CRS using the exact parameters from ncdump
    rotated_crs = ccrs.RotatedPole(
        pole_latitude=pole_lat,
        pole_longitude=pole_lon
    )

    # 5. PLOTTING
    print("Generating plot...")

    # Define boundaries for the domain outlines (for style consistency)
    rlon_full = rlon
    rlat_full = rlat
    # Dummy inner boundaries
    inner_domain_file = xr.open_dataset("PolarRES_WP3_Antarctic_domain.nc")
    rlon_inner = inner_domain_file['rlon'].values
    rlat_inner = inner_domain_file['rlat'].values

    # --- PLOTTING SETUP ---
    fig, ax = plt.subplots(
        figsize=(12, 10),
        subplot_kw={"projection": ccrs.Orthographic(0, -90)} 
    )

    # Set the geographical extent (Antarctic region focus)
    ax.set_extent([-180, 180, -90, -10], crs=ccrs.PlateCarree())
    ax.coastlines(color="#969696", resolution='50m')

    # Gridlines
    gl = ax.gridlines(draw_labels=False,
                      xlocs=[-180, -120, -60, 0, 60, 120, 180],
                      ylocs=[-90, -75, -60, -45, -30],
                      color="gray", linewidth=0.5, alpha=0.7,
                      crs=ccrs.PlateCarree())
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}
    gl.top_labels = False
    gl.right_labels = False

    # Outer domain outline (Yellow) 
    top = np.column_stack([rlon_full, np.full_like(rlon_full, rlat_full[-1])])
    bot = np.column_stack([rlon_full, np.full_like(rlon_full, rlat_full[0])])
    left = np.column_stack([np.full_like(rlat_full, rlon_full[0]), rlat_full])
    right = np.column_stack([np.full_like(rlat_full, rlon_full[-1]), rlat_full])
    outer = np.vstack([bot, right, top[::-1], left[::-1]])
    ax.fill(outer[:, 0], outer[:, 1],
            facecolor="none",
            edgecolor="#EABB0F",
            linewidth=3,
            transform=rotated_crs)

    # Inner domain outline (Red)
    top_i = np.column_stack([rlon_inner, np.full_like(rlon_inner, rlat_inner[-1])])
    bot_i = np.column_stack([rlon_inner, np.full_like(rlon_inner, rlat_inner[0])])
    left_i = np.column_stack([np.full_like(rlat_inner, rlon_inner[0]), rlat_inner])
    right_i = np.column_stack([np.full_like(rlat_inner, rlon_inner[-1]), rlat_inner])
    inner = np.vstack([bot_i, right_i, top_i[::-1], left_i[::-1]])
    #ax.plot(inner[:, 0], inner[:, 1], color="red", linewidth=2, transform=rotated_crs)


    # --- Mean Annual Precipitation Layer ---
    cmap = "YlGnBu"
    norm = PowerNorm(gamma=0.4)

    # The levels are chosen to highlight the low-value interior and high-value coast.
    contour_levels = [100, 200, 500, 1000, 2000, 4000, 6000 ] 

    # Add Contour Lines
    contour_lines = ax.contour(
        rlon, rlat, pr_data,
        levels=contour_levels,
        colors='black', 
        linewidths=0.8,
        linestyles='solid',
        transform=rotated_crs,
        zorder=1 
    )
    
    # Add labels to the contour lines for easy reading
    ax.clabel(
        contour_lines, 
        inline=True, 
        fontsize=8, 
        fmt='%1.0f' # Format as whole numbers
    )
    
    pcm = ax.pcolormesh(
        rlon, rlat, pr_data,
        cmap=cmap,
        norm=norm,
        transform=rotated_crs, # Crucially, use the correct Rotated Pole CRS
        zorder=0
    )

    # Colorbar 
    cbar = plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04, orientation='horizontal')
    
    # New: Define custom ticks for better readability and alignment with contours
    cbar_ticks = [0, 100, 200, 500, 1000, 2000, 4000, 6000] # Example ticks
    cbar.set_ticks(cbar_ticks) 
    cbar.set_ticklabels([f'{t}' for t in cbar_ticks])
    
    cbar.set_label(f"Mean Total Annual Precipitation (mm/year)", fontsize=14)

    # Title
    #ax.set_title(f"Mean Total Annual Precipitation ({start_year}-{end_year})", fontsize=14)

    # Save plot
    fig.savefig(output_plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved to: {output_plot_file}")
    print("Done!")

# --- Execute the function ---
if __name__ == "__main__":
    compute_and_plot_mean_annual_pr(base_dir, years, file_pattern, output_plot_file, POLE_LAT, POLE_LON)