# This script processes precipitation (pr) files from the specified model and years.
# It converts the units from kg m-2 s-1 (mean flux) to m (total accumulation over 1 hour) by applying a conversion factor of 3.6.
# Additionally, it shifts the time labels forward by 30 minutes to represent the end of the hour (e.g., 00:30 -> 01:00), and align with ERA5's convention for hourly precipitation.
# The original file is backed up with a .raw.nc extension before modification.
# This is the first step of the model data embedding within ERA5 fo hourly precipitation.

import os
import pandas as pd
from tqdm import tqdm
import xarray as xr

model_name = "RACMO2" # HCLIM or MetUM or RACMO2
base_folder = os.path.abspath(f'/moaap_project/PolarRES/{model_name}/raw')
years = range(2001, 2021)

# Conversion factor: (3600 seconds / 1000 kg/m3) = 3.6
# This converts kg m-2 s-1 (mean flux) to m (total accumulation over 1 hour)
CONVERSION_FACTOR = 3.6

for year in years:
    year_dir = os.path.join(base_folder, str(year))
    if not os.path.exists(year_dir):
        continue
    
    # Filter for precipitation files starting with 'pr'
    files = [f for f in os.listdir(year_dir) if f.startswith('pr') 
             and f.endswith('.nc') and not f.endswith('.raw.nc') and not f.startswith('.')]

    for fname in tqdm(files, desc=f"Processing {year}"):
        file_path = os.path.join(year_dir, fname)
        backup_path = file_path.replace('.nc', '.raw.nc')
        temp_path = file_path + ".tmp"

        # Safety check: Prevent double-scaling if script is re-run
        if os.path.exists(backup_path):
            continue

        try:
            # use_cftime=True ensures we handle model calendars (like noleap) correctly
            ds = xr.open_dataset(file_path, use_cftime=True)
            
            if 'pr' in ds.data_vars and ds['pr'].attrs.get('units') == 'kg m-2 s-1':
                
                # 1. Apply Unit Conversion
                ds['pr'] = ds['pr'] * CONVERSION_FACTOR
                
                # 2. Shift Time Labels Forward by 30 Minutes
                # Result: 00:30 (center) -> 01:00 (end of hour) -- just a matter of different convention, physically they are the same
                import datetime
                new_time = ds.indexes['time'] + datetime.timedelta(minutes=30)
                ds = ds.assign_coords(time=new_time)
                
                # 3. Update Metadata
                ds['pr'].attrs.update({
                    'units': 'm',
                    'long_name': 'Total precipitation',
                    'standard_name': 'precipitation_amount',
                    'cell_methods': 'area: time: sum',
                    'comment': 'Units converted from kg m-2 s-1 to m; time shifted +30min to represent end of hour.'
                })

                # Define encoding for the variables (especially the large precipitation array)
                encoding = {
                    'pr': {
                        'zlib': True, 
                        'complevel': 4, # Compression level 1-9
                        'dtype': 'float32'
                    }
                }

                # Save with compression
                ds.to_netcdf(temp_path, encoding=encoding, unlimited_dims=['time'])
                ds.close()
                
                # Atomic swap: backup original and put modified file in its place
                os.rename(file_path, backup_path)
                os.rename(temp_path, file_path)
            else:
                ds.close()

        except Exception as e:
            print(f"\n[ERROR] Failed to process {fname}: {e}")
            if os.path.exists(temp_path): os.remove(temp_path)

        
print("\n Precipitation processing complete.")