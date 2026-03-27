# This script processes geopotential height (zg) files from the specified model and years.
# It checks if the zg variable is in meters and converts it to geopotential (m2 s-2) if needed (so to align with ERA5).
# The original file is backed up with a .raw.nc extension before modification.
# It represent the first step of the model data embedding within ERA5.

import xarray as xr
import os
from tqdm import tqdm

model_name = "RACMO2" # HCLIM or MetUM or RACMO2
base_folder = os.path.abspath(f'PolarRES/{model_name}/raw')
years = range(2001, 2021)
G = 9.80665 

for year in years:
    year_dir = os.path.join(base_folder, str(year))
    if not os.path.exists(year_dir):
        continue
    
    # 1. Strictly filter for files starting with 'zg'
    files = [f for f in os.listdir(year_dir) if f.startswith('zg') 
             and f.endswith('.nc') and not f.endswith('.raw.nc') and not f.startswith('.')]

    for fname in tqdm(files, desc=f"Processing {year}"):
        file_path = os.path.join(year_dir, fname)
        backup_path = file_path.replace('.nc', '.raw.nc')
        temp_path = file_path + ".tmp"

        # Safety check: Don't process if a backup already exists (prevents double-scaling)
        if os.path.exists(backup_path):
            print(f"\n[SKIP] {fname} already has a backup (.raw.nc).")
            continue

        if os.path.getsize(file_path) == 0:
            print(f"\n[SKIP] {fname} is 0 bytes.")
            continue

        try:
            ds = xr.open_dataset(file_path)
            
            # 2. Double-check: We only care about the variable matching the 'zg' filename
            # Usually the variable name in the file is 'zg'
            var_candidates = [v for v in ds.data_vars if v.startswith('zg')]
            
            if not var_candidates:
                print(f"\n[SKIP] No 'zg' variable found inside {fname}.")
                ds.close()
                continue
                
            var_name = var_candidates[0]
            modified = False

            # Check if units are in meters (m)
            current_units = ds[var_name].attrs.get('units', '')
            if current_units == 'm':
                print(f"\nScaling {var_name} in {fname} to geopotential...")
                
                # Conversion: Geopotential Height (m) -> Geopotential (m2 s-2)
                ds[var_name] = (ds[var_name] * G).astype('float32')
                
                # Metadata Update
                ds[var_name].attrs['units'] = 'm2 s-2'
                ds[var_name].attrs['long_name'] = 'Geopotential'
                ds[var_name].attrs['standard_name'] = 'geopotential'
                modified = True

            if modified:
                # Save to temp, rename original to .raw.nc, rename temp to original
                ds.to_netcdf(temp_path)
                ds.close()
                
                os.rename(file_path, backup_path)
                os.rename(temp_path, file_path)
                print(f"✅ Fixed {fname}")
            else:
                ds.close()

        except Exception as e:
            print(f"\n[ERROR] Failed to process {fname}: {e}")
            if os.path.exists(temp_path): os.remove(temp_path)
            continue