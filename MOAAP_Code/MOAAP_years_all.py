import sys
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm
import pyproj
from scipy.interpolate import griddata
import gc
import traceback
import time

# Clean path setup (optional but safe)
sys.path = [p for p in sys.path if '/cluster/software/' not in p]
if 'CONDA_PREFIX' in os.environ:
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    sys.path.insert(0, os.path.join(os.environ['CONDA_PREFIX'], 'lib', pyver, 'site-packages'))
os.environ['PYTHONNOUSERSITE'] = '1'

# Imports that depend on environment
from Tracking_Functions_ANT import moaap

print("✅ All imports successful")
############################################################################


def process_year(year, data_type):
    """Run MOAAP for a single year sequentially."""
    start = time.time()
    print(f"\n🧠 Starting year {year}...")

    try:
        # Grid coords (same for all datasets)
        data_vars = xr.open_dataset('ERA5/2001/remapped/ANT_All_vars_2001_remapped.nc')
        lat2D = data_vars['lat'].values
        lon2D = data_vars['lon'].values
        data_vars.close()

        if data_type == "ERA5":
            print(f"Using ERA5 data for year {year}")

        ########################## --- IMPORTANT: MODIFY THIS BASED ON WHAT YOU TRACK --- ##########################

            # AR tracking using PolarRES data
            ivte_ds = xr.open_dataset(f'ERA5/{year}/remapped/ANT_ivte_computed_{year}_remapped.nc')
            ivtn_ds = xr.open_dataset(f'ERA5/{year}/remapped/ANT_ivtn_computed_{year}_remapped.nc')
            time_datetime = pd.to_datetime(np.array(ivte_ds['time'].values, dtype='datetime64'))

            # CY tracking using ERA5 data
            #z500_ds = xr.open_dataset(f'ERA5/{year}/remapped/ANT_{data_type}_zg500_{year}_remapped.nc')
            z500_name = 'z'
            #time_datetime = pd.to_datetime(np.array(z500_ds['time'].values, dtype='datetime64'))

            # JET200 tracking using ERA5 data
            #ua200_ds = xr.open_dataset(f'ERA5/{year}/remapped/ANT_{data_type}_ua200_{year}_remapped.nc')
            #va200_ds = xr.open_dataset(f'ERA5/{year}/remapped/ANT_{data_type}_va200_{year}_remapped.nc')
            #u200 = u300_ds['u'].sel(pressure_level=200)
            #v200 = v300_ds['v'].sel(pressure_level=200)
            # u_name = 'u'
            # v_name = 'v'
            #time_datetime = pd.to_datetime(np.array(ua200_ds['time'].values, dtype='datetime64'))

            # JET300 tracking using ERA5 data
            #ua300_ds = xr.open_dataset(f'ERA5/{year}/remapped/ANT_{data_type}_ua300_{year}_remapped.nc')
            #va300_ds = xr.open_dataset(f'ERA5/{year}/remapped/ANT_{data_type}_va300_{year}_remapped.nc')
            #u300 = u300_ds['u'].sel(pressure_level=300)
            #v300 = v300_ds['v'].sel(pressure_level=300)
            # u_name = 'u'
            # v_name = 'v'
            #time_datetime = pd.to_datetime(np.array(ua300_ds['time'].values, dtype='datetime64'))

            # FR tracking using ERA5 data
            #ua850_ds = xr.open_dataset(f'ERA5/{year}/remapped/ANT_ua850_{year}_remapped.nc')
            # u_name = 'u'
            #va850_ds = xr.open_dataset(f'ERA5/{year}/remapped/ANT_va850_{year}_remapped.nc')
            # v_name = 'v'
            #ta850_ds = xr.open_dataset(f'ERA5/{year}/remapped/ANT_ta850_{year}_remapped.nc')
            # t_name = 't'
            # Geopotential at surface for topography mask is only one file! (needed for fronts)
            #zgsfc_ds = xr.open_dataset(f'ERA5/{year}/remapped/ANT_zgsfc_{year}_remapped.nc')
            # zsfc_name = 'z'
            #time_datetime = pd.to_datetime(np.array(ua850_ds['time'].values, dtype='datetime64'))
            
            # For surface cyclones tracking (only for ERA5)
            #slp_ds = xr.open_dataset(f'ERA5/{year}/remapped/ANT_slp_{year}_remapped.nc')
            #time_datetime = pd.to_datetime(np.array(slp_ds['valid_time'].values, dtype='datetime64'))
            #slp_name = 'msl'

            #pr_name = 'tp'

        else:
            print(f"Using {data_type} data for year {year}")

            # AR tracking using PolarRES data
            ivte_ds = xr.open_dataset(f'PolarRES/{data_type}/nested/{year}/ANT_{data_type}_ivte_computed_{year}_nested.nc')
            ivtn_ds = xr.open_dataset(f'PolarRES/{data_type}/nested/{year}/ANT_{data_type}_ivtn_computed_{year}_nested.nc')
            time_datetime = pd.to_datetime(np.array(ivte_ds['time'].values, dtype='datetime64'))

            # CY tracking using PolarRES data
            #z500_ds = xr.open_dataset(f'PolarRES/{data_type}/nested/{year}/ANT_{data_type}_zg500_{year}_nested.nc')
            # z_name = 'zg500'
            #time_datetime = pd.to_datetime(np.array(z500_ds['time'].values, dtype='datetime64'))

            # JET 200 tracking using PolarRES data
            #ua200_ds = xr.open_dataset(f'PolarRES/{data_type}/nested/{year}/ANT_{data_type}_ua200_{year}_nested.nc')
            #va200_ds = xr.open_dataset(f'PolarRES/{data_type}/nested/{year}/ANT_{data_type}_va200_{year}_nested.nc')
            # u_name = 'ua200'
            # v_name = 'va200'
            #time_datetime = pd.to_datetime(np.array(ua200_ds['time'].values, dtype='datetime64'))

            # JET 300 tracking using PolarRES data
            #ua300_ds = xr.open_dataset(f'PolarRES/{data_type}/nested/{year}/ANT_{data_type}_ua300_{year}_nested.nc')
            #va300_ds = xr.open_dataset(f'PolarRES/{data_type}/nested/{year}/ANT_{data_type}_va300_{year}_nested.nc')
            # u_name = 'ua300'
            # v_name = 'va300'
            #time_datetime = pd.to_datetime(np.array(ua300_ds['time'].values, dtype='datetime64'))

            # FR tracking using PolarRES data
            #ua850_ds = xr.open_dataset(f'PolarRES/{data_type}/nested/{year}/ANT_{data_type}_ua850_{year}_nested.nc')
            # u_name = 'ua850'
            #va850_ds = xr.open_dataset(f'PolarRES/{data_type}/nested/{year}/ANT_{data_type}_va850_{year}_nested.nc')
            # v_name = 'va850'
            #ta850_ds = xr.open_dataset(f'PolarRES/{data_type}/nested/{year}/ANT_{data_type}_ta850_{year}_nested.nc')
            # t_name = 'ta850'
            # Geopotential at surface for topography mask is only one file!
            #zgsfc_ds = xr.open_dataset(f'PolarRES/{data_type}/nested/2001/ANT_{data_type}_zgsfc_nested.nc')
            # z_sfcname = 'zgsfc'
            #time_datetime = pd.to_datetime(np.array(ua850_ds['time'].values, dtype='datetime64'))

            pr_name = 'pr'
            


        dT = 6  # time step
        DataName = f'{data_type}_ANT_{year}'
        ############################################################################################
        # Modify this based on what you track
        Mask = np.where(np.isfinite(ivte_ds['ivte'][0]), 1, 0)
        OutputFolder = os.path.abspath(f'moaap_output_{data_type}_test_AR_IVT_relaxed/{year}') + '/'
        ###########################################################################################
        os.makedirs(OutputFolder, exist_ok=True)

        # Run MOAAP
        object_split = moaap(
            lon2D,
            lat2D,
            time_datetime,
            dT,
            Mask,
            Gridspacing=11170, # grid spacing in meters (polarCORDEX with 0.11° resolution)
            #v850 = va850_ds[v_name].values,
            #u850 = ua850_ds[u_name].values,
            #t850 = ta850_ds[t_name].values,
            #q850 = data_vars['Q850'].values,
            #slp = data_vars[slp_name].values,
            ivte = ivte_ds['ivte'].values,
            ivtn = ivtn_ds['ivtn'].values,
            #z500 = z500_ds[z_name].values,
            #v200 = va200_ds[v_name].values,
            #u200 = ua200_ds[u_name].values,
            v300 = None,
            u300 = None,
            # pr   = data_vars[pr_name].values,
            # z = zgsfc_ds[zsfc_name].values,  # geopotential at surface (for topography mask)
            #u70 = u70_vars.values,
            u70 = None,
            DataName = DataName,
            OutputFolder = OutputFolder,
        )

        # Cleanup
        data_vars.close()
        del lat2D, lon2D, object_split
        gc.collect()

        elapsed = (time.time() - start) / 60
        print(f"✅ [Year {year}] Finished in {elapsed:.2f} min")
        return (year, "ok")

    except Exception as e:
        tb = traceback.format_exc()
        print(f"❌ [Year {year}] ERROR: {e}\n{tb}")
        try:
            data_vars.close()
        except Exception:
            pass
        gc.collect()
        return (year, "error")


if __name__ == "__main__":

    years = list(range(2001, 2021)) # 2001 to 2020
    results = []
    data_type = "ERA5"  # ERA5 or HCLIM or MetUM or RACMO2

    for year in years:
        result = process_year(year, data_type)
        results.append(result)

    print("\n🏁 All years completed.")
    print("Results:", results)


