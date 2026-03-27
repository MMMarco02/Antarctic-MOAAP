# This script computes ivte and ivtn from wind components (ua, va) and specific humidity (hus) using the trapezoidal rule for vertical integration. 
# It loads each level one at a time to save memory, applies a surface mask based on geopotential height to avoid below-surface levels, 
# and saves the results as new NetCDF files. The script also handles the known Lustre cache corruption issue by copying files to local /tmp before reading.

# Notes: 
# 1. Unfortunately, this code is specifically applied to RACMO2 data, but the logic used to compute ivte and ivtn for ERA5, HCLIM and MetUM is the same.
# 2. This script shoudl run after the "prep_model_data.py" script

import os
import shutil
import numpy as np
import xarray as xr
import gc
import dask

# This part can likely be removed if the Lustre issue is resolved, but we keep it here for safety.
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
dask.config.set(scheduler='synchronous')


# ---------------------------------------------------------------------------
# Load one level — copy to /dev/shm first to bypass Lustre cache corruption
# ---------------------------------------------------------------------------
def load_single_level(prefix, lvl, year_path, year):
    fname = (
        f"{prefix}{lvl}_ANT-12_ERA5_evaluation_r1i1p1f1_UU-IMAU_RACMO24P-NN_v1-r1"
        f"_6hr_{year}01010000-{year}12311800.nc"
    ) # specific to RACMO2 data — modify if using other datasets with different naming conventions
    src_path = os.path.join(year_path, fname)
    var_name = f"{prefix}{lvl}"

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"File not found: {src_path}")

    tmp_path = os.path.join("/dev/shm", f"_ivt_{os.getpid()}_{fname}")

    try:
        print(f"  --> Copying to /dev/shm: {fname}")
        shutil.copy2(src_path, tmp_path)

        ds  = xr.open_dataset(tmp_path, engine='netcdf4')
        da  = ds[var_name].load()

        crs_obj = None
        for crs_var in ['crs', 'rotated_pole']:
            if crs_var in ds:
                crs_obj = ds[crs_var].copy(deep=True).load()  # force into RAM
                break
        ds.close()

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        gc.collect()

    return da, crs_obj


# ---------------------------------------------------------------------------
# IVT computation — trapezoidal rule
# ---------------------------------------------------------------------------
def compute_ivt_trapz(levels, year_path, year, z_sfc_da):
    """
    Trapezoidal integration:
        IVT = (1/g) * integral(q * wind, dp)
    
    np.trapz linearly interpolates between levels, which is more accurate
    than the box/midpoint method — especially with coarse, uneven level spacing.
    
    Surface masking: levels below the surface orography are set to 0
    before integration, so they contribute nothing to the integral.
    """
    g = 9.80665

    # Sort levels and convert to Pa
    levels_sorted = sorted(levels)
    p_pa = np.array(levels_sorted, dtype=float)
    if p_pa.max() < 2000:
        p_pa *= 100.0

    print(f"Pressure levels (Pa): {p_pa}")
    print(f"Level spacing (Pa):   {np.diff(p_pa)}")

    # --- Load all levels for all 4 variables ---
    ua_list, va_list, hus_list, zg_list = [], [], [], []
    crs_obj = None

    for lvl in levels_sorted:
        print(f"\n--- Loading level {lvl} hPa ---")

        ua_da,  crs = load_single_level("ua",  lvl, year_path, year)
        va_da,  _   = load_single_level("va",  lvl, year_path, year)
        hus_da, _   = load_single_level("hus", lvl, year_path, year)
        zg_da,  _   = load_single_level("zg",  lvl, year_path, year)

        if crs is not None and crs_obj is None:
            crs_obj = crs

        # Convert zg from m to Pa-equivalent is NOT needed —
        # we compare zg (geopotential height, metres) against z_sfc (orog, metres)
        ua_list.append(ua_da.values)
        va_list.append(va_da.values)
        hus_list.append(hus_da.values)
        zg_list.append(zg_da.values)

        # Keep coords from last da for output
        template = ua_da

        del ua_da, va_da, hus_da, zg_da
        gc.collect()

    # Stack into (plev, time, y, x)
    print("\nStacking levels...")
    ua_arr  = np.stack(ua_list,  axis=0).astype("float32")  # (plev, time, y, x)
    va_arr  = np.stack(va_list,  axis=0).astype("float32")
    hus_arr = np.stack(hus_list, axis=0).astype("float32")
    zg_arr  = np.stack(zg_list,  axis=0).astype("float32")

    del ua_list, va_list, hus_list, zg_list
    gc.collect()

    # --- Surface mask ---
    # z_sfc shape: (y, x) — broadcast against (plev, time, y, x)
    z_sfc_np = z_sfc_da.values[np.newaxis, np.newaxis, :, :]  # (1, 1, y, x)
    below_surface = zg_arr < z_sfc_np  # True where level is underground

    hus_arr[below_surface] = 0.0
    ua_arr[below_surface]  = 0.0
    va_arr[below_surface]  = 0.0

    del zg_arr, below_surface
    gc.collect()

    # --- Trapezoidal integration along plev axis (axis=0) ---

    print("Computing IVTE with trapezoidal rule...")
    ivte_np = np.trapezoid(hus_arr * ua_arr, p_pa, axis=0) / g
    print("Computing IVTN with trapezoidal rule...")
    ivtn_np = np.trapezoid(hus_arr * va_arr, p_pa, axis=0) / g

    del ua_arr, va_arr, hus_arr
    gc.collect()

    # Wrap back into xarray using the template coords (time, y, x)
    dims   = [d for d in template.dims if d != "plev"] if "plev" in template.dims else template.dims
    coords = {k: v for k, v in template.coords.items() if k != "plev"}

    ivte = xr.DataArray(ivte_np.astype("float32"), dims=dims, coords=coords)
    ivtn = xr.DataArray(ivtn_np.astype("float32"), dims=dims, coords=coords)

    return ivte, ivtn, crs_obj


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------
def save_ivt(ivte, ivtn, crs_obj, year_path, year):
    grid_mapping_name = crs_obj.name if crs_obj is not None else "crs"

    for da, short_name, var_name in zip([ivte, ivtn], ["ivte_computed", "ivtn_computed"], ["ivte", "ivtn"]):
        da.attrs = {
            "units":        "kg m-1 s-1",
            "long_name":    f"Integrated Vapor Transport ({var_name.upper()})",
            "grid_mapping": grid_mapping_name,
        }
        ds_out = da.to_dataset(name=var_name)  # internal variable name
        if crs_obj is not None:
            ds_out[grid_mapping_name] = crs_obj

        fname = (
            f"{short_name}_ANT-12_ERA5_evaluation_r1i1p1f1_UU-IMAU_RACMO24P-NN"
            f"_v1-r1_6hr_{year}01010000-{year}12311800.nc"
        )
        out_path = os.path.join(year_path, fname)
        print(f"Saving {short_name} → {out_path}")
        ds_out.to_netcdf(out_path)
        print(f"Saved ✓")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    base_path = "PolarRES/RACMO2/raw" # folder where to save the computed ivte and ivtn files (specific for RACMO)
    years     = list(range(2001, 2021))
    levels    = [300, 400, 500, 600, 700, 850, 925, 1000] # available pressure levels in RCMs

    # Load surface geopotential
    sfc_src = os.path.join(base_path, "2001/zgsfc_ANT-12_ERA5_evaluation_r1i1p1f1_UU-IMAU_RACMO24P-NN_v1-r1_fx.nc")
    sfc_tmp = os.path.join("/dev/shm", f"_ivt_{os.getpid()}_zgsfc.nc")
    try:
        print("Copying surface geopotential to /dev/shm...")
        shutil.copy2(sfc_src, sfc_tmp)
        z_sfc = xr.open_dataset(sfc_tmp)["orog"].load()
        print("Surface geopotential loaded ✓")
    finally:
        if os.path.exists(sfc_tmp):
            os.remove(sfc_tmp)

    for year in years:
        year_path = os.path.join(base_path, str(year))
        print(f"\n{'='*60}\n  Processing Year {year}\n{'='*60}")

        try:
            ivte, ivtn, crs_obj = compute_ivt_trapz(levels, year_path, year, z_sfc)
            save_ivt(ivte, ivtn, crs_obj, year_path, year)
            del ivte, ivtn
            gc.collect()

        except Exception as e:
            print(f"\nFailed at year {year}: {e}")
            import traceback
            traceback.print_exc()
