
import os
import numpy as np
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
import json
from pathlib import Path
import warnings

import json
try:
    with open("../SDM_config.json", "r", encoding="utf-8") as _f:
        _cfg = json.load(_f)
except:
    _cfg = {}


# Use dask for cleaner memory management if available, else numpy
# warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Configuration ---
# Input Directory (Monthly CMIP6 Data)
CMIP6_KOREA_DIR = Path(_cfg.get("CMIP6_KOREA_DIR", "../Climate_Raw/AR6_Monthly"))

# Output Directory
BASE_OUT_DIR = Path(_cfg.get("OUTPUT_BASE_DIR", "../Model/")) / "Input_Bioclim_KMA"

# Reference Grid (for Reprojection)
REF_ASCII_PATH = _cfg.get("REF_ASCII_PATH", "../Model/sc1gr6.fil.asc")
TARGET_CRS = "EPSG:5186"

# Scenarios to Process
SCENARIOS = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
# Years to Process
YEARS = range(2021, 2051) # 2021 to 2050 (User accepted 2021 as start)

# Variable Mapping (NetCDF Variable Name per file type)
# File patterns: AR6_{SSP}_5ENSMN_skorea_{VAR}_gridraw_monthly_2021_2100.nc
# Variables: TAMAX (tmax), TAMIN (tmin), RN (prec), TA (tavg)
 
NC_VAR_NAMES = { # Internal variable name inside NetCDF
    'tmax': 'TAMAX',
    'tmin': 'TAMIN',
    'tavg': 'TA',
    'prec': 'RN'
}

def load_reference_grid():
    """Load reference grid for reprojection match."""
    return rioxarray.open_rasterio(REF_ASCII_PATH).squeeze(drop=True)

def calculate_bioclim(tmin, tmax, tavg, prec):
    """
    Calculate 19 Bioclimatic variables.
    tmin, tmax, tavg, prec: xarray DataArrays with dimension 'month' (size 12).
    """
    
    # BIO1 = Annual Mean Temperature
    # Use tavg directly if available
    bio1 = tavg.mean(dim='month')
    
    # BIO2 = Mean Diurnal Range (Mean of monthly (max temp - min temp))
    bio2 = (tmax - tmin).mean(dim='month')
    
    # BIO4 = Temperature Seasonality (standard deviation * 100)
    bio4 = tavg.std(dim='month') * 100
    
    # BIO5 = Max Temperature of Warmest Month
    bio5 = tmax.max(dim='month')
    
    # BIO6 = Min Temperature of Coldest Month
    bio6 = tmin.min(dim='month')
    
    # BIO7 = Temperature Annual Range (BIO5-BIO6)
    bio7 = bio5 - bio6
    
    # BIO3 = Isothermality (BIO2/BIO7) * 100
    # Avoid div by zero
    bio3 = (bio2 / bio7) * 100
    
    # BIO12 = Annual Precipitation
    bio12 = prec.sum(dim='month')
    
    # BIO13 = Precipitation of Wettest Month
    bio13 = prec.max(dim='month')
    
    # BIO14 = Precipitation of Driest Month
    bio14 = prec.min(dim='month')
    
    # BIO15 = Precipitation Seasonality (Coefficient of Variation)
    # std / mean * 100. Add epsilon to mean to avoid div by zero.
    bio15 = (prec.std(dim='month') / (prec.mean(dim='month') + 1e-9)) * 100
    
    # QUARTERLY VARIABLES
    # We use a rolling window of 3 months to calculate quarterly sums/means.
    # To correctly handle the circular nature of the year (i.e., Winter crossing Dec-Jan),
    # we pad the time series: [Dec] + [Jan...Dec] + [Jan].
    tmean_concat = xr.concat([tavg.isel(month=-1), tavg, tavg.isel(month=0)], dim='month')
    prec_concat = xr.concat([prec.isel(month=-1), prec, prec.isel(month=0)], dim='month')
    
    # Rolling sum/mean with center=True on the padded array.
    # We slice (1, 13) to extract the 12 valid quarters centered on each month.
    # Index 0 (Jan slot) -> Window centered on Jan (Dec-Jan-Feb) -> DJF
    # Index 1 (Feb slot) -> Window centered on Feb (Jan-Feb-Mar) -> JFM
    # ...
    # Index 11 (Dec slot) -> Window centered on Dec (Nov-Dec-Jan) -> NDJ
    # This covers all 12 possible continuous 3-month periods.
    tmean_quart = tmean_concat.rolling(month=3, center=True).mean().isel(month=slice(1, 13))
    prec_quart = prec_concat.rolling(month=3, center=True).sum().isel(month=slice(1, 13))
    
    # BIO16 = Precipitation of Wettest Quarter (Max of quarterly precip)
    bio16 = prec_quart.max(dim='month')
    
    # BIO17 = Precipitation of Driest Quarter (Min of quarterly precip)
    bio17 = prec_quart.min(dim='month')
    
    # BIO8 = Mean Temperature of Wettest Quarter
    # Logic: Identify the quarter(s) where Precip == MaxPrecip (Bio16), then take the Tmean of that quarter.
    # If multiple quarters tie for max precip, we average their Tmean values.
    bio8 = tmean_quart.where(prec_quart == bio16).mean(dim='month')
    
    # BIO9 = Mean Temperature of Driest Quarter
    bio9 = tmean_quart.where(prec_quart == bio17).mean(dim='month')
    
    # BIO10 = Mean Temperature of Warmest Quarter (Max of quarterly Tmean)
    bio10 = tmean_quart.max(dim='month')
    
    # BIO11 = Mean Temperature of Coldest Quarter (Min of quarterly Tmean)
    bio11 = tmean_quart.min(dim='month')
    
    # BIO18 = Precipitation of Warmest Quarter
    # Logic: Identify quarter(s) with Max Tmean (Bio10), get their Precip.
    bio18 = prec_quart.where(tmean_quart == bio10).mean(dim='month')
    
    # BIO19 = Precipitation of Coldest Quarter
    # Logic: Identify quarter(s) with Min Tmean (Bio11), get their Precip.
    bio19 = prec_quart.where(tmean_quart == bio11).mean(dim='month')
    
    return {
        'bio1': bio1, 'bio2': bio2, 'bio3': bio3, 'bio4': bio4, 'bio5': bio5,
        'bio6': bio6, 'bio7': bio7, 'bio8': bio8, 'bio9': bio9, 'bio10': bio10,
        'bio11': bio11, 'bio12': bio12, 'bio13': bio13, 'bio14': bio14, 'bio15': bio15,
        'bio16': bio16, 'bio17': bio17, 'bio18': bio18, 'bio19': bio19
    }

def process_scenario(ssp_scenario, ref_da):
    print(f"\nProcessing Scenario: {ssp_scenario}")
    
    # Define/Create Output Directory
    out_dir = BASE_OUT_DIR / ssp_scenario 
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Input files
    # tmax: "TAMAX", tmin: "TAMIN", tavg: "TA", prec: "RN"
    f_tmax = CMIP6_KOREA_DIR / f"AR6_{ssp_scenario.upper()}_5ENSMN_skorea_TAMAX_gridraw_monthly_2021_2100.nc"
    f_tmin = CMIP6_KOREA_DIR / f"AR6_{ssp_scenario.upper()}_5ENSMN_skorea_TAMIN_gridraw_monthly_2021_2100.nc"
    f_tavg = CMIP6_KOREA_DIR / f"AR6_{ssp_scenario.upper()}_5ENSMN_skorea_TA_gridraw_monthly_2021_2100.nc"
    f_prec = CMIP6_KOREA_DIR / f"AR6_{ssp_scenario.upper()}_5ENSMN_skorea_RN_gridraw_monthly_2021_2100.nc"
    
    # Check existence
    if not (f_tmax.exists() and f_tmin.exists() and f_tavg.exists() and f_prec.exists()):
        print(f"[ERROR] Missing input files for {ssp_scenario}. Skipping.")
        return

    # Open Datasets (decode_times=False to handle 360_day without cftime if needed)
    ds_tmax = xr.open_dataset(f_tmax, decode_times=False)
    ds_tmin = xr.open_dataset(f_tmin, decode_times=False)
    ds_tavg = xr.open_dataset(f_tavg, decode_times=False)
    ds_prec = xr.open_dataset(f_prec, decode_times=False)
    
    da_tmax_all = ds_tmax['TAMAX']
    da_tmin_all = ds_tmin['TAMIN']
    da_tavg_all = ds_tavg['TA']
    da_prec_all = ds_prec['RN']
    
    # Loop Years
    for year in YEARS:
        print(f"  Year {year}...", end=" ", flush=True)
        
        # Calculate Index
        # Start 2021 == Index 0
        start_idx = (year - 2021) * 12
        end_idx = start_idx + 12
        
        # Extract 12 months
        try:
            tmax_yr = da_tmax_all.isel(time=slice(start_idx, end_idx))
            tmin_yr = da_tmin_all.isel(time=slice(start_idx, end_idx))
            tavg_yr = da_tavg_all.isel(time=slice(start_idx, end_idx))
            prec_yr = da_prec_all.isel(time=slice(start_idx, end_idx))
            
            # Verify we got 12 months
            if tmax_yr.sizes['time'] != 12:
                print(f"Skipping (incomplete data range)")
                continue
                
            # Rename 'time' to 'month' for clarity in calc function
            tmax_yr = tmax_yr.rename({'time': 'month'})
            tmin_yr = tmin_yr.rename({'time': 'month'})
            tavg_yr = tavg_yr.rename({'time': 'month'})
            prec_yr = prec_yr.rename({'time': 'month'})
            
            # Calculate Bioclim
            bioclims = calculate_bioclim(tmin_yr, tmax_yr, tavg_yr, prec_yr)
            
            # Save Each Bio Variable
            for bio_key, bio_da in bioclims.items():
                # Reproject to Target CRS using Reference Grid Match
                if not bio_da.rio.crs:
                    bio_da = bio_da.rio.write_crs("EPSG:4326")
                
                # Reproject to Match Reference (EPSG:5186)
                bio_reproj = bio_da.rio.reproject_match(ref_da, resampling=Resampling.bilinear)
                
                # Fill NoData
                bio_reproj = bio_reproj.fillna(-9999)
                
                # Save
                fname = f"kma_{ssp_scenario}_{year}_{bio_key}.asc"
                fpath = out_dir / fname
                
                # Using AAIGrid driver
                bio_reproj.rio.to_raster(fpath, driver="AAIGrid")
            
            print("Done.")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

def main():
    print("--- Calculating Annual Bioclim (2021-2050) from CMIP6 Monthly Data ---")
    
    # Load Ref Grid once
    try:
        ref_da = load_reference_grid()
        # Ensure CRS is set on ref if missing (though creating from ASC might not have it, we know the target)
        if not ref_da.rio.crs:
            ref_da = ref_da.rio.write_crs(TARGET_CRS)
    except Exception as e:
        print(f"[ERROR] Could not load reference grid: {e}")
        return

    for ssp in SCENARIOS:
        process_scenario(ssp, ref_da)

if __name__ == "__main__":
    main()
