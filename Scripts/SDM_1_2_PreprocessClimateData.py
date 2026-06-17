

# --- CMIP6 Data Preprocessing & Projection ---
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.fill import fillnodata
import numpy as np
import os
from pathlib import Path
import json

# Load Configuration from JSON
CONFIG_PATH = 'SDM/sdm_config.json'

with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

CLIMATE_DATA_DIR = config['CLIMATE_DATA_DIR']
OUTPUT_BASE_DIR = config['OUTPUT_BASE_DIR']
REF_ASCII_PATH = config['REF_ASCII_PATH']
TARGET_CRS = config['TARGET_CRS']
YEAR_COUNT = config['YEAR_COUNT']
SCENARIOS = config['SCENARIOS']
VARIABLES_MAP = config['VARIABLES_MAP']
print(f'Configuration loaded from {CONFIG_PATH}')




def process_cmip6_scenarios():
    print("Starting CMIP6 Data Processing...")
    if not os.path.exists(REF_ASCII_PATH):
        print(f"[ERROR] Reference file not found: {REF_ASCII_PATH}")
        return

    with rasterio.open(REF_ASCII_PATH) as ref_ds:
        dst_transform = ref_ds.transform
        dst_width = ref_ds.width
        dst_height = ref_ds.height
        dst_nodata = ref_ds.nodata or -9999
        dst_crs = TARGET_CRS
        ref_data = ref_ds.read(1)
        target_valid_mask = ~np.isclose(ref_data, dst_nodata)

    ascii_header = (
        f"ncols         {dst_width}\n"
        f"nrows         {dst_height}\n"
        f"xllcorner     {int(dst_transform.c)}\n"
        f"yllcorner     {int(dst_transform.f + (dst_height * dst_transform.e))}\n"
        f"cellsize      {int(dst_transform.a)}\n"
        f"NODATA_value  {int(dst_nodata)}\n"
    )

    for scenario in SCENARIOS:
        print(f"Processing Scenario: {scenario}")
        scenario_out_dir = os.path.join(OUTPUT_BASE_DIR, f"Input_Climate_{scenario}")
        os.makedirs(scenario_out_dir, exist_ok=True)

        for var_code, info in VARIABLES_MAP.items():
            filename = f"AR6_{scenario}_5ENSMN_skorea_{var_code}_gridraw_yearly_2021_2100.nc"
            file_path = os.path.join(CLIMATE_DATA_DIR, filename)

            if not os.path.exists(file_path):
                # Only warn if missing, dont spam
                pass 
                continue

            try:
                with rasterio.open(file_path) as src_ds:
                    limit_years = min(src_ds.count, YEAR_COUNT)
                    src_crs = src_ds.crs or rasterio.crs.CRS.from_string("EPSG:4326")

                    for i in range(limit_years):
                        dst_array = np.full((dst_height, dst_width), dst_nodata, dtype=np.float32)
                        src_data = src_ds.read(i+1)
                        src_nodata = src_ds.nodata

                        mask = (src_data != src_nodata).astype(np.uint8) if src_nodata is not None else np.ones_like(src_data, dtype=np.uint8)
                        filled = fillnodata(src_data, mask=mask, max_search_distance=100)

                        reproject(
                            source=filled, destination=dst_array,
                            src_transform=src_ds.transform, src_crs=src_crs, src_nodata=src_nodata,
                            dst_transform=dst_transform, dst_crs=dst_crs, dst_nodata=dst_nodata,
                            resampling=Resampling.bilinear
                        )
                        dst_array[~target_valid_mask] = dst_nodata

                        out_name = f"{info['prefix']}.{i}.asc"
                        with open(os.path.join(scenario_out_dir, out_name), 'w') as f:
                             f.write(ascii_header)
                             np.savetxt(f, dst_array, fmt='%.2f', delimiter=' ')
            except Exception as e:
                print(f"  [ERROR] {info['name']}: {e}")



def process_worldclim_geotiff_to_asc(input_tif, output_dir, ref_profile, target_valid_mask, scenario, time_period):
    '''Reproject and save WorldClim bands as ASC files.'''
    print(f"Processing WorldClim: {os.path.basename(input_tif)}...")
    
    with rasterio.open(input_tif) as src:
        for band_idx in range(1, src.count + 1):
            out_name = f"wc_{scenario}_{time_period}_bio{band_idx}.asc"
            out_path = os.path.join(output_dir, out_name)
            
            if os.path.exists(out_path):
                continue

            dst_array = np.full((ref_profile['height'], ref_profile['width']), ref_profile['nodata'], dtype=np.float32)
            
            reproject(
                source=rasterio.band(src, band_idx),
                destination=dst_array,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=ref_profile['transform'],
                dst_crs=ref_profile['crs'],
                dst_nodata=ref_profile['nodata'],
                resampling=Resampling.bilinear
            )
            
            dst_array[~target_valid_mask] = ref_profile['nodata']
            dst_array = np.round(dst_array, 2)
            
            header = (
                f"ncols         {ref_profile['width']}\n"
                f"nrows         {ref_profile['height']}\n"
                f"xllcorner     {int(ref_profile['transform'].c)}\n"
                f"yllcorner     {int(ref_profile['transform'].f + (ref_profile['height'] * ref_profile['transform'].e))}\n"
                f"cellsize      {int(ref_profile['transform'].a)}\n"
                f"NODATA_value  {int(ref_profile['nodata'])}\n"
            )
            
            with open(out_path, 'w') as f:
                f.write(header)
                np.savetxt(f, dst_array, fmt='%.2f', delimiter=' ')
                
    print(f"[DONE] Processed {os.path.basename(input_tif)}")

def process_worldclim_scenarios():
    print("Starting WorldClim Data Processing...")
    
    DOWNLOAD_DIR = "../Climate_Raw/WorldClim_Raw"
    
    if not os.path.exists(REF_ASCII_PATH):
        print(f"[ERROR] Reference file not found: {REF_ASCII_PATH}")
        return

    with rasterio.open(REF_ASCII_PATH) as ref_ds:
        ref_profile = ref_ds.profile.copy()
        ref_profile.update({'driver': 'AAIGrid', 'crs': TARGET_CRS})
        ref_data = ref_ds.read(1)
        ref_nodata = ref_ds.nodata or -9999
        ref_profile['nodata'] = ref_nodata
        target_valid_mask = ~np.isclose(ref_data, ref_nodata)
    
    WC_SCENARIOS = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    WC_MODEL = 'EC-Earth3-Veg'
    WC_VAR = 'bioc'
    WC_TIME_PERIODS = ['2021-2040', '2041-2060', '2061-2080']
    
    for ssp in WC_SCENARIOS:
        print(f"\n=== WorldClim Scenario: {ssp} ===")
        scenario_out_dir = os.path.join(OUTPUT_BASE_DIR, "Input_Climate_WorldClim", ssp)
        os.makedirs(scenario_out_dir, exist_ok=True)
        
        for time in WC_TIME_PERIODS:
            filename = f"wc2.1_30s_{WC_VAR}_{WC_MODEL}_{ssp}_{time}.tif"
            local_tif_path = os.path.join(DOWNLOAD_DIR, filename)
            
            if os.path.exists(local_tif_path):
                process_worldclim_geotiff_to_asc(local_tif_path, scenario_out_dir, ref_profile, target_valid_mask, ssp, time)
            else:
                print(f"[WARN] Source file not found: {local_tif_path}")


def main():
    
    # Uncomment to run preprocessing
    process_cmip6_scenarios()
    process_worldclim_scenarios()

if __name__ == "__main__":
    main()



