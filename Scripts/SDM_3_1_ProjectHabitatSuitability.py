
import os
import numpy as np
import pandas as pd
import rasterio
import joblib
import rioxarray
import xarray as xr
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
import geopandas as gpd
from rasterio.features import rasterize

import json
try:
    with open("../sdm_config.json", "r", encoding="utf-8") as _f:
        _cfg = json.load(_f)
except:
    _cfg = {}


try:
    import elapid
except ImportError:
    raise ImportError("elapid is required for MaxEnt but not installed.")

# --- Configuration ---

# Load SDM Config
with open(os.path.join(os.path.dirname(__file__), "../sdm_config.json"), 'r') as f:
    sdm_config = json.load(f)

# Load Species List from 'models' folder
models_dir = os.path.join(sdm_config['SDMMODEL_DIR'], 'Models')
SPECIES_LIST = []

# Load Northern Species
gpkg_path = _cfg.get("PLANT_GPKG_PATH", "../Data/Plant_Spatial_Data_05Feb2026.gpkg")
gdf = gpd.read_file(gpkg_path)
north_species_list = gdf[gdf['기후변화.취약식물'] == 1]['종명'].unique()
 
print(f"Scanning models directory: {models_dir}")
for fname in os.listdir(models_dir):
    if fname.endswith('.pkl') and fname.startswith("MaxEnt_baseline_"):
        # Format: MaxEnt_baseline_{species}.pkl
        prefix = "MaxEnt_baseline_"
        suffix = ".pkl"
        species_name = fname[len(prefix):-len(suffix)]
        SPECIES_LIST.append(species_name)
SPECIES_LIST = sorted(list(set(SPECIES_LIST)))
print(f"Loaded {len(SPECIES_LIST)} species from {models_dir}")

# Load Species Thermal Limits (Hard Constraints)
LIMITS_JSON_PATH = os.path.join(sdm_config.get('SDMMODEL_DIR', '../Output/SDM'), "ThermalLimits", "Species_ThermalLimits.json")
thermal_limits = {}
if os.path.exists(LIMITS_JSON_PATH):
    with open(LIMITS_JSON_PATH, 'r') as f:
        thermal_limits = json.load(f)
    print(f"Loaded GBIF thermal limits for {len(thermal_limits)} species.")
else:
    print(f"[WARN] Thermal limits file not found at {LIMITS_JSON_PATH}. Defaulting to 40.0 C.")




# FULL_SCENARIO_FNAMES = ["BAU-SSP126", "BAU-SSP245",
# "BAU-SSP370", "BAU-SSP585", "Climate-SSP126", "Biodiversity-SSP126"]
# CLIMATE_SCENARIO_NAMES = ["ssp126", "ssp245", "ssp370", "ssp585", "ssp126", "ssp126"]

FULL_SCENARIO_FNAMES = ["BAU-SSP585", "Climate-SSP126", "Biodiversity-SSP126", "Biodiversity-SSP245"]
CLIMATE_SCENARIO_NAMES = ["ssp585", "ssp126", "ssp126", "ssp245"]


SIDO_SHP_PATH = _cfg.get("SIDO_SHP_PATH", "../GIS_Data/BND_SIDO_PG_TM.shp")
SIDO_shp  = gpd.read_file(SIDO_SHP_PATH)

# # --- PROCESS SIDO POLYGONS (Rasterize for Overlay) ---
# try:
#     # User defined SIDO_shp global at top. Ensure alignment.
#     if SIDO_shp.crs != ref_raster.rio.crs:
#         sido_gdf_proj = SIDO_shp.to_crs(ref_raster.rio.crs)
#     else:
#         sido_gdf_proj = SIDO_shp
        
#     # Rasterize boundaries (value=1)
#     shapes = [(geom, 1) for geom in sido_gdf_proj.boundary] 
    
#     si_do_polygons = rasterize(
#         shapes=shapes,
#         out_shape=ref_raster.shape,
#         transform=ref_raster.rio.transform(),
#         fill=0,
#         default_value=1,
#         dtype=np.uint8
#     )
    
#     # Convert to float and set background (0) to NaN for transparency
#     si_do_polygons = si_do_polygons.astype(float)
#     si_do_polygons[si_do_polygons == 0] = np.nan
    
# except Exception as e:
#     print(f"[WARN] Failed to process SIDO polygons: {e}")
#     si_do_polygons = np.full(ref_raster.shape, np.nan)



LULC_RUN_BASE_DIR = _cfg.get("LULC_RUN_BASE_DIR", "../LULC_Scenarios/")



BASE_YEAR = 2021 # CMIP6 starts 2021
BASE_YEAR_NOMINAL = BASE_YEAR - 1 #
TARGET_YEAR = 2050

# Input Directories
MODEL_INPUT_DIR = _cfg.get("OUTPUT_BASE_DIR", "../Model/") 



# 시각화 설정
# plt.rcParams['font.family'] = 'Malgun Gothic' # Windows의 경우
plt.rcParams['font.family'] = 'AppleGothic' # Mac의 경우
plt.rcParams['axes.unicode_minus'] = False



# SCENARIO_IDX = 1

# CLIMATE_SCENARIO = CLIMATE_SCENARIO_NAMES[SCENARIO_IDX]
# SCENARIO_FNAME = FULL_SCENARIO_FNAMES[SCENARIO_IDX]

# print(SCENARIO_FNAME)
# LULC_RUN_DIR = f"{LULC_RUN_BASE_DIR}/{SCENARIO_FNAME}/"
# OUTPUT_DIR = f"../Output/SDM/Projection_{SCENARIO_FNAME}_{TARGET_YEAR}"






# --- Helper Functions ---
def load_raster(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    # Use nearest neighbor resampling to avoid interpolation artifacts at coastlines
    from rasterio.enums import Resampling
    return rioxarray.open_rasterio(path, masked=True, resampling=Resampling.nearest).squeeze(drop=True)


def _save_float32_raster(array: np.ndarray, out_path, ref_raster):
    """Write a 2-D float32 array to a GeoTIFF using ref_raster for CRS/transform."""
    crs = ref_raster.rio.crs or sdm_config.get('TARGET_CRS', 'EPSG:5179')
    transform = ref_raster.rio.transform()
    rows, cols = array.shape
    with rasterio.open(
        str(out_path), 'w', driver='GTiff',
        height=rows, width=cols, count=1,
        dtype=np.float32, crs=crs,
        transform=transform, nodata=np.nan
    ) as dst:
        dst.write(array.astype(np.float32), 1)

def predict_year(model, species_name, year, yr_idx, clim_kma_dir, clim_bio_dir, lulc_dir, run_mode):
    """Predicts habitat suitability for a specific year using custom paths."""
    print(f"  -> Predicting for {year}...")
    
    
    # --- Load Predictors ---
    
    # 1. Topography and Soil
    elev = load_raster(Path(MODEL_INPUT_DIR) / "sc1gr0.0.asc")
    slope = load_raster(Path(MODEL_INPUT_DIR) / "sc1gr1.0.asc")
    soil_deep = load_raster(Path(MODEL_INPUT_DIR) / "sc1gr2.0.asc")
    soil_dra = load_raster(Path(MODEL_INPUT_DIR) / "sc1gr3.0.asc")

    # 2. LULC (Moved up for masking)
    lulc_path = Path(lulc_dir) / f"cov_all.{yr_idx}.asc"
    if not lulc_path.exists():
         # Fallback to base in scenario dir
         lulc_path = Path(lulc_dir) / "cov_all.0.asc"
    lulc = load_raster(lulc_path)
    
    # 3. Create Valid Mask (Elevation & LULC must be valid)
    # Assuming -9000 is nodata threshold for Topo, and -1 or -9999 for LULC
    base_mask = (elev.values > -9000) & (lulc.values > -1)
    
    # 4. Create Coastal Edge Mask to filter interpolation artifacts
    # Identify edge pixels (land pixels adjacent to ocean/nodata)
    from scipy.ndimage import binary_erosion
    
    # Erode the land mask by 2 pixels to identify edge pixels
    # Pixels that are land but disappear after erosion are edge pixels
    eroded_mask = binary_erosion(base_mask, iterations=3)
    edge_pixels = base_mask & ~eroded_mask
    
    # Final mask: valid land pixels excluding coastal edges
    mask = base_mask & ~edge_pixels
    
    rows, cols = elev.shape


    # Load environmental layers based on config
    env_config = sdm_config.get('ENV_LAYERS', {})
    bioclim_names_map = sdm_config.get('BIOCLIM_NAMES', {})
    
    # Load bioclim variables specified in config
    bioclims = {}
    for bio_num in env_config.get('bioclim', []):
        # For baseline year, use the baseline scenario from config to match training data
        # For future years, use the selected scenario
        if year <= BASE_YEAR_NOMINAL:
            baseline_scenario = sdm_config.get('BASELINE_SCENARIO', 'ssp126')
            baseline_year = sdm_config.get('BASELINE_YEAR', 2021)
            scen_name = baseline_scenario
            year_bioclim = baseline_year
        else:
            scen_name = Path(clim_bio_dir).name
            year_bioclim = year

        fname = f"kma_{scen_name}_{year_bioclim}_bio{bio_num}.asc"
        
        # Construct path using baseline scenario directory for baseline year
        if year <= BASE_YEAR_NOMINAL:
            fpath = Path(clim_bio_dir).parent / baseline_scenario / fname
        else:
            fpath = Path(clim_bio_dir) / fname
            
        if not fpath.exists():
            print(f"  [WARN] Bioclim file not found: {fname}")
        else:
            bioclims[f"bioclim{bio_num}"] = load_raster(fpath)

    # Force load bio5 explicitly for the Hard Constraint if not in env config
    if run_mode == "Constrained":
        if 5 not in env_config.get('bioclim', []):
            if year <= BASE_YEAR_NOMINAL:
                b5_path = Path(clim_bio_dir).parent / sdm_config.get('BASELINE_SCENARIO', 'ssp126') / f"kma_{sdm_config.get('BASELINE_SCENARIO', 'ssp126')}_{sdm_config.get('BASELINE_YEAR', 2021)}_bio5.asc"
            else:
                b5_path = Path(clim_bio_dir) / f"kma_{Path(clim_bio_dir).name}_{year}_bio5.asc"
                
            if b5_path.exists():
                bioclims["bioclim5_force"] = load_raster(b5_path)
            else:
                print(f"  [WARN] bioclim5 hard constraint raster missing: {b5_path}")
                
        if 6 not in env_config.get('bioclim', []):
            if year <= BASE_YEAR_NOMINAL:
                b6_path = Path(clim_bio_dir).parent / sdm_config.get('BASELINE_SCENARIO', 'ssp126') / f"kma_{sdm_config.get('BASELINE_SCENARIO', 'ssp126')}_{sdm_config.get('BASELINE_YEAR', 2021)}_bio6.asc"
            else:
                b6_path = Path(clim_bio_dir) / f"kma_{Path(clim_bio_dir).name}_{year}_bio6.asc"
                
            if b6_path.exists():
                bioclims["bioclim6_force"] = load_raster(b6_path)
            else:
                print(f"  [WARN] bioclim6 hard constraint raster missing: {b6_path}")

    # Load SI and RH - use baseline scenario for baseline year
    if year <= BASE_YEAR_NOMINAL:
        baseline_scenario = sdm_config.get('BASELINE_SCENARIO', 'ssp126')
        baseline_clim_dir = Path(clim_kma_dir).parent / baseline_scenario
        si_path = baseline_clim_dir / f"sc1gr_si.{yr_idx}.asc"
        rh_path = baseline_clim_dir / f"sc1gr_rhm.{yr_idx}.asc"
        
        if not si_path.exists(): si_path = baseline_clim_dir / "sc1gr_si.0.asc"
        if not rh_path.exists(): rh_path = baseline_clim_dir / "sc1gr_rhm.0.asc"
    else:
        si_path = Path(clim_kma_dir) / f"sc1gr_si.{yr_idx}.asc"
        rh_path = Path(clim_kma_dir) / f"sc1gr_rhm.{yr_idx}.asc"
        
        if not si_path.exists(): si_path = Path(clim_kma_dir) / "sc1gr_si.0.asc"
        if not rh_path.exists(): rh_path = Path(clim_kma_dir) / "sc1gr_rhm.0.asc"
    
    si = load_raster(si_path)
    rh = load_raster(rh_path)

    print(f"  DEBUG: LULC stats (masked): min={lulc.values[mask].min()}, max={lulc.values[mask].max()}")

    # Build feature dictionary dynamically from config
    data = {}
    
    # Topography
    for var in env_config.get('topography', []):
        if var == 'elevation':
            data['elevation'] = elev.values[mask]
        elif var == 'slope':
            data['slope'] = slope.values[mask]
    
    # Soil
    for var in env_config.get('soil', []):
        if var == 'soildeep3':
            data['soildeep3'] = soil_deep.values[mask]
        elif var == 'soildra1':
            data['soildra1'] = soil_dra.values[mask]
    
    # Bioclim
    for bio_num in env_config.get('bioclim', []):
        bio_key = f'bioclim{bio_num}'
        if bio_key in bioclims:
            data[bio_key] = bioclims[bio_key].values[mask]
    
    # Climate
    for var in env_config.get('climate', []):
        if var == 'solarradiation':
            data['solarradiation'] = si.values[mask]
        elif var == 'relative_humidity':
            data['relative_humidity'] = rh.values[mask]
        elif var == 'VPD':
            # Calculate VPD dynamically if required by config
            temp_c_key = 'bioclim1'
            if temp_c_key not in bioclims:
                 # Fallback to another temp if bioclim1 not present, else assume roughly Bio5 or similar
                 temp_c_key = next((k for k in bioclims.keys() if k in ['bioclim5', 'bioclim6', 'bioclim10']), None)
            
            if temp_c_key:
                temp_c = bioclims[temp_c_key].values[mask]
                rh_vals = rh.values[mask]
                
                vpd = np.full_like(temp_c, fill_value=np.nan, dtype=np.float32)
                valid_mask_vpd = (temp_c > -500) & (rh_vals > -500)
                
                # Formula
                e_s = 6.112 * np.exp((17.67 * temp_c[valid_mask_vpd]) / (temp_c[valid_mask_vpd] + 243.5))
                e_a = e_s * (rh_vals[valid_mask_vpd] / 100.0)
                vpd[valid_mask_vpd] = e_s - e_a
                
                vpd[np.isnan(vpd)] = -9999
                data['VPD'] = vpd

    # Land cover
    for var in env_config.get('landcover', []):
        if var == 'LULC':
            data['LULC'] = lulc.values[mask]
            
    X_pred = pd.DataFrame(data)
    if X_pred.isnull().values.any():
        X_pred = X_pred.fillna(0)

    # Ensure LULC is treated as categorical with fixed categories
    lulc_categories = [int(c) for c in sdm_config.get('LULC_CATEGORIES')]
    lulc_dtype = pd.CategoricalDtype(categories=lulc_categories, ordered=False)

    if 'LULC' in X_pred.columns:
        X_pred['LULC'] = X_pred['LULC'].astype(int).astype(lulc_dtype)
    
    # Identify rows with NaN in LULC (invalid/unknown categories)
    # This happens if LULC value was not in [0-6]
    valid_pred_mask = X_pred['LULC'].notna()

    
    # Initialize y_prob with NaNs
    y_prob = np.full(len(X_pred), np.nan)
    
    if valid_pred_mask.any():
        # Predict only on valid rows
        X_valid = X_pred[valid_pred_mask]
        
        # # Try to predict, but handle unknown LULC categories
        # try:
        #     # Elapid MaxEnt uses dtypes for categorical features, no kwarg needed for predict_proba
        #     probs = model.predict_proba(X_valid)[:, 1]
        #     y_prob[valid_pred_mask] = probs
        # except ValueError as e:
            # If we encounter unknown categories, extract which ones are known from the model
            # and filter X_valid to only include those
            # if "unknown categories" in str(e).lower():
        print(f"  [INFO] Filtering pixels with unknown LULC categories...")
        # try:
        # Extract known categories from the trained model
        known_cats = model.transformer.estimators_['categorical'].estimators_[0].categories_[0]
        
        # Filter to only pixels with known LULC categories
        lulc_in_known = X_valid['LULC'].isin(known_cats)
        n_filtered = (~lulc_in_known).sum()
        
        if n_filtered > 0:
            print(f"  [INFO] Filtered out {n_filtered} pixels with unseen LULC categories (set to 0 suitability)")
        
        X_valid_filtered = X_valid[lulc_in_known]
        
        # Set unseen LULC pixels to 0 suitability (species doesn't use those habitats)
        valid_indices = np.where(valid_pred_mask)[0]
        unseen_indices = valid_indices[~lulc_in_known]
        y_prob[unseen_indices] = 0.0
        
        if len(X_valid_filtered) > 0:
            # --- Z-TRANSFORM (Standardization) ---
            apply_z_transform = sdm_config.get('APPLY_Z_TRANSFORM', False)
            if apply_z_transform:
                # Load the corresponding scaler used during training
                SDMMODEL_DIR = sdm_config.get('SDMMODEL_DIR', '../Output/SDM')
                scaler_path = f'{SDMMODEL_DIR}/Models/Scaler_baseline_{species_name}.pkl'
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)
                    continuous_cols = [col for col in X_valid_filtered.columns if col != 'LULC']
                    
                    if continuous_cols:
                        print(f"  [INFO] Applying Z-transform from training scaler to {len(continuous_cols)} variables.")
                        
                        # Ensure columns match scaler order dynamically just in case
                        try:
                             # Ensure ordering matches scaler exactly (it expects same order as fit)
                             # We apply transform without triggering SettingWithCopyWarning
                             X_valid_filtered_scaled = X_valid_filtered.copy()
                             X_valid_filtered_scaled[continuous_cols] = scaler.transform(X_valid_filtered_scaled[continuous_cols])
                             X_valid_filtered = X_valid_filtered_scaled
                        except ValueError as e:
                             print(f"  [WARN] Scaler failed to transform (feature order mismatch?). Falling back... Error: {e}")
                else:
                     print(f"  [INFO] No scaler found at {scaler_path}. Proceeding with raw variables.")
            else:
                 print(f"  [INFO] Skipping Z-transform (APPLY_Z_TRANSFORM config is False).")
                 
            # Predict on filtered (and scaled) data
            probs = model.predict_proba(X_valid_filtered)[:, 1]
            
            # Application of Extreme Temperature Mask
            # Based on combined GBIF + Local species-specific survival limits.
            has_bio5 = ('bioclim5' in bioclims or 'bioclim5_force' in bioclims) and sdm_config.get('APPLY_HOT_CONSTRAINT', True)
            has_bio6 = ('bioclim6' in bioclims or 'bioclim6_force' in bioclims) and sdm_config.get('APPLY_COLD_CONSTRAINT', True)
            if run_mode == "Constrained":
                hot_mask = np.zeros_like(probs, dtype=bool)
                cold_mask = np.zeros_like(probs, dtype=bool)
                
                sp_limit_hot = "DISABLED"
                sp_limit_cold = "DISABLED"
                
                # Hot Margin (bio5)
                if has_bio5:
                    sp_limit_hot = 40.0
                    if species_name in thermal_limits:
                        # Use GBIF limit if available, otherwise local limit (as requested by user)
                        local_val = thermal_limits[species_name].get("local_bio5_max")
                        gbif_val = thermal_limits[species_name].get("gbif_bio5_max")
                        
                        if gbif_val is not None:
                            val = gbif_val
                        else:
                            val = local_val
                        # Fallback to old format if necessary
                        if val is None: val = thermal_limits[species_name].get("bio5_max", 40.0)
                        sp_limit_hot = float(val)
                    
                    bio5_raster = bioclims.get('bioclim5', bioclims.get('bioclim5_force'))
                    b5_valid_filtered = bio5_raster.values[mask][valid_indices[lulc_in_known]]
                    hot_mask = b5_valid_filtered >= sp_limit_hot
                    
                # Cold Margin (bio6)
                if has_bio6:
                    sp_limit_cold = -40.0
                    if species_name in thermal_limits:
                        val = thermal_limits[species_name].get("combined_bio6_min")
                        if val is None: val = thermal_limits[species_name].get("bio6_min", -40.0)
                        sp_limit_cold = float(val)
                    
                    bio6_raster = bioclims.get('bioclim6', bioclims.get('bioclim6_force'))
                    b6_valid_filtered = bio6_raster.values[mask][valid_indices[lulc_in_known]]
                    cold_mask = b6_valid_filtered <= sp_limit_cold
                    
                total_mask = hot_mask | cold_mask
                n_filtered = np.sum(total_mask)
                if n_filtered > 0:
                    n_hot = np.sum(hot_mask)
                    n_cold = np.sum(cold_mask)
                    print(f"  [INFO] [{run_mode}] Filtered {n_filtered} pixels (Hot >= {sp_limit_hot}C: {n_hot}, Cold <= {sp_limit_cold}C: {n_cold})")
                    probs[total_mask] = 0.0
            
            # Map predictions to pixels with known LULC categories
            filtered_indices = valid_indices[lulc_in_known]
            y_prob[filtered_indices] = probs
    #                 else:
    #                     print("  [WARN] No pixels remaining after filtering unknown LULC categories")
    #             except Exception as inner_e:
    #                 print(f"  [ERROR] Failed to filter unknown categories: {inner_e}")
    #                 raise
    #         else:
    #             # Re-raise if it's a different ValueError
    #             raise

    else:
        print("  [WARN] No valid LULC pixels found for prediction in this chunk.")


    
    out_grid = np.full((rows, cols), np.nan, dtype=np.float32)
    flat_mask = mask.flatten()
    valid_indices = np.where(flat_mask)[0]
    out_flat = out_grid.flatten()
    out_flat[valid_indices] = y_prob
    out_grid = out_flat.reshape(rows, cols)
    
    return out_grid, elev, mask

def process_scenario(species_name, scenario_idx, climate_scenario_name, full_scenario_name, run_mode):
    print(f"\n==================================================")
    print(f"Projecting {species_name} | [{run_mode}] Scenario {scenario_idx+1}/{len(CLIMATE_SCENARIO_NAMES)}: {climate_scenario_name}")
    print(f"==================================================")

    future_output_dir = f"../Output/SDM/Projection_{full_scenario_name}_{TARGET_YEAR}_{run_mode}"
    os.makedirs(future_output_dir, exist_ok=True)
    raster_dir = Path(future_output_dir) / "Rasters"
    raster_dir.mkdir(parents=True, exist_ok=True)
    
    lulc_run_dir = f"{LULC_RUN_BASE_DIR}/{full_scenario_name}/"
    climate_kma_dir = f"{_cfg.get('OUTPUT_BASE_DIR', '../Model/')}Input_Climate_KMA/{climate_scenario_name}"
    climate_bioclim_dir = f"{_cfg.get('OUTPUT_BASE_DIR', '../Model/')}Input_Bioclim_KMA/{climate_scenario_name}"
    
    # Load model
    SDMMODEL_DIR = sdm_config.get('SDMMODEL_DIR', '../Output/SDM')
    model_path = f"{SDMMODEL_DIR}/Models/MaxEnt_baseline_{species_name}.pkl"
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return None
    model = joblib.load(model_path)
    
    base_scenario = CLIMATE_SCENARIO_NAMES[0]
    base_kma_dir = f"{_cfg.get('OUTPUT_BASE_DIR', '../Model/')}Input_Climate_KMA/{base_scenario}"
    base_bio_dir = f"{_cfg.get('OUTPUT_BASE_DIR', '../Model/')}Input_Bioclim_KMA/{base_scenario}"
    
    hq_base, ref_raster_base, mask_base = predict_year(model, species_name, BASE_YEAR_NOMINAL, yr_idx=0,
                                     clim_kma_dir=base_kma_dir, clim_bio_dir=base_bio_dir, lulc_dir=lulc_run_dir, run_mode=run_mode)
                                     
    hq_future, ref_raster_future, mask_future = predict_year(model, species_name, TARGET_YEAR, yr_idx=30,
                                       clim_kma_dir=climate_kma_dir, clim_bio_dir=climate_bioclim_dir, lulc_dir=lulc_run_dir, run_mode=run_mode)
                                       
    # Mask future with baseline mask to avoid coastal artifacts and no-data areas
    hq_future[~mask_base] = np.nan
                                       
    out_base = raster_dir / f"HS_Base_{species_name}.tif"
    out_future = raster_dir / f"HS_Future_{species_name}.tif"
    
    _save_float32_raster(hq_base.astype(np.float32), out_base, ref_raster_base)
    _save_float32_raster(hq_future.astype(np.float32), out_future, ref_raster_future)
    
    print(f"  Saved raw suitabilities to {raster_dir}")
    return True

def run_all_scenarios():
    print(f"Total Species to Process: {len(SPECIES_LIST)}")
    for run_mode in ["Constrained", "Unconstrained"]:
        print(f"\n{'#'*60}")
        print(f"### RUNNING PROJECTIONS IN MODE: {run_mode.upper()}")
        print(f"{'#'*60}")
        for sp_name in SPECIES_LIST:
            print(f"*** Processing Species: {sp_name} [{run_mode}] ***")
            for idx in range(len(FULL_SCENARIO_FNAMES)): 
                s_name = CLIMATE_SCENARIO_NAMES[idx]
                s_fname = FULL_SCENARIO_FNAMES[idx]
                process_scenario(sp_name, idx, s_name, s_fname, run_mode)

if __name__ == "__main__":
    run_all_scenarios()
