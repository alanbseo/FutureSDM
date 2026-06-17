# %%
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path
import random
from shapely.geometry import Point
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
import rioxarray
import xarray as xr
import json
import joblib
from SDM_functions import extract_values_at_points

import json
try:
    with open("../SDM_config.json", "r", encoding="utf-8") as _f:
        _cfg = json.load(_f)
except:
    _cfg = {}


# --- CONFIGURATION FROM JSON ---
config_path = os.path.join('../SDM_config.json')
# config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../SDM_config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

# Optional: Elapid import for MaxEnt
try:
    import elapid
except ImportError:
    elapid = None

# 시각화 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# --- LOAD SPECIES DATA (old)---
# ENDANGERED_SPECIES_PATH = "../Data/멸종위기종목록_UTF8 환경부멸종위기종.csv" 
# SPECIES_DATA_PATH = "../Data/3차 전국자연환경조사/3차_식물상.shp" 
# INDICATOR_SPECIES_PATH = "../Data/Climate-sensitive Biological Indicator Species.xlsx"

# endangered_species_df = pd.read_csv(ENDANGERED_SPECIES_PATH)
# # endangered_species_df = endangered_species_df[endangered_species_df["분류군명"] == "육상식물"]

# plant_3rd = gpd.read_file(SPECIES_DATA_PATH)
# endangered_plant_3rd = plant_3rd[plant_3rd['종_국명'].isin(endangered_species_df['국명']) | plant_3rd['종_학명'].isin(endangered_species_df['학명'])]

# print(f"Endangered plant species data loaded: {len(endangered_plant_3rd)} records")

# %%

# load spatial data
plant_spatial_data = gpd.read_file(_cfg.get("PLANT_GPKG_PATH", "../Data/Plant_Spatial_Data_05Feb2026.gpkg"))


# %%
# what are the column names?
print(plant_spatial_data.columns)
print(plant_spatial_data.info())


print(plant_spatial_data['기후변화.취약식물'].value_counts())
print(plant_spatial_data['멸종위기 등급'].value_counts())

 
# %%

target_species1 = plant_spatial_data.loc[
    plant_spatial_data['기후변화.취약식물'] == 1,
    '종명'
].unique()

print(target_species1)

target_species2 = plant_spatial_data.loc[
    plant_spatial_data['기후변화.취약식물'] == 2,
    '종명'
].unique()

target_species3 = plant_spatial_data.loc[
    plant_spatial_data['기후변화.취약식물'] == 3,
    '종명'
].unique()


target_species4 = plant_spatial_data.loc[
    plant_spatial_data['멸종위기 등급'] == '1',
    '종명'
].unique()

target_species5 = plant_spatial_data.loc[
    plant_spatial_data['멸종위기 등급'] == '2',
    '종명'
].unique()


print("target_species1:", target_species1)
print("target_species2:", target_species2)
print("target_species3:", target_species3)
print("target_species4:", target_species4)
print("target_species5:", target_species5)


# %%


# combine target_species1 and target_species2
combined_target_species = np.unique(
    np.concatenate([
        target_species1,
        target_species2,
        target_species3,
        target_species4,
        target_species5
    ])
)

print("combined_target_species:", len(combined_target_species))
combined_target_species_df = pd.DataFrame(combined_target_species, columns=['Species_Name'])

 
# output_csv_path = os.path.join("Output", "SDM", "target_plant_species_6Feb2026.csv")

# Ensure directory exists
#os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

# combined_target_species_df.to_csv(output_csv_path, index=False)
# print(f"Saved target species list to {output_csv_path}")


# %%

# check how many records for each species
# create a new list to store the number of records for each species

species_records_df = (
    plant_spatial_data
    .query("종명 in @combined_target_species")
    .groupby('종명')
    .size()
    .reset_index(name='count')
    .rename(columns={'종명': 'Species_Name'})
)

# species_records_df = species_records_df[
#     species_records_df['Species_Name'].isin(
#         combined_target_species_df['Species_Name']
#     )
# ]
species_records_df.shape

# %%
print(species_records_df)

species_records_gt10 = species_records_df.query("count > 10")
print(species_records_gt10)

species_records_gt10.info()
species_records_gt10.shape
# %%
# %%
# save to csv
out_dir = os.path.join("..", "Output", "SDM")
os.makedirs(out_dir, exist_ok=True)
species_records_df.to_csv(os.path.join(out_dir, "species_records.csv"), index=False)
print(f"Saved species records to {os.path.join(out_dir, 'species_records.csv')}")





# %%

# # Export Species List for SDM_3
# species_list_path = os.path.join(config['SDMMODEL_DIR'], 'target_species_list.json')
# os.makedirs(config['SDMMODEL_DIR'], exist_ok=True)
# with open(species_list_path, 'w') as f:
#     json.dump(combined_target_species.tolist(), f)
# print(f"Exported target species list to {species_list_path}")


# and tabulate how many came from climnatesensitve and how many other from endangered and see if there are any duplicates
# print("ClimateSensitive:", len(target_species1['Species_Name'].unique()))
# print("Endangered:", len(target_species2['Species_Name'].unique()))
# print("Duplicates:", len(combined_target_species['Species_Name'].unique()) - len(target_species1['Species_Name'].unique()) - len(target_species2['Species_Name'].unique()))

# print("combined_target_species after removing duplicates:", combined_target_species)

# %%
# --- LOAD ENVIRONMENTAL DATA ---
# Define Project Root (One level up from SDM folder)
PRJ_ROOT = ".."

BIOCLIM_DIR = os.path.join(PRJ_ROOT, "Model", "Input_Bioclim_KMA")
MODELINPUT_DIR = os.path.join(PRJ_ROOT, "Model")
CLIMATE_DIR = os.path.join(PRJ_ROOT, "Model", "Input_Climate_KMA")
scenario_name = "ssp126"

# Load Bioclim (2021)
bioclim_dir = os.path.join(BIOCLIM_DIR, scenario_name)
bioclim_names = config['BIOCLIM_NAMES']
bioclim_list = []
sorted_bio_keys = sorted(bioclim_names.keys(), key=lambda x: int(x.replace('bio', '')))

print("Loading Bioclim...")
for bio_key in sorted_bio_keys:
    fpath = os.path.join(bioclim_dir, f'kma_{scenario_name}_2021_{bio_key}.asc')
    da = rioxarray.open_rasterio(fpath, masked=True).squeeze("band", drop=True)
    da_bio = da.assign_coords(bio=bioclim_names[bio_key]).expand_dims("bio")
    bioclim_list.append(da_bio)

bioclim = xr.concat(bioclim_list, dim="bio")

# Load Other Vars (2020/2021 Baseline)
SI_2020 = rioxarray.open_rasterio(Path(CLIMATE_DIR) / f"{scenario_name}/sc1gr_si.0.asc", masked=True).squeeze(drop=True)
RH_2020 = rioxarray.open_rasterio(Path(CLIMATE_DIR) / f"{scenario_name}/sc1gr_rhm.0.asc", masked=True).squeeze(drop=True)

elevation_2020 = rioxarray.open_rasterio(Path(MODELINPUT_DIR) / "sc1gr0.0.asc", masked=True).squeeze(drop=True)
slope_2020 = rioxarray.open_rasterio(Path(MODELINPUT_DIR) / "sc1gr1.0.asc", masked=True).squeeze(drop=True)
soildeep3_2020 = rioxarray.open_rasterio(Path(MODELINPUT_DIR) / "sc1gr2.0.asc", masked=True).squeeze(drop=True)
soildra1_2020 = rioxarray.open_rasterio(Path(MODELINPUT_DIR) / "sc1gr3.0.asc", masked=True).squeeze(drop=True)
soilstone1_2020 = rioxarray.open_rasterio(Path(MODELINPUT_DIR) / "sc1gr4.0.asc", masked=True).squeeze(drop=True)
LULC_2020 = rioxarray.open_rasterio(Path(MODELINPUT_DIR) / "cov_all.0.asc", masked=True).squeeze(drop=True)

# --- PREPARE MODEL DATA ---
# target_species = '산분꽃나무'
# presence_gdf = endangered_plant_3rd[endangered_plant_3rd['종_국명'] == target_species].copy()
# print(f'Selected Species: {target_species}, Presence records: {len(presence_gdf)}')

# Env Dictionary
# env_layers = {
#     'elevation': elevation_2020.values,
#     'slope': slope_2020.values,
#     'soildeep3': soildeep3_2020.values,
#     'soildra1': soildra1_2020.values,
#     'soilstone1': soilstone1_2020.values,
#     'bioclim1': bioclim.sel(bio='Annual Mean Temperature').values,
#     'bioclim2': bioclim.sel(bio='Mean Diurnal Range').values,
#     'bioclim3': bioclim.sel(bio='Isothermality').values,
#     'bioclim4': bioclim.sel(bio='Temperature Seasonality').values,
#     'bioclim5': bioclim.sel(bio='Max Temperature of Warmest Month').values,
#     'bioclim6': bioclim.sel(bio='Min Temperature of Coldest Month').values,
#     'bioclim7': bioclim.sel(bio='Temperature Annual Range').values,
#     'bioclim8': bioclim.sel(bio='Mean Temperature of Wettest Quarter').values,
#     'bioclim9': bioclim.sel(bio='Mean Temperature of Driest Quarter').values,
#     'bioclim10': bioclim.sel(bio='Mean Temperature of Warmest Quarter').values, # Newly added
#     'bioclim11': bioclim.sel(bio='Mean Temperature of Coldest Quarter').values,
#     'bioclim12': bioclim.sel(bio='Annual Precipitation').values,
#     'bioclim13': bioclim.sel(bio='Precipitation of Wettest Month').values,
#     'bioclim14': bioclim.sel(bio='Precipitation of Driest Month').values,
#     'bioclim15': bioclim.sel(bio='Precipitation Seasonality').values,
#     'bioclim16': bioclim.sel(bio='Precipitation of Wettest Quarter').values,
#     'bioclim17': bioclim.sel(bio='Precipitation of Driest Quarter').values,
#     'bioclim18': bioclim.sel(bio='Precipitation of Warmest Quarter').values,
#     'bioclim19': bioclim.sel(bio='Precipitation of Coldest Quarter').values,
#     "solarradiation": SI_2020.values,
#     "relative_humidity": RH_2020.values,
#     'LULC': LULC_2020.values
# }

# # Add VPD logically here if the user wanted it in the old hardcoded block
# # (The real configuration builds the map dynamically below)
# # Calculate VPD from SI and RH if available
# # This is a placeholder for the actual calculation logic
# # For example, if you have temperature data (e.g., from bioclim1)
# # es = 0.6108 * np.exp((17.27 * temp) / (temp + 237.3)) # Saturation vapor pressure
# # ea = es * (RH_2020.values / 100) # Actual vapor pressure
# # VPD = es - ea
# # env_layers['VPD'] = VPD
# # }




# Build env_layers from config
env_config = config.get('ENV_LAYERS', {})
env_layers = {}

# Topography
for var in env_config.get('topography', []):
    if var == 'elevation':
        env_layers['elevation'] = elevation_2020.values
    elif var == 'slope':
        env_layers['slope'] = slope_2020.values

# Soil
for var in env_config.get('soil', []):
    if var == 'soildeep3':
        env_layers['soildeep3'] = soildeep3_2020.values
    elif var == 'soildra1':
        env_layers['soildra1'] = soildra1_2020.values
    elif var == 'soilstone1':
        env_layers['soilstone1'] = soilstone1_2020.values

# Bioclim
bioclim_names_map = config.get('BIOCLIM_NAMES', {})
for bio_num in env_config.get('bioclim', []):
    bio_key = f'bio{bio_num}'
    if bio_key in bioclim_names_map:
        env_layers[f'bioclim{bio_num}'] = bioclim.sel(bio=bioclim_names_map[bio_key]).values

# Climate
for var in env_config.get('climate', []):
    if var == 'solarradiation':
        env_layers['solarradiation'] = SI_2020.values
    elif var == 'relative_humidity':
        env_layers['relative_humidity'] = RH_2020.values

# Land cover
for var in env_config.get('landcover', []):
    if var == 'LULC':
        env_layers['LULC'] = LULC_2020.values

print(f"Loaded {len(env_layers)} environmental layers from config: {list(env_layers.keys())}")

# Calculate VPD dynamically if it was requested in the config
if 'VPD' in env_config.get('climate', []):
    # Try to find a temperature layer to use (prefer Bio1, then fallback to others)
    temp_key = None
    if 'bioclim1' in env_layers:
        temp_key = 'bioclim1'
    else:
        # Fallback to Bio5, 10, or 6 if Bio1 isn't loaded
        for fallback in ['bioclim5', 'bioclim10', 'bioclim6']:
            if fallback in env_layers:
                temp_key = fallback
                break
                
    if temp_key:
        temp_c = env_layers[temp_key]
        rh = RH_2020.values
        
        # Check if there are invalid values (-9999) and handle them
        valid_mask = (temp_c > -500) & (rh > -500) # Since some are -9999
        
        vpd = np.full_like(temp_c, fill_value=np.nan, dtype=np.float32)
        
        # Calculate Saturated Vapor Pressure (e_s) in hPa
        e_s = 6.112 * np.exp((17.67 * temp_c[valid_mask]) / (temp_c[valid_mask] + 243.5))
        
        # Calculate Actual Vapor Pressure (e_a)
        e_a = e_s * (rh[valid_mask] / 100.0)
        
        # VPD in hPa
        vpd[valid_mask] = e_s - e_a
        
        # Assign back 
        # Fill nan with -9999 for consistency with other layers
        vpd[np.isnan(vpd)] = -9999
        env_layers['VPD'] = vpd
        
        print(f"Computed VPD logically from {temp_key} and Relative Humidity.")
    else:
        raise ValueError("[WARN] VPD was requested in config, but Temperature layers are missing!")

# Raster Info
with rasterio.open(Path(MODELINPUT_DIR) / "sc1gr0.0.asc") as src:
    transform = src.transform
    rst_shape = src.shape
    crs = src.crs

if crs is None:
    print(f"Warning: Raster has no CRS. Using TARGET_CRS from config: {config['TARGET_CRS']}")
    crs = config['TARGET_CRS']




# %%

# okay, do the loop for sp_idx from 0 to len(combined_target_species)
# can you create a function (or procedure doing the following steps? then do the loop



def process_species_sdm(target_species, plant_spatial_data, env_layers, src, transform, rst_shape, crs, config, elapid=None):
    """
    Process a single species for Species Distribution Modeling (SDM).
    
    Args:
        target_species (str): Name of the species to process.
        plant_spatial_data (GeoDataFrame): Spatial data containing species occurrences.
        env_layers (dict): Dictionary of environmental raster layers (values).
        src (rasterio.DatasetReader or object): Rasterio source object (for bounds).
        transform (Affine): Raster transform.
        rst_shape (tuple): Raster shape.
        crs (CRS): Coordinate Reference System.
        config (dict): Configuration dictionary.
        elapid (module, optional): Elapid module for MaxEnt. Defaults to None.
    """
    print(f'Processing Species: {target_species}')
    presence_gdf = plant_spatial_data[plant_spatial_data['종명'] == target_species].copy()

    # species occurrence가 10개 미만인 species는 모델링에서 제외
    if len(presence_gdf) < 10:
        print(f'Selected Species: {target_species}, Presence records: {len(presence_gdf)}')
        print('Species occurrence is less than 10, excluding from modeling')
        return

    print(f'Selected Species: {target_species}, Presence records: {len(presence_gdf)}')

    # Save presence_gdf
    gdf_dir = os.path.join('..', 'Model', 'gdf')
    os.makedirs(gdf_dir, exist_ok=True)
    presence_gdf.to_file(os.path.join(gdf_dir, f'{target_species}_presence.gpkg'), driver='GPKG')
    print(f"Saved presence_gdf to {gdf_dir}/{target_species}_presence.gpkg")


    # Background Points (Must be specified in sdm_config.json)
    if 'BG_RATIO' not in config or 'MIN_BG_POINTS' not in config:
        raise KeyError("[Error] 'BG_RATIO' and 'MIN_BG_POINTS' MUST be specified in sdm_config.json.")
    
    bg_ratio = config['BG_RATIO']
    min_bg_points = config['MIN_BG_POINTS']
    num_background = max(int(len(presence_gdf) * bg_ratio), min_bg_points)
    background_points = []

    min_x, min_y, max_x, max_y = src.bounds
    max_tries = num_background * 50
    tries = 0

    while len(background_points) < num_background and tries < max_tries:
        rx = random.uniform(min_x, max_x)
        ry = random.uniform(min_y, max_y)

        r, c = rasterio.transform.rowcol(transform, rx, ry)

        if 0 <= r < rst_shape[0] and 0 <= c < rst_shape[1]:
            # Use the global elevation baseline to check if the pixel is on land (valid)
            # This allows background points to be generated safely even if elevation
            # is NOT included in the final env_layers dict for MaxEnt training.
            elev_val = elevation_2020.values[r, c]
            
            if np.isfinite(elev_val) and elev_val > -9000:
                background_points.append(Point(rx, ry))

        tries += 1

    if len(background_points) < num_background:
        print(f"[WARN] Only generated {len(background_points)} background points (requested {num_background}).")

    background_gdf = gpd.GeoDataFrame(geometry=background_points, crs=crs)

    # Extract Features
    # Extract Features
    print("Extracting features...")
    
    # Reproject to match raster CRS
    if presence_gdf.crs != crs:
        presence_gdf = presence_gdf.to_crs(crs)

    X_presence = extract_values_at_points(presence_gdf, env_layers, transform, rst_shape)
    
    # Filter out NoData or invalid values (-9999, NaN)
    # Important for island areas or edge cases
    X_presence = X_presence.replace(-9999, np.nan).dropna()
    
    if len(X_presence) < 10:
        print(f'Selected Species: {target_species}, Valid presence records after extraction: {len(X_presence)}')
        print('Valid species occurrence is less than 10 (points likely in NoData areas), excluding from modeling')
        return

    y_presence = np.ones(len(X_presence))
    
    X_background = extract_values_at_points(background_gdf, env_layers, transform, rst_shape)
    # Filter out NoData or invalid values (-9999, NaN)
    X_background = X_background.replace(-9999, np.nan).dropna()
    y_background = np.zeros(len(X_background))

    # Ensure LULC is treated as categorical with fixed categories
    lulc_categories = [int(c) for c in config['LULC_CATEGORIES']]
    lulc_dtype = pd.CategoricalDtype(categories=lulc_categories, ordered=False)

    X_presence['LULC'] = X_presence['LULC'].astype(int).astype(lulc_dtype)
    X_background['LULC'] = X_background['LULC'].astype(int).astype(lulc_dtype)

    X = pd.concat([X_presence, X_background], ignore_index=True)
    y = np.concatenate([y_presence, y_background])

    # Train/Test Split
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # --- Z-TRANSFORM (Standardization) ---
    apply_z_transform = config['APPLY_Z_TRANSFORM']
    scaler = None
    continuous_cols = []
    
    if apply_z_transform:
        print("Normalizing (Z-transforming) continuous variables...")
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        
        # Identify continuous columns dynamically from config
        categorical_layers = config.get('CATEGORICAL_LAYERS', ['LULC'])
        continuous_cols = [col for col in X_train_full.columns if col not in categorical_layers]
        
        # Fit scaler strictly on training set to avoid data leakage
        if continuous_cols:
            scaler.fit(X_train_full[continuous_cols])
            
            # Apply transformation to train and test sets
            X_train_full[continuous_cols] = scaler.transform(X_train_full[continuous_cols])
            X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])
            
            # Also apply to the full X_presence array used for thresholding later
            X_presence[continuous_cols] = scaler.transform(X_presence[continuous_cols])
            
            print(f"Applied Z-transform to {len(continuous_cols)} variables.")
        else:
            print("[WARN] No continuous columns found for normalization.")
    else:
        print("Skipping Z-transform normalization (APPLY_Z_TRANSFORM config is False).")
    
    # CV & Training
    print(f"Training MaxEnt for {target_species}...")
    if elapid is None:
        raise ImportError("elapid is required for MaxEnt but not installed.")
    
    kf = KFold(n_splits=config['N_FOLDS'], shuffle=True, random_state=42)

    X_train_full = X_train_full.reset_index(drop=True)
    if isinstance(y_train_full, pd.Series): y_train_full = y_train_full.values

    maxent_params = config['MAXENT_PARAMS']
    feature_types = maxent_params['feature_types']
    beta_multipliers = maxent_params.get('beta_multipliers', [maxent_params.get('beta_multiplier', 1.5)])
    n_cpus = maxent_params['n_cpus']
    clamp = maxent_params.get('clamp', True)

    best_mean_auc = -1
    best_beta = beta_multipliers[0]
    beta_stats = []

    print(f"Testing beta multipliers: {beta_multipliers}")

    for beta in beta_multipliers:
        fold_aucs = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
            X_f_train = X_train_full.iloc[train_idx]
            y_f_train = y_train_full[train_idx]
            X_f_val = X_train_full.iloc[val_idx]
            y_f_val = y_train_full[val_idx]

            model = elapid.MaxentModel(
                feature_types=feature_types,
                beta_multiplier=beta,
                n_cpus=n_cpus,
                clamp=clamp
            )
            
            # Check class balance in current fold
            unique_classes = np.unique(y_f_train)
            if len(unique_classes) < 2:
                continue

            model.fit(X_f_train, y_f_train)
            
            # Filter unknown categories in validation set
            try:
                known_cats = model.transformer.estimators_['categorical'].estimators_[0].categories_[0]
                valid_mask = X_f_val['LULC'].isin(known_cats)
                n_dropped = (~valid_mask).sum()
                if n_dropped > 0:
                     X_f_val = X_f_val[valid_mask]
                     y_f_val = y_f_val[valid_mask]
            except Exception:
                pass

            probs = model.predict_proba(X_f_val)
            if probs.shape[1] == 2:
                pred_prob = probs[:, 1]
            else:
                pred_prob = probs[:, 0] if model.classes_[0] == 1 else np.zeros(len(probs))
                
            try:
                auc = roc_auc_score(y_f_val, pred_prob)
                fold_aucs.append(auc)
            except ValueError:
                pass

        mean_auc = np.mean(fold_aucs) if fold_aucs else 0
        print(f"Beta {beta} - Average CV AUC: {mean_auc:.4f}")
        beta_stats.append((beta, mean_auc))
        
        if mean_auc > best_mean_auc:
            best_mean_auc = mean_auc
            best_beta = beta

    if best_mean_auc <= 0:
        print(f"Skipping {target_species}: No valid models trained (data quality issues).")
        return

    # --- PLOT CALIBRATION ---
    if len(beta_stats) > 1:
        SDMMODEL_DIR = config['SDMMODEL_DIR']
        os.makedirs(os.path.join(SDMMODEL_DIR, "ROC"), exist_ok=True)
        betas, aucs = zip(*beta_stats)
        plt.figure(figsize=(8, 5))
        plt.plot(betas, aucs, marker='o', linestyle='-', color='b')
        plt.scatter([best_beta], [best_mean_auc], color='red', s=100, zorder=5, label=f'Best Beta: {best_beta}')
        plt.title(f'Hyperparameter Calibration (Beta Multiplier)\n{target_species}')
        plt.xlabel('Beta Multiplier')
        plt.ylabel('Mean CV AUC')
        plt.grid(True, alpha=0.3)
        plt.legend()
        calib_plot_path = os.path.join(SDMMODEL_DIR, "ROC", f'Calibration_Beta_{target_species}.png')
        plt.savefig(calib_plot_path, dpi=300)
        plt.close()
        print(f"Saved calibration plot to: {calib_plot_path}")

    print(f"Optimal beta_multiplier chosen: {best_beta} with CV AUC: {best_mean_auc:.4f}")
    
    # Train Final Model on 100% of Training Data
    print("Training final model on full training set...")
    best_model = elapid.MaxentModel(
        feature_types=feature_types,
        beta_multiplier=best_beta,
        n_cpus=n_cpus,
        clamp=clamp
    )
    best_model.fit(X_train_full, y_train_full)
    
    # Get probabilities
    # Elapid predict returns expected suitability (continuous), predict_proba returns [1-suit, suit]
    # Filter X_test to ensure only known categories are present
    try:
         # Access allowed categories from the trained model's transformer
        # Structure: model.transformer.estimators_['categorical'].estimators_[0].categories_[0]
        # Assumes 'LULC' is the only/first categorical feature
        known_cats = best_model.transformer.estimators_['categorical'].estimators_[0].categories_[0]
        
        valid_mask = X_test['LULC'].isin(known_cats)
        n_dropped = (~valid_mask).sum()
        if n_dropped > 0:
            print(f"[WARN] Dropping {n_dropped} test samples with unknown LULC categories not in training set.")
            X_test = X_test[valid_mask]
            y_test = y_test[valid_mask]
    except Exception as e:
        print(f"[WARN] Could not filter unknown categories: {e}. Prediction may fail if new categories exist.")

    y_prob = best_model.predict_proba(X_test)[:, 1]

    # --- THRESHOLD CALCULATION ---
    threshold_method = config['THRESHOLD_METHOD']
    
    if threshold_method == 'TP10':
        # 10th Percentile Training Presence (Using ALL presence points for stability with small N)
        # Predict on all presence points
        try:
             # Repredict on all presence points for thresholding
            probs_presence = best_model.predict_proba(X_presence)
            if probs_presence.shape[1] == 2:
                all_presence_probs = probs_presence[:, 1]
            else:
                all_presence_probs = probs_presence[:, 0] if best_model.classes_[0] == 1 else np.zeros(len(probs_presence))
            
            best_threshold = np.percentile(all_presence_probs, 10)
            print(f"TP10 Threshold (based on {len(all_presence_probs)} presence points): {best_threshold:.4f}")
            
        except Exception as e:
            print(f"[WARN] Failed to calc TP10 on full set: {e}. Fallback to Youden.")
            threshold_method = 'Youden'

    # Calculate ROC curve for plotting and Youden calculation if needed
    if len(np.unique(y_test)) > 1:
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        youden_j = tpr - fpr

        # Calculate Youden's J for each threshold
        youden_threshold = thresholds[np.argmax(youden_j)]
            
        test_auc_score = float(roc_auc_score(y_test, y_prob))
        print(f"ROC-AUC Score: {test_auc_score:.4f}")
    else:
        print("[WARN] Only one class present in y_test after LULC filtering. Cannot compute ROC/AUC.")
        fpr, tpr, thresholds = np.array([0., 1.]), np.array([0., 1.]), np.array([1., 0.])
        youden_threshold = 0.5
        youden_j = np.array([0., 0.])
        test_auc_score = np.nan

    if threshold_method != 'TP10' or 'best_threshold' not in locals():
        best_threshold = youden_threshold
        print(f"Optimal Threshold (Youden's J 95% CI lower bound): {best_threshold:.4f}")

    # Find closest index in ROC curve for plotting
    best_threshold_idx = np.argmin(np.abs(thresholds - best_threshold))

    # Generate binary predictions using optimal threshold
    y_pred = (y_prob >= best_threshold).astype(int)

    print("Classification Report:")
    print("[NOTE] Precision/Recall metrics below reflect binary evaluation against pseudo-absences and should be interpreted carefully given inherent class imbalance.")
    if len(np.unique(y_test)) > 1:
        print(classification_report(y_test, y_pred))
    else:
        print("[WARN] Classification report unavailable (single class in test set).")

    # Save Model
    SDMMODEL_DIR = config['SDMMODEL_DIR']
    
    # Ensure all required subdirectories exist
    os.makedirs(os.path.join(SDMMODEL_DIR, "Models"), exist_ok=True)
    os.makedirs(os.path.join(SDMMODEL_DIR, "Thresholds"), exist_ok=True)
    os.makedirs(os.path.join(SDMMODEL_DIR, "ROC"), exist_ok=True)
    
    model_filename = f'MaxEnt_baseline_{target_species}.pkl' 
    joblib.dump(best_model, os.path.join(SDMMODEL_DIR, "Models", model_filename))
    
    # Save Scaler
    if continuous_cols:
        scaler_filename = f'Scaler_baseline_{target_species}.pkl'
        joblib.dump(scaler, os.path.join(SDMMODEL_DIR, "Models", scaler_filename))

    threshold_data = {
        'threshold': float(best_threshold),
        'auc': test_auc_score,
        'method': threshold_method
    }
    threshold_path = os.path.join(SDMMODEL_DIR, "Thresholds", f'threshold_MaxEnt_baseline_{target_species}_{threshold_method}.json')
    with open(threshold_path, 'w') as f:
        json.dump(threshold_data, f)
    print(f"Saved threshold to {threshold_path}")

    # --- PLOT ROC CURVE WITH THRESHOLD ---
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {test_auc_score:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess', alpha=0.7)

    # Plot optimal threshold point
    best_fpr = fpr[best_threshold_idx]
    best_tpr = tpr[best_threshold_idx]
    plt.scatter(best_fpr, best_tpr, c='red', s=100, label=f'Optimal Threshold ({best_threshold:.4f})', zorder=10, edgecolors='black')

    # Annotate
    annotation_text = f'Threshold: {best_threshold:.4f}\nYouden\'s J: {youden_j[best_threshold_idx]:.4f}\nTPR: {best_tpr:.3f}, FPR: {best_fpr:.3f}'
    plt.annotate(annotation_text, (best_fpr, best_tpr), textcoords="offset points", xytext=(20, -20), 
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8), fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'ROC Curve & {threshold_method} Threshold\n{target_species} (baseline)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    roc_plot_path = os.path.join(SDMMODEL_DIR, "ROC", f'ROC_MaxEnt_baseline_{target_species}_{threshold_method}.png')
    plt.savefig(roc_plot_path, dpi=300)
    print(f"Saved ROC plot to: {roc_plot_path}")
    plt.close()


# %%
target_species = species_records_df.iloc[11]["Species_Name"]
process_species_sdm(target_species, plant_spatial_data, env_layers, src, transform, rst_shape, crs, config, elapid)

# # %%
#  # 상동잎쥐똥나무
# target_species = combined_target_species[sp_idx]
# process_species_sdm(target_species, plant_spatial_data, env_layers, src, transform, rst_shape, crs, config, elapid)
# 정향풀 n=29



# %%
# do the loop for sp_idx from 0 to len(combined_target_species)


for idx, row in species_records_gt10.iterrows():
    target_species = row["Species_Name"]
    process_species_sdm(
        target_species,
        plant_spatial_data,
        env_layers,
        src,
        transform,
        rst_shape,
        crs,
        config,
        elapid
    )