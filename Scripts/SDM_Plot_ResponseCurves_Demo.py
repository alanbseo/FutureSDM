import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
import rioxarray
import rasterio
import xarray as xr
from pathlib import Path
import warnings

# Suppress elapid matmul overflow warnings during prediction
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Setup env
try:
    import elapid
except ImportError:
    print("Please run this script in the CLUE environment.")
    exit(1)

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

config_path = 'SDM/sdm_config.json'
with open(config_path, 'r') as f:
    config = json.load(f)

print("Loading subset of environmental layers for ranges...")
PRJ_ROOT = os.path.dirname(".")
BIOCLIM_DIR = os.path.join(PRJ_ROOT, "Model", "Input_Bioclim_KMA")
MODELINPUT_DIR = os.path.join(PRJ_ROOT, "Model")
CLIMATE_DIR = os.path.join(PRJ_ROOT, "Model", "Input_Climate_KMA")
scenario_name = "ssp126"

bioclim_dir = os.path.join(BIOCLIM_DIR, scenario_name)
bioclim_names = config['BIOCLIM_NAMES']
bioclim_list = []
sorted_bio_keys = sorted(bioclim_names.keys(), key=lambda x: int(x.replace('bio', '')))

for bio_key in sorted_bio_keys:
    fpath = os.path.join(bioclim_dir, f'kma_{scenario_name}_2021_{bio_key}.asc')
    if os.path.exists(fpath):
        da = rioxarray.open_rasterio(fpath, masked=True).squeeze("band", drop=True)
        da_bio = da.assign_coords(bio=bioclim_names[bio_key]).expand_dims("bio")
        bioclim_list.append(da_bio)

if bioclim_list:
    bioclim = xr.concat(bioclim_list, dim="bio")

elevation_2020 = rioxarray.open_rasterio(Path(MODELINPUT_DIR) / "sc1gr0.0.asc", masked=True).squeeze(drop=True)
slope_2020 = rioxarray.open_rasterio(Path(MODELINPUT_DIR) / "sc1gr1.0.asc", masked=True).squeeze(drop=True)

# Just need approximate ranges, we don't need to load every single layer. We will dynamically load min/max if needed.
# Let's map variable to rough ranges explicitly or gracefully handle.
env_ranges_cache = {}

def get_range(var_name):
    if var_name == 'elevation': return 0, 1950
    elif var_name == 'slope': return 0, 90
    elif var_name == 'soildeep3': return 0, 10
    elif var_name == 'soildra1': return 0, 5
    elif var_name == 'soilstone1': return 0, 5
    elif 'bioclim' in var_name:
        bionum = var_name.replace('bioclim', '')
        bio_str = bioclim_names.get(f"bio{bionum}")
        if bio_str and 'bioclim' in locals():
            arr = bioclim.sel(bio=bio_str).values
            valid = arr[arr > -9000]
            if len(valid) > 0:
                p1, p99 = np.nanpercentile(valid, [1, 99])
                return float(p1), float(p99)
        # Default fallback
        if bionum in ['1','5','6','8','9','10','11']: return -10, 35
        elif bionum in ['4']: return 1000, 15000
        elif bionum in ['7']: return 20, 50
        elif bionum in ['12']: return 800, 2500
        elif bionum in ['13','14','16','17','18','19']: return 0, 1000
        else: return 0, 100
    elif var_name == 'VPD': return 0.0, 15.0
    return 0, 100

env_config = config.get('ENV_LAYERS', {})
continuous_cols = []
for var in env_config.get('topography', []): continuous_cols.append(var)
for var in env_config.get('soil', []): continuous_cols.append(var)
for var in env_config.get('bioclim', []): continuous_cols.append(f'bioclim{var}')
for var in env_config.get('climate', []): continuous_cols.append(var)

categorical_cols = config.get('CATEGORICAL_LAYERS', ['LULC'])
lulc_categories = [int(c) for c in config['LULC_CATEGORIES']]
lulc_dtype = pd.CategoricalDtype(categories=lulc_categories, ordered=False)

all_columns = continuous_cols + categorical_cols

output_dir = '../Output/SDM/ResponseCurves'
os.makedirs(output_dir, exist_ok=True)

models = glob.glob('../Output/SDM/Models/MaxEnt_baseline_*.pkl')
target_species_list = ['구상나무', '난쟁이바위솔', '광릉요강꽃', '가문비나무', '설앵초', '금강초롱꽃']
models_to_process = [m for m in models if any(ts in m for ts in target_species_list)]

if not models_to_process:
    models_to_process = models[:3]

print(f"Generating Response Curves for {len(models_to_process)} species...")

for model_path in models_to_process:
    species_name = os.path.basename(model_path).replace('MaxEnt_baseline_', '').replace('.pkl', '')
    
    print(f"Processing {species_name}...")
    model = joblib.load(model_path)
    
    n_vars = len(continuous_cols) + len(categorical_cols)
    cols = 4
    rows = (n_vars + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten()
    fig.suptitle(f'{species_name} - Marginal Response Curves', fontsize=16)
    
    for idx, var in enumerate(continuous_cols):
        ax = axes[idx]
        
        vmin, vmax = get_range(var)
        x_vals = np.linspace(vmin, vmax, 100)
        
        # Create a dummy dataframe with median values
        dummy_data = np.zeros((100, len(continuous_cols)))
        for j, c in enumerate(continuous_cols):
            cvmin, cvmax = get_range(c)
            dummy_data[:, j] = (cvmin + cvmax) / 2.0
            
        dummy_df = pd.DataFrame(dummy_data, columns=continuous_cols)
        
        # Set the target variable to the sequence
        dummy_df[var] = x_vals
        
        # Categorical columns
        dummy_df['LULC'] = pd.Series([1]*100).astype(lulc_dtype)  # LULC == Forest
        
        # Rearrange columns
        try:
           X_pred = dummy_df[all_columns]
        except KeyError:
           continue
        
        # Predict
        try:
            probs = model.predict_proba(X_pred)
            # Determine full display name
            display_name = var
            if var.startswith('bioclim'):
                bio_key = var.replace('bioclim', 'bio')
                bio_desc = config.get('BIOCLIM_NAMES', {}).get(bio_key, '')
                if bio_desc:
                    display_name = f"{var}\n({bio_desc})"
            elif var in config.get('VARIABLE_LABELS', {}):
                display_name = f"{var}\n({config['VARIABLE_LABELS'][var]})"
                
            ax.plot(x_vals, prob, '-', color='darkgreen', linewidth=2)
            ax.set_title(display_name, fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)[:20]}", ha='center', va='center')
            
    # Do LULC
    if 'LULC' in categorical_cols:
        ax = axes[len(continuous_cols)]
        cats = config['LULC_CATEGORIES']
        probs = []
        for cat in cats:
            dummy_data = np.zeros((1, len(continuous_cols)))
            for j, c in enumerate(continuous_cols):
                cvmin, cvmax = get_range(c)
                dummy_data[:, j] = (cvmin + cvmax) / 2.0
            dummy_df = pd.DataFrame(dummy_data, columns=continuous_cols)
            dummy_df['LULC'] = pd.Series([cat]).astype(lulc_dtype)
            X_pred = dummy_df[all_columns]
            
            try:
                p = model.predict_proba(X_pred)
                p_val = p[0, 1] if p.shape[1] == 2 else (p[0, 0] if model.classes_[0] == 1 else 0)
            except:
                p_val = 0
            probs.append(p_val)
            
        ax.bar([str(c) for c in cats], probs, color='saddlebrown')
        ax.set_title('LULC Categories', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis='y')

    # Remove empty plots
    for i in range(n_vars, len(axes)):
        fig.delaxes(axes[i])
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(output_dir, f'ResponseCurve_{species_name}.png')
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved: {save_path}")

print("Done plotting response curves.")
