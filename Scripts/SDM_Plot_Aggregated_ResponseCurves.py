import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
import geopandas as gpd
import warnings

import json
try:
    with open("../SDM_config.json", "r", encoding="utf-8") as _f:
        _cfg = json.load(_f)
except:
    _cfg = {}


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

print("1. Loading configurations and group data...")
config_path = '../SDM_config.json'
with open(config_path, 'r') as f:
    config = json.load(f)

# Group logic similar to SDM_5
PLANT_GPKG_PATH = _cfg.get("PLANT_GPKG_PATH", "../Data/Plant_Spatial_Data_05Feb2026.gpkg")
gdf = gpd.read_file(PLANT_GPKG_PATH)

GROUP_CONFIG = [
    {"key": "All", "label": "All Species (전체 종)", "filter": None},
    {"key": "endangered_1",  "label": "Endangered Level 1 (멸종위기 1급)", "filter": ("멸종위기 등급", "1")},
    {"key": "endangered_2",  "label": "Endangered Level 2 (멸종위기 2급)", "filter": ("멸종위기 등급", "2")},
    {"key": "climate_north", "label": "Northern Climate Sensitive (위기/북방계)", "filter": ("기후변화.취약식물", 1)},
    {"key": "climate_south", "label": "Southern Climate Sensitive (위기/남방계)", "filter": ("기후변화.취약식물", 2)},
]

# Map species to groups
species_to_groups = {}
# Ensure basic structure
for sp in gdf['종명'].dropna().unique():
    species_to_groups[sp] = ["All"]

for cfg in GROUP_CONFIG[1:]:
    col, val = cfg["filter"]
    if col in gdf.columns:
        sp_in_group = gdf.loc[gdf[col] == val, "종명"].dropna().unique()
        for sp in sp_in_group:
            if sp in species_to_groups:
                species_to_groups[sp].append(cfg["key"])


# Environmental variables logic
def get_range(var_name):
    # Base heuristic ranges (simplified for speed since we want standard ranges across species)
    if var_name == 'elevation': return 0, 1950
    elif var_name == 'slope': return 0, 60
    elif var_name == 'soildeep3': return 0, 10
    elif var_name == 'soildra1': return 0, 5
    elif var_name == 'soilstone1': return 0, 5
    elif 'bioclim' in var_name:
        bionum = var_name.replace('bioclim', '')
        # Hard limits mimicking Korean terrain typical climate limits from previous scripts
        if bionum in ['1','5','6','8','9','10','11']: return -10, 35
        elif bionum in ['4']: return 1000, 15000
        elif bionum in ['7']: return 20, 50
        elif bionum in ['12']: return 800, 2500
        elif bionum in ['13','14','16','17','18','19']: return 0, 1000
        else: return 0, 100
    elif var_name == 'VPD': return 0.0, 15.0
    elif var_name == 'solarradiation': return 0, 30000
    elif var_name == 'relative_humidity': return 0, 100
    return 0, 100

env_config = config.get('ENV_LAYERS', {})
continuous_cols = []
for layer_type in ['topography', 'soil', 'climate']:
    for var in env_config.get(layer_type, []): 
        continuous_cols.append(var)
for var in env_config.get('bioclim', []): 
    continuous_cols.append(f'bioclim{var}')

categorical_cols = config.get('CATEGORICAL_LAYERS', ['LULC'])
lulc_categories = [int(c) for c in config['LULC_CATEGORIES']]
lulc_dtype = pd.CategoricalDtype(categories=lulc_categories, ordered=False)

all_columns = continuous_cols + categorical_cols

output_dir = '../Output/Plotting/ResponseCurves_Aggregated'
os.makedirs(output_dir, exist_ok=True)

model_dir = os.path.join(config.get('SDMMODEL_DIR', '../Output/SDM_Legacy'), 'Models')
models = glob.glob(os.path.join(model_dir, 'MaxEnt_baseline_*.pkl'))
print(f"2. Found {len(models)} models. Pre-computing responses...")

# Load Thermal Limits
limits_json_path = '../Output/SDM/ThermalLimits/Species_ThermalLimits.json'
thermal_limits = {}
if os.path.exists(limits_json_path):
    with open(limits_json_path, 'r') as f:
        thermal_limits = json.load(f)
    print(f"Loaded thermal limits for {len(thermal_limits)} species.")

plot_cols = list(continuous_cols)
if 'bioclim5' not in plot_cols: plot_cols.append('bioclim5')
if 'bioclim6' not in plot_cols: plot_cols.append('bioclim6')

# Store predictions: Group -> Var -> List of 100-element arrays
group_preds_unc = {cfg['key']: {var: [] for var in plot_cols} for cfg in GROUP_CONFIG}
group_preds_con = {cfg['key']: {var: [] for var in plot_cols} for cfg in GROUP_CONFIG}
# For LULC: Group -> List of (len(categories) array)
group_lulc_preds_unc = {cfg['key']: [] for cfg in GROUP_CONFIG}
group_lulc_preds_con = {cfg['key']: [] for cfg in GROUP_CONFIG}

for model_path in models:
    species_name = os.path.basename(model_path).replace('MaxEnt_baseline_', '').replace('.pkl', '')
    
    # Identify which groups this species belongs to
    my_groups = species_to_groups.get(species_name, [])
    if not my_groups: continue
    
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading {species_name}: {e}")
        continue
        
    # Calculate for each continuous var (including explicitly added plot_cols)
    for var in plot_cols:
        vmin, vmax = get_range(var)
        x_vals = np.linspace(vmin, vmax, 100)
        
        dummy_data = np.zeros((100, len(continuous_cols)))
        for j, c in enumerate(continuous_cols):
            cvmin, cvmax = get_range(c)
            dummy_data[:, j] = (cvmin + cvmax) / 2.0
            
        dummy_df = pd.DataFrame(dummy_data, columns=continuous_cols)
        if var in continuous_cols:
            dummy_df[var] = x_vals

        dummy_df['LULC'] = pd.Series([1]*100).astype(lulc_dtype)
        X_pred = dummy_df[all_columns]
        
        try:
            probs = model.predict_proba(X_pred)
            prob = probs[:, 1] if probs.shape[1] == 2 else (probs[:, 0] if model.classes_[0] == 1 else np.zeros(100))
        except:
            prob = np.zeros(100)
            
        prob_con = np.copy(prob)
        apply_hot = config.get('APPLY_HOT_CONSTRAINT', True)
        apply_cold = config.get('APPLY_COLD_CONSTRAINT', True)
        
        if apply_hot and var == 'bioclim5' and species_name in thermal_limits:
            limit = float(thermal_limits[species_name].get("combined_bio5_max", thermal_limits[species_name].get("bio5_max", 40.0)))
            prob_con[x_vals >= limit] = 0.0
            
        if apply_cold and var == 'bioclim6' and species_name in thermal_limits:
            limit = float(thermal_limits[species_name].get("combined_bio6_min", thermal_limits[species_name].get("bio6_min", -40.0)))
            prob_con[x_vals <= limit] = 0.0
            
        for g in my_groups:
            if g in group_preds_unc:
                group_preds_unc[g][var].append(prob)
                group_preds_con[g][var].append(prob_con)
                
    # Calculate for categorical LULC
    if 'LULC' in categorical_cols:
        cats = config['LULC_CATEGORIES']
        cats_probs = []
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
                
            p_val_con = p_val # For LULC, we don't apply thermal constraints (bio5 is fixed at midpoint, usually safe)
            cats_probs.append(p_val)
            
        for g in my_groups:
            if g in group_lulc_preds_unc:
                group_lulc_preds_unc[g].append(cats_probs)
                group_lulc_preds_con[g].append(cats_probs)


print("3. Drawing aggregate plots...")

for cfg in GROUP_CONFIG:
    grp_key = cfg["key"]
    grp_label = cfg["label"]
    
    n_species_in_group = len(group_preds_con[grp_key].get(plot_cols[0], []))
    if n_species_in_group == 0:
        print(f"Skipping {grp_label}: no modelled species found.")
        continue
        
    print(f"  Plotting {grp_label} ({n_species_in_group} species)...")
    
    n_vars = len(plot_cols) + len(categorical_cols)
    cols = 4
    rows = (n_vars + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten()
    fig.suptitle(f'Aggregated Marginal Response Curves\n{grp_label} (n={n_species_in_group})', fontsize=16)
    
    for idx, var in enumerate(plot_cols):
        ax = axes[idx]
        
        vmin, vmax = get_range(var)
        x_vals = np.linspace(vmin, vmax, 100)
        
        all_lines_unc = group_preds_unc[grp_key][var]
        all_lines_con = group_preds_con[grp_key][var]
        
        if not all_lines_unc:
            continue
            
        arr_unc = np.array(all_lines_unc)
        arr_con = np.array(all_lines_con)
        
        # Mean for Unconstrained & Constrained
        mean_unc = np.nanmean(arr_unc, axis=0)
        mean_con = np.nanmean(arr_con, axis=0)
        
        # Display Name
        display_name = var
        if var.startswith('bioclim'):
            bio_key = var.replace('bioclim', 'bio')
            bio_desc = config.get('BIOCLIM_NAMES', {}).get(bio_key, '')
            if bio_desc:
                display_name = f"{var}\n({bio_desc})"
        elif var in config.get('VARIABLE_LABELS', {}):
            display_name = f"{var}\n({config['VARIABLE_LABELS'][var]})"
        
        # Plot Unconstrained (Red Dashed)
        ax.plot(x_vals, mean_unc, color='red', linestyle='--', linewidth=3, alpha=0.5, label='MaxEnt (Unstrained)')
        
        # Plot Constrained (Blue Solid)
        ax.plot(x_vals, mean_con, color='blue', linestyle='-', linewidth=1.5, alpha=0.9, label='Hybrid SDM (Constrained)')
        
        ax.set_title(display_name, fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if idx == 0: ax.legend(fontsize=8, loc='upper right')

    # Do LULC
    if 'LULC' in categorical_cols:
        ax = axes[len(plot_cols)]
        all_lulc = group_lulc_preds_con[grp_key]
        if all_lulc:
            all_lulc_arr = np.array(all_lulc) # shape (N_species, N_cats)
            mean_cat = np.nanmean(all_lulc_arr, axis=0)
            std_cat = np.nanstd(all_lulc_arr, axis=0)
            cats = config['LULC_CATEGORIES']
            
            # Individual faint dots for scatter spread (optional, or just bar with error)
            ax.bar([str(c) for c in cats], mean_cat, yerr=std_cat, capsize=3, color='saddlebrown', alpha=0.7)
            ax.set_title('LULC Categories', fontsize=10)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3, axis='y')

    # Remove empty plots
    for i in range(n_vars, len(axes)):
        fig.delaxes(axes[i])
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(output_dir, f'Agg_ResponseCurve_{grp_key}.png')
    plt.savefig(save_path, dpi=200)
    plt.close()
    
print("Successfully generated all aggregated plots.")
