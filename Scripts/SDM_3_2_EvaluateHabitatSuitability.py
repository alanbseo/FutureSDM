
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


def process_scenario(species_name, scenario_idx, climate_scenario_name, full_scenario_name, run_mode):
    print(f"\n==================================================")
    print(f"Evaluating {species_name} | [{run_mode}] Scenario {scenario_idx+1}/{len(CLIMATE_SCENARIO_NAMES)}: {climate_scenario_name}")
    print(f"==================================================")

    future_output_dir = f"../Output/SDM/Projection_{full_scenario_name}_{TARGET_YEAR}_{run_mode}"
    raster_dir = Path(future_output_dir) / "Rasters"
    
    base_path = raster_dir / f"HS_Base_{species_name}.tif"
    future_path = raster_dir / f"HS_Future_{species_name}.tif"
    
    if not os.path.exists(base_path) or not os.path.exists(future_path):
        print(f"[WARN] Result tifs missing for {species_name} {full_scenario_name}. Did you run script 1? Skipping.")
        return None

    # Load previously saved predictions
    ref_raster = load_raster(base_path)
    hq_base = ref_raster.values
    hq_future = load_raster(future_path).values
    
    hq_delta = hq_future - hq_base
    
    # Get Extent for Plotting
    bounds = ref_raster.rio.bounds()
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]

    # Load Threshold
    threshold = 0.5 # default
    
    # The threshold json file now has a suffix (e.g., _TP10.json or _Youden.json)
    # Use glob to find the correct file
    import glob
    search_pattern = os.path.join(sdm_config['SDMMODEL_DIR'], "Thresholds", f"threshold_MaxEnt_baseline_{species_name}_*.json")
    threshold_files = glob.glob(search_pattern)
    
    # Fallback for backward compatibility with old files
    if not threshold_files:
        legacy_path = os.path.join(sdm_config['SDMMODEL_DIR'], "Thresholds", f"threshold_MaxEnt_baseline_{species_name}.json")
        if os.path.exists(legacy_path):
            threshold_files = [legacy_path]
            
    if threshold_files:
        threshold_path = threshold_files[0]
        try:
            with open(threshold_path, 'r') as f:
                threshold_data = json.load(f)
            threshold = threshold_data.get('threshold', 0.5)
            print(f"Loaded Threshold from {threshold_path}: {threshold:.4f}")
        except Exception as e:
            print(f"[WARN] Error reading threshold JSON {threshold_path}: {e}")
    else:
        print(f"[WARN] No threshold JSON found for {species_name}. Defaulting to 0.5")
        return None

    # Load Observations (Presence Points)
    gdf_path = f"gdf/{species_name}_presence.gpkg"
    if os.path.exists(gdf_path):
        presence_gdf = gpd.read_file(gdf_path)
        # Ensure CRS matches
        target_crs = ref_raster.rio.crs
        if target_crs is None:
             target_crs = sdm_config.get('TARGET_CRS', 'EPSG:5179')
             print(f"[WARN] Raster CRS missing. Using fallback: {target_crs}")
             
        if presence_gdf.crs != target_crs:
            presence_gdf = presence_gdf.to_crs(target_crs)
    else:
        print(f"[WARN] Presence GDF not found at {gdf_path}")
        presence_gdf = None

    # Apply Threshold
    # We first apply the threshold, but note that the NaN mask should be propagated from base/future
    bin_base = np.where(np.isnan(hq_base), np.nan, (hq_base >= threshold).astype(float))
    bin_future = np.where(np.isnan(hq_future), np.nan, (hq_future >= threshold).astype(float))
    
    # Optional: explicitly mask bin_future with the nodata areas from bin_base
    bin_future[np.isnan(bin_base)] = np.nan

    # Change Class
    change_class = bin_base * 2 + bin_future
    
    # --- Plotting ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Continuous Probability
    im1 = axes[0, 0].imshow(hq_base, cmap='viridis', vmin=0, vmax=1, extent=extent)
    axes[0, 0].set_title(f"Baseline Habitat Suitability ({BASE_YEAR_NOMINAL})")
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    axes[0, 0].axis('off')

    # Overlay Observations
    if presence_gdf is not None:
        presence_gdf.plot(ax=axes[0, 0], color='red', markersize=10, marker='*', label='Observations', zorder=10)

    im2 = axes[0, 1].imshow(hq_future, cmap='viridis', vmin=0, vmax=1, extent=extent)
    axes[0, 1].set_title(f"Future Habitat Suitability ({TARGET_YEAR})")
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    axes[0, 1].axis('off')
    
    # Robust TwoSlopeNorm
    delta_min = np.nanmin(hq_delta)
    delta_max = np.nanmax(hq_delta)
    
    # Handle all-NaN or zero-range cases
    if np.isnan(delta_min) or np.isnan(delta_max):
        delta_min, delta_max = -0.1, 0.1
    
    # Ensure range spans 0 for TwoSlopeNorm
    if delta_min >= 0: delta_min = -0.01
    if delta_max <= 0: delta_max = 0.01
    
    div_norm = mcolors.TwoSlopeNorm(vmin=delta_min, vcenter=0, vmax=delta_max)
    im3 = axes[0, 2].imshow(hq_delta, cmap='RdBu', norm=div_norm, extent=extent)
    axes[0, 2].set_title(f"Habitat Suitability Change")
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04, label="delta HS")
    axes[0, 2].axis('off')
    
    # Row 2: Binary & Change
    # Custom CMP for Change (Pastel Tones): 
    # 0: Unsuitable (#f0f0f0 - Very Light Gray)
    # 1: Expansion (#a6cee3 - Pastel Blue)
    # 2: Loss (#fb9a99 - Pastel Red)
    # 3: Stable (#b2df8a - Pastel Green)
    cmap_change = mcolors.ListedColormap(['#f0f0f0', '#a6cee3', '#fb9a99', '#b2df8a'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm_change = mcolors.BoundaryNorm(bounds, cmap_change.N)
    
    # Base Binary
    axes[1, 0].imshow(bin_base, cmap='Greys', vmin=0, vmax=1, extent=extent) 
    axes[1, 0].imshow(np.where(bin_base==1, 1, np.nan), cmap='Greens', vmin=0, vmax=1, alpha=0.7, extent=extent)
    axes[1, 0].set_title(f"Suitable Habitat ({BASE_YEAR_NOMINAL})\nBased on Youden's J(= {threshold:.3f})")
    axes[1, 0].axis('off')
    
    # Overlay Observations on Binary Map too
    if presence_gdf is not None:
        presence_gdf.plot(ax=axes[1, 0], color='red', markersize=10, marker='*', zorder=10)


    # Future Binary
    axes[1, 1].imshow(bin_future, cmap='Greys', vmin=0, vmax=1, extent=extent)
    axes[1, 1].imshow(np.where(bin_future==1, 1, np.nan), cmap='Greens', vmin=0, vmax=1, alpha=0.7, extent=extent)
    axes[1, 0].set_title(f"Suitable Habitat ({TARGET_YEAR})\nBased on Youden's J(= {threshold:.3f})")

    axes[1, 1].axis('off')
    
    # Change Map
    im6 = axes[1, 2].imshow(change_class, cmap=cmap_change, norm=norm_change, extent=extent)
    axes[1, 2].set_title(f"Habitat Suitability Change ({BASE_YEAR_NOMINAL} to {TARGET_YEAR})")
    axes[1, 2].axis('off')
    
    
    # Legend for Change (Custom Map Legend)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='#b2df8a', edgecolor='gray', label='Stable'),
        Patch(facecolor='#a6cee3', edgecolor='gray', label='Gain'),
        Patch(facecolor='#fb9a99', edgecolor='gray', label='Loss'),
        Patch(facecolor='#f0f0f0', edgecolor='gray', label='Unsuitable')
    ]
    
    # Add Observations to Legend if present
    if presence_gdf is not None:
         legend_elements.append(Line2D([0], [0], marker='*', color='w', label='Observations',
                          markerfacecolor='red', markersize=10))

    axes[1, 2].legend(handles=legend_elements, loc='lower right', fontsize='small', 
                      title='Habitat Change', title_fontsize='small', framealpha=0.9)
    # Remove old colorbar code
    # cbar = plt.colorbar... (removed)
    # cbar.ax.set_yticklabels... (removed)

    # add si-do (korean provincial boundaries) polygons on each map

    # for ax in axes.flat:
    #     ax.imshow(si_do_polygons, cmap='Greys', alpha=0.5)

    # Add colorbar
    cbar = plt.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(['Unsuitable', 'Gain', 'Loss', 'Stable'])
    
    plt.suptitle(f"Habitat Suitability Dynamics: {species_name} ({full_scenario_name})", fontsize=16)
    plt.tight_layout()
    
    out_plot = Path(future_output_dir) / f"Dynamics_Map_MaxEnt_{species_name}_{full_scenario_name}_{BASE_YEAR_NOMINAL}-{TARGET_YEAR}.png"
    plt.savefig(out_plot, dpi=300)
    print(f"Saved dynamics plot to: {out_plot}")
    plt.close(fig) # Close to free memory
    
    # --- Apply Mountain (5) & Grassland (4) LULC filtering for stats ---
    # Load base LULC (cov_all.0.asc)
    lulc_base_path = Path(MODEL_INPUT_DIR) / "cov_all.0.asc"
    lulc_base = load_raster(lulc_base_path).values if lulc_base_path.exists() else np.full_like(change_class, -1)
    
    # Load future LULC
    # Target year is 2050, which corresponds to yr_idx = 2050 - 2021 + 1 = 30
    yr_idx = TARGET_YEAR - 2021 + 1
    lulc_dir = f"{LULC_RUN_BASE_DIR}/{full_scenario_name}"
    lulc_future_path = Path(lulc_dir) / f"cov_all.{yr_idx}.asc"
    if not lulc_future_path.exists():
        lulc_future_path = lulc_base_path # fallback
    lulc_future = load_raster(lulc_future_path).values if lulc_future_path.exists() else np.full_like(change_class, -1)
    
    # Mask definition: pixel must be natural (2=Forest, 3=Grassland)
    # in both the start and end year to be analyzed for suitability stats
    natural_base = (lulc_base == 2) | (lulc_base == 3) | (lulc_base == 5)
    natural_future = (lulc_future == 2) | (lulc_future == 3) | (lulc_future == 5)
    #valid_natural_mask = natural_base & natural_future
    valid_natural_mask =  natural_future

    # Cast to int first so == comparisons are exact; use -1 as nodata sentinel.
    cc_int = np.where(np.isnan(change_class), -1, np.round(change_class).astype(int))
    
    # Filter stats to ONLY include valid natural mask areas
    n_stable     = int(np.sum((cc_int == 3) & valid_natural_mask))
    n_gain       = int(np.sum((cc_int == 1) & valid_natural_mask))
    n_loss       = int(np.sum((cc_int == 2) & valid_natural_mask))
    n_unsuitable = int(np.sum((cc_int == 0) & valid_natural_mask))

    # --- Save Binary Rasters for SDM_5 group stacking ---
    raster_dir = Path(future_output_dir) / "Rasters"
    raster_dir.mkdir(parents=True, exist_ok=True)

    # Suitable 2050 (bin_future: 0/1/NaN)
    _save_float32_raster(
        bin_future.astype(np.float32),
        raster_dir / f"Suitable2050_{species_name}.tif",
        ref_raster
    )
    # Gain: pixels that become suitable (0 → 1)
    gain_arr = np.where(cc_int == -1, np.nan,
                        np.where(cc_int == 1, 1.0, 0.0)).astype(np.float32)
    _save_float32_raster(
        gain_arr,
        raster_dir / f"Gain_{species_name}.tif",
        ref_raster
    )
    # Loss: pixels that lose suitability (1 → 0)
    loss_arr = np.where(cc_int == -1, np.nan,
                        np.where(cc_int == 2, 1.0, 0.0)).astype(np.float32)
    _save_float32_raster(
        loss_arr,
        raster_dir / f"Loss_{species_name}.tif",
        ref_raster
    )
    print(f"  Saved rasters to {raster_dir}")

    return {
        "scenario":      full_scenario_name,
        "suitable_2020": n_stable + n_loss,   # base-suitable = stable + loss
        "suitable_2050": n_stable + n_gain,   # future-suitable = stable + gain
        "stable":        n_stable,
        "gain":          n_gain,
        "loss":          n_loss,
        "unsuitable":    n_unsuitable,
    }

def plot_scenario_comparison(species_name, stats_list, run_mode):
    """Generates comparison plots for all scenarios."""
    df = pd.DataFrame(stats_list)
    output_dir = "../Output/SDM/"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, f"Stats_{run_mode}"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, f"Comparison_{run_mode}"), exist_ok=True)
    
    # Save stats
    df.to_csv(os.path.join(output_dir, f"Stats_{run_mode}/Scenario_Stats_MaxEnt_{species_name}.csv"), index=False)
    
    # 1. Suitable Habitat Area (2020 vs 2050)
    plt.figure(figsize=(10, 6))
    
    # Data Preparation: 2020 (Base) once, then 2050 (Future) for each scenario
    val_2020 = df["suitable_2020"].iloc[0]
    vals_2050 = df["suitable_2050"].tolist()
    
    labels = [str(BASE_YEAR_NOMINAL)] + df["scenario"].tolist()
    values = [val_2020] + vals_2050
    colors = ['skyblue'] + ['orange'] * len(vals_2050)
    
    x_pos = np.arange(len(labels))
    
    plt.bar(x_pos, values, color=colors)
    
    plt.xlabel('Scenarios')
    plt.ylabel('Suitable Habitat Area (Km2)')
    plt.title(f'Suitable Habitat Area Comparison: {species_name}')
    plt.xticks(x_pos, labels, rotation=45, ha='right')
    
    # Custom Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='skyblue', label=f'{BASE_YEAR_NOMINAL} (Baseline)'),
        Patch(facecolor='orange', label=f'{TARGET_YEAR} (Scenario)')
    ]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Comparison_{run_mode}/Scenario_Comparison_Area_MaxEnt_{species_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Habitat Change Dynamics (Stacked Bar)
    plt.figure(figsize=(12, 6))
    
    # Categories: Unsuitable(0), Gain(1), Loss(2), Stable(3)
    # Order for stacking: Stable, Gain, Loss, Unsuitable (or user pref)
    # Change class 0 is "Unsuitable" (0->0).
    # Let's stack: Stable, Gain, Loss, Unsuitable (0->0)
    
    p4 = plt.bar(df["scenario"], df["unsuitable"], bottom=df["stable"]+df["gain"]+df["loss"], label='Unsuitable', color='#f0f0f0')
    p3 = plt.bar(df["scenario"], df["loss"], bottom=df["stable"]+df["gain"], label='Loss', color='#fb9a99')
    p2 = plt.bar(df["scenario"], df["gain"], bottom=df["stable"], label='Gain', color='#a6cee3')
    p1 = plt.bar(df["scenario"], df["stable"], label='Stable', color='#b2df8a')

    plt.xlabel('Scenarios')
    plt.ylabel('Area (Km2)')
    plt.title(f'Habitat Change Dynamics (2050): {species_name}')
    plt.xticks(rotation=45, ha='right')
    # Legend reverse order usually looks better for stacks, but standard is fine
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Comparison_{run_mode}/Scenario_Comparison_Change_MaxEnt_{species_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def run_all_scenarios():
    print(f"Total Species to Process: {len(SPECIES_LIST)}")
    for run_mode in ["Constrained", "Unconstrained"]:
        print(f"\n{'#'*60}")
        print(f"### EVALUATING PROJECTIONS IN MODE: {run_mode.upper()}")
        print(f"{'#'*60}")
        for sp_name in SPECIES_LIST:
            print(f"*** Processing Species: {sp_name} [{run_mode}] ***")
            stats_list = []
            for idx in range(len(FULL_SCENARIO_FNAMES)): 
                s_name = CLIMATE_SCENARIO_NAMES[idx]
                s_fname = FULL_SCENARIO_FNAMES[idx]
                stats = process_scenario(sp_name, idx, s_name, s_fname, run_mode)
                if stats:
                    stats_list.append(stats)
            
            if stats_list:
                plot_scenario_comparison(sp_name, stats_list, run_mode)
        

if __name__ == "__main__":
    run_all_scenarios()


