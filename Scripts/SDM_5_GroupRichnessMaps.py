"""
SDM_5_GroupRichnessMaps.py
==========================
Plot richness, gain, and loss maps in 2050 for each of four species groups:

  - 멸종위기종 1급  (멸종위기 등급 == '1')
  - 멸종위기종 2급  (멸종위기 등급 == '2')
  - 기후변화 민감종 북방종 (기후변화.취약식물 == 1)
  - 기후변화 민감종 남방종 (기후변화.취약식물 == 2)

Features:
  1. Saves aggregated rasters for each Group + Scenario combinations as .tif.
  2. Generates comprehensive 4x3 plot for each Scenario.
  3. Generates comprehensive 4x3 plot for each Group.
  4. Generates individual 1x3 plots for each Group + Scenario combination.
"""

import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

import json
try:
    with open("../SDM_config.json", "r", encoding="utf-8") as _f:
        _cfg = json.load(_f)
except:
    _cfg = {}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

FULL_SCENARIO_FNAMES = [
    "BAU-SSP585",
    "Climate-SSP126",
    "Biodiversity-SSP126",
    "Biodiversity-SSP245",
]

TARGET_YEAR = 2050

PLANT_GPKG_PATH = (
    "../Data/"
    "NIE Data/생태계서비스팀 공간정보/07. 전국자연환경조사/"
    "Plant_Spatial_Data_05Feb2026.gpkg"
)

SDM_OUTPUT_DIR = "../Output/SDM"

# ---------------------------------------------------------------------------
# Group Definitions & Metrics
# ---------------------------------------------------------------------------
GROUP_CONFIG = [
    {"key": "endangered_1",  "label": "멸종위기종 1급", "filter": ("멸종위기 등급", "1")},
    {"key": "endangered_2",  "label": "멸종위기종 2급", "filter": ("멸종위기 등급", "2")},
    {"key": "climate_north", "label": "기후변화 민감종\n북방종", "filter": ("기후변화.취약식물", 1)},
    {"key": "climate_south", "label": "기후변화 민감종\n남방종", "filter": ("기후변화.취약식물", 2)},
]

METRIC_CONFIGS = [
    {"prefix": "Suitable2020", "label": "Richness (2020)", "cmap": "YlGn"},
    {"prefix": "Suitable2050", "label": "Richness (2050)", "cmap": "YlGn"},
    {"prefix": "Loss",         "label": "Loss richness",   "cmap": "Reds"},
]


# ---------------------------------------------------------------------------
# Raster Helpers
# ---------------------------------------------------------------------------
def read_raster(path: Path) -> np.ndarray:
    with rasterio.open(str(path)) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr

def get_raster_meta(path: Path) -> dict:
    with rasterio.open(str(path)) as src:
        return {"crs": src.crs, "transform": src.transform, "shape": (src.height, src.width)}

def save_raster(arr: np.ndarray, meta: dict, out_path: Path):
    with rasterio.open(
        str(out_path), 'w', driver='GTiff',
        height=meta["shape"][0], width=meta["shape"][1], count=1,
        dtype=np.float32, crs=meta["crs"], transform=meta["transform"], nodata=np.nan
    ) as dst:
        dst.write(arr.astype(np.float32), 1)

# ---------------------------------------------------------------------------
# Data Processing
# ---------------------------------------------------------------------------
def build_group_species(gpkg_path: str, modelled_species: set) -> dict:
    print("Loading plant spatial data for group membership ...")
    gdf = gpd.read_file(gpkg_path)
    groups = {}
    for cfg in GROUP_CONFIG:
        col, val = cfg["filter"]
        if col not in gdf.columns:
            groups[cfg["key"]] = []
            continue
        sp_in_group = gdf.loc[gdf[col] == val, "종명"].dropna().unique().tolist()
        sp_in_group = [s for s in sp_in_group if s in modelled_species]
        groups[cfg["key"]] = sp_in_group
        print(f"  {cfg['label'].replace(chr(10), ' ')}: {len(sp_in_group)} modelled spp.")
    return groups

def stack_rasters(raster_dir: Path, prefix: str, species_list: list, global_mask: np.ndarray = None):
    accumulated = None
    n_found = 0
    meta = None
    valid_mask = None

    for sp in species_list:
        if prefix == "Suitable2020":
            # Reconstruct Suitable2020 = Suitable2050 - Gain + Loss
            p_2050 = raster_dir / f"Suitable2050_{sp}.tif"
            p_gain = raster_dir / f"Gain_{sp}.tif"
            p_loss = raster_dir / f"Loss_{sp}.tif"
            if p_2050.exists() and p_gain.exists() and p_loss.exists():
                arr_2050 = read_raster(p_2050)
                arr_gain = read_raster(p_gain)
                arr_loss = read_raster(p_loss)
                arr = arr_2050 - arr_gain + arr_loss
                if meta is None:
                    meta = get_raster_meta(p_2050)
            else:
                continue
        else:
            fpath = raster_dir / f"{prefix}_{sp}.tif"
            if not fpath.exists():
                continue
            if meta is None:
                meta = get_raster_meta(fpath)
            arr = read_raster(fpath)
        
        if valid_mask is None:
            valid_mask = ~np.isnan(arr)
        else:
            valid_mask = valid_mask | (~np.isnan(arr))
            
        arr_filled = np.where(np.isnan(arr), 0.0, arr)
        
        if accumulated is None:
            accumulated = arr_filled
        else:
            accumulated += arr_filled
        n_found += 1

    if accumulated is not None and valid_mask is not None:
        # Restore NaN outside of the valid area of all rasters combined
        accumulated[~valid_mask] = np.nan
        # If a strict terrestrial/LULC mask is provided, apply it here
        if global_mask is not None:
            accumulated[~global_mask] = np.nan

    return accumulated, meta, n_found

def process_and_cache_group_rasters(scenario_name: str, group_species: dict, stats_dict: dict, global_mask: np.ndarray, raster_out_dir: Path, phys_mode: str):
    """Stacks and saves the aggregated rasters for a single scenario."""
    raster_dir = Path(SDM_OUTPUT_DIR) / f"Projection_{scenario_name}_{TARGET_YEAR}_{phys_mode}" / "Rasters"
    if not raster_dir.exists():
        print(f"  [WARN] Raster dir not found: {raster_dir}. Skipping cache.")
        return

    print(f"  Aggregating rasters for {scenario_name}...")
    for grp_cfg in GROUP_CONFIG:
        grp_key = grp_cfg["key"]
        sp_list = group_species.get(grp_key, [])
        for met_cfg in METRIC_CONFIGS:
            out_name = f"{met_cfg['prefix']}_{grp_key}_{scenario_name}.tif"
            out_path = raster_out_dir / out_name
            
            arr, meta, n_found = stack_rasters(raster_dir, met_cfg["prefix"], sp_list, global_mask)
            stats_dict[(scenario_name, grp_key, met_cfg["prefix"])] = {
                "n_found": n_found,
                "n_total": len(sp_list),
                "vmax": float(np.nanmax(arr)) if arr is not None else 1.0,
            }
            if arr is not None and meta is not None:
                save_raster(arr, meta, out_path)

# ---------------------------------------------------------------------------
# Plotting Functions
# ---------------------------------------------------------------------------
def _add_map_subplot(ax, arr, met_cfg, grp_label, n_found, n_total, vmax_override=None):
    ax.set_xticks([])
    ax.set_yticks([])
    if arr is None:
        if n_total == 0:
            ax.set_title(f"{grp_label}\n{met_cfg['label']}\n(no species)", fontsize=9)
            ax.text(0.5, 0.5, "no species in group", ha="center", va="center", transform=ax.transAxes, color="gray")
        else:
            ax.set_title(f"{grp_label}\n{met_cfg['label']}", fontsize=9)
            ax.text(0.5, 0.5, "rasters missing", ha="center", va="center", transform=ax.transAxes, color="gray")
        return

    vmax = vmax_override if vmax_override is not None else max(np.nanmax(arr), 1)
    im = ax.imshow(arr, cmap=met_cfg["cmap"], vmin=0, vmax=vmax, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Number of species")
    title_str = f"{grp_label}\n{met_cfg['label']}\n({n_found}/{n_total} spp. mapped)"
    ax.set_title(title_str, fontsize=9)


def plot_by_scenario(scenario_name: str, group_species: dict, stats_dict: dict, map_out_dir: Path, raster_out_dir: Path):
    """Plot all groups (4) x all metrics (3) for a given scenario."""
    n_groups = len(GROUP_CONFIG)
    n_metrics = len(METRIC_CONFIGS)

    fig, axes = plt.subplots(n_groups, n_metrics, figsize=(5 * n_metrics, 4.5 * n_groups), squeeze=False)

    for row_idx, grp_cfg in enumerate(GROUP_CONFIG):
        grp_key = grp_cfg["key"]
        for col_idx, met_cfg in enumerate(METRIC_CONFIGS):
            ax = axes[row_idx, col_idx]
            
            stat = stats_dict.get((scenario_name, grp_key, met_cfg["prefix"]), {})
            n_found = stat.get("n_found", 0)
            n_total = stat.get("n_total", 0)
            
            tif_path = raster_out_dir / f"{met_cfg['prefix']}_{grp_key}_{scenario_name}.tif"
            arr = read_raster(tif_path) if tif_path.exists() else None
            
            _add_map_subplot(ax, arr, met_cfg, grp_cfg["label"], n_found, n_total)

    plt.suptitle(f"Group Habitat Dynamics – {scenario_name} ({TARGET_YEAR})", fontsize=16, y=1.01)
    plt.tight_layout()
    out_path = map_out_dir / f"GroupMap_ByScenario_{scenario_name}_{TARGET_YEAR}.png"
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_by_group(grp_cfg: dict, stats_dict: dict, map_out_dir: Path, raster_out_dir: Path, binary: bool = False):
    """Plot all scenarios (4) x all metrics (3) for a given species group."""
    grp_key = grp_cfg["key"]
    grp_label = grp_cfg["label"]
    
    n_scenarios = len(FULL_SCENARIO_FNAMES)
    n_metrics = len(METRIC_CONFIGS)

    fig, axes = plt.subplots(n_scenarios, n_metrics, figsize=(5 * n_metrics, 4.5 * n_scenarios), squeeze=False)

    # To make maps comparable across scenarios for the same group, find global max for each metric
    vmax_dict = {}
    for col_idx, met_cfg in enumerate(METRIC_CONFIGS):
        if binary:
            vmax_dict[col_idx] = 1.0
        else:
            vmax_list = [
                stats_dict.get((scen, grp_key, met_cfg["prefix"]), {}).get("vmax", 0.0)
                for scen in FULL_SCENARIO_FNAMES
            ]
            vmax_dict[col_idx] = max(max(vmax_list), 1.0) if vmax_list else 1.0
            
    if not binary and len(vmax_dict) >= 2:
        max_richness = max(vmax_dict[0], vmax_dict[1])
        vmax_dict[0] = max_richness
        vmax_dict[1] = max_richness

    for row_idx, scenario_name in enumerate(FULL_SCENARIO_FNAMES):
        for col_idx, met_cfg in enumerate(METRIC_CONFIGS):
            ax = axes[row_idx, col_idx]
            
            stat = stats_dict.get((scenario_name, grp_key, met_cfg["prefix"]), {})
            n_found = stat.get("n_found", 0)
            n_total = stat.get("n_total", 0)
            
            tif_path = raster_out_dir / f"{met_cfg['prefix']}_{grp_key}_{scenario_name}.tif"
            arr = read_raster(tif_path) if tif_path.exists() else None
            
            if binary and arr is not None:
                arr_bin = np.zeros_like(arr)
                arr_bin[arr > 0] = 1.0
                arr_bin[np.isnan(arr)] = np.nan
                arr = arr_bin
            
            combined_label = f"[{scenario_name}]\n{grp_label}"
            
            _add_map_subplot(ax, arr, met_cfg, combined_label, n_found, n_total, vmax_override=vmax_dict[col_idx])

    group_title = grp_label.replace("\n", " ")
    bin_str = " (Binary)" if binary else ""
    plt.suptitle(f"Scenario Comparison{bin_str} – {group_title} ({TARGET_YEAR})", fontsize=16, y=1.01)
    plt.tight_layout()
    bin_suffix = "_Binary" if binary else ""
    out_path = map_out_dir / f"GroupMap_ByGroup_{grp_key}{bin_suffix}_{TARGET_YEAR}.png"
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_combinations(group_species: dict, stats_dict: dict, map_out_dir: Path, raster_out_dir: Path):
    """Generate individual 1x3 plots for EACH group & scenario pair."""
    for grp_cfg in GROUP_CONFIG:
        grp_key = grp_cfg["key"]
        grp_label = grp_cfg["label"]
        
        # Calculate global max for this group across scenarios to standardize color scales
        vmax_dict = {}
        for col_idx, met_cfg in enumerate(METRIC_CONFIGS):
            vmax_list = [
                stats_dict.get((scen, grp_key, met_cfg["prefix"]), {}).get("vmax", 0.0)
                for scen in FULL_SCENARIO_FNAMES
            ]
            vmax_dict[col_idx] = max(max(vmax_list), 1.0) if vmax_list else 1.0
        
        if len(vmax_dict) >= 2:
            max_richness = max(vmax_dict[0], vmax_dict[1])
            vmax_dict[0] = max_richness
            vmax_dict[1] = max_richness
            
        for scenario_name in FULL_SCENARIO_FNAMES:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)
            for col_idx, met_cfg in enumerate(METRIC_CONFIGS):
                ax = axes[0, col_idx]
                
                stat = stats_dict.get((scenario_name, grp_key, met_cfg["prefix"]), {})
                n_found = stat.get("n_found", 0)
                n_total = stat.get("n_total", 0)
                
                tif_path = raster_out_dir / f"{met_cfg['prefix']}_{grp_key}_{scenario_name}.tif"
                arr = read_raster(tif_path) if tif_path.exists() else None
                
                combined_label = f"[{scenario_name}]\n{grp_label}"
                _add_map_subplot(ax, arr, met_cfg, combined_label, n_found, n_total, vmax_override=vmax_dict[col_idx])

            title_str = f"{grp_label.replace(chr(10), ' ')} – {scenario_name} ({TARGET_YEAR})"
            plt.suptitle(title_str, fontsize=14, y=1.05)
            plt.tight_layout()
            out_path = map_out_dir / f"GroupMap_Individual_{grp_key}_{scenario_name}_{TARGET_YEAR}.png"
            plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
            plt.close(fig)
    print(f"  Saved all individual combination plots.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    with open("SDM/sdm_config.json", "r") as f:
        config = json.load(f)
    models_dir = Path(config.get("SDMMODEL_DIR", SDM_OUTPUT_DIR)) / "Models"
    modelled_species = set()
    if models_dir.exists():
        for fname in os.listdir(models_dir):
            if fname.startswith("MaxEnt_baseline_") and fname.endswith(".pkl"):
                sp = fname[len("MaxEnt_baseline_"):-len(".pkl")]
                modelled_species.add(sp)
    print(f"Found {len(modelled_species)} modelled species.")

    group_species = build_group_species(PLANT_GPKG_PATH, modelled_species)
    
    # 0. Pre-load Masks
    # LULC masking logic (Mountain=5, Grassland=4)
    # We load base, and we can load future if needed, but for simplicity we will
    # build a scenario-specific mask inside the loop to be robust.
    MODEL_INPUT_DIR = _cfg.get("OUTPUT_BASE_DIR", "../Model/")
    LULC_RUN_BASE_DIR = _cfg.get("LULC_RUN_BASE_DIR", "../LULC_Scenarios/")
    
    # Run 4 times: combinations of Physiological Constraint and LULC Mask
    for phys_mode in ["Constrained", "Unconstrained"]:
        for lulc_mode in ["Unmasked", "Masked"]:
            print(f"\n{'='*50}")
            print(f"Starting pipeline for mode: {phys_mode.upper()} & {lulc_mode.upper()}")
            print(f"{'='*50}")
            
            map_out_dir = Path("../Output/Plotting") / f"GroupRichnessMaps_{phys_mode}_{lulc_mode}"
            raster_out_dir = map_out_dir / "Rasters"
            map_out_dir.mkdir(parents=True, exist_ok=True)
            raster_out_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Process & Cache Rasters
            print(f"\n--- Phase 1: Aggregating into Group Rasters ({phys_mode}-{lulc_mode}) ---")
            stats_dict = {}
            for scenario in FULL_SCENARIO_FNAMES:
                
                global_mask = None
                if lulc_mode == "Masked":
                    # Generate intersection mask for base and future year for this scenario
                    lulc_base_path = Path(MODEL_INPUT_DIR) / "cov_all.0.asc"
                    lulc_base = read_raster(lulc_base_path) if lulc_base_path.exists() else None
                    
                    yr_idx = TARGET_YEAR - 2021 + 1
                    lulc_future_path = Path(f"{LULC_RUN_BASE_DIR}/{scenario}") / f"cov_all.{yr_idx}.asc"
                    if not lulc_future_path.exists():
                        lulc_future_path = lulc_base_path # fallback
                    lulc_future = read_raster(lulc_future_path) if lulc_future_path.exists() else None
                    
                    if lulc_future is not None:
                        # natural = 2 (Forest), 3 (Grassland), 5 (Barren)
                        natural_future = (lulc_future == 2) | (lulc_future == 3) | (lulc_future == 5)
                        global_mask = natural_future
                
                process_and_cache_group_rasters(scenario, group_species, stats_dict, global_mask, raster_out_dir, phys_mode)
    
            print(f"\n--- Phase 2: Generating By-Scenario Plots ({phys_mode}-{lulc_mode}) ---")
            for scenario in FULL_SCENARIO_FNAMES:
                plot_by_scenario(scenario, group_species, stats_dict, map_out_dir, raster_out_dir)
    
            print(f"\n--- Phase 3: Generating By-Group Plots ({phys_mode}-{lulc_mode}) ---")
            for grp_cfg in GROUP_CONFIG:
                plot_by_group(grp_cfg, stats_dict, map_out_dir, raster_out_dir, binary=False)
                plot_by_group(grp_cfg, stats_dict, map_out_dir, raster_out_dir, binary=True)
                
            print(f"\n--- Phase 4: Generating Individual Combinations ({phys_mode}-{lulc_mode}) ---")
            plot_combinations(group_species, stats_dict, map_out_dir, raster_out_dir)
    
            print(f"\nDone with {phys_mode} & {lulc_mode}. Maps and Rasters saved to:", map_out_dir)


if __name__ == "__main__":
    main()
