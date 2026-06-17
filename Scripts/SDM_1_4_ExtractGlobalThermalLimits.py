import os
import json
import time
import requests
import rasterio
import numpy as np
import pandas as pd
from urllib.request import urlretrieve
import zipfile
import geopandas as gpd

import json
try:
    with open("../sdm_config.json", "r", encoding="utf-8") as _f:
        _cfg = json.load(_f)
except:
    _cfg = {}


# --- CONFIGURATION ---
CONFIG_PATH = 'SDM/sdm_config.json'
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

OUTPUT_DIR = os.path.join(config.get('SDMMODEL_DIR', '../Output/SDM'), "ThermalLimits")
os.makedirs(OUTPUT_DIR, exist_ok=True)
LIMITS_JSON_PATH = os.path.join(OUTPUT_DIR, "Species_ThermalLimits.json")

WORLDCLIM_ZIP_URL = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_bio.zip"
WC_DIR = os.path.join(_cfg.get("OUTPUT_BASE_DIR", "../Model/"), "Input_Climate_WorldClim_Global")
os.makedirs(WC_DIR, exist_ok=True)
BIO5_TIFF = os.path.join(WC_DIR, "wc2.1_10m_bio_5.tif")
BIO6_TIFF = os.path.join(WC_DIR, "wc2.1_10m_bio_6.tif")

PLANT_GPKG_PATH = _cfg.get("PLANT_GPKG_PATH", "../Data/Plant_Spatial_Data_05Feb2026.gpkg")

# Target species list
try:
    species_df = pd.read_csv("../Output/SDM/species_records.csv")
    target_species_list = species_df.query("count > 10")['Species_Name'].tolist()
except FileNotFoundError:
    print("[ERROR] species_records.csv not found. Please run SDM_2_BuildSDM.py first to generate the species list.")
    exit(1)


# --- HELPERS ---
def download_worldclim_bio5_6():
    if os.path.exists(BIO5_TIFF) and os.path.exists(BIO6_TIFF):
        print(f"Global Bio5 and Bio6 rasters already exist in {WC_DIR}")
        return
        
    zip_path = os.path.join(WC_DIR, "wc2.1_10m_bio.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading WorldClim 10m global bioclim data (approx 250MB)...")
        headers = {'User-Agent': 'Mozilla/5.0'}
        with requests.get(WORLDCLIM_ZIP_URL, stream=True, headers=headers) as r:
            r.raise_for_status()
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete.")
        
    print("Extracting bio5 and bio6...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        files = z.namelist()
        bio5_file = [f for f in files if f.endswith('bio_5.tif')][0]
        bio6_file = [f for f in files if f.endswith('bio_6.tif')][0]
        z.extract(bio5_file, WC_DIR)
        z.extract(bio6_file, WC_DIR)
        
    print(f"Successfully extracted {BIO5_TIFF} and {BIO6_TIFF}")


def suggest_scientific_name(korean_name):
    """
    Tries to map a Korean common name to a scientific name using Korean Wikipedia -> Wikidata extraction.
    GBIF's API does not natively support many Korean vernacular names.
    """
    headers = {'User-Agent': 'SDM_Bot/1.0 (contact@example.com)'}
    
    # 1. First, attempt Korean Wikipedia page properties
    try:
        wp_url = f"https://ko.wikipedia.org/w/api.php?action=query&prop=pageprops&titles={korean_name}&format=json"
        wp_resp = requests.get(wp_url, headers=headers, timeout=10).json()
        pages = wp_resp.get('query', {}).get('pages', {})
        
        for page_id, page_info in pages.items():
            if 'pageprops' in page_info and 'wikibase_item' in page_info['pageprops']:
                wd_id = page_info['pageprops']['wikibase_item']
                
                # 2. Extract P225 (taxon name) from Wikidata
                wd_url = f"https://www.wikidata.org/w/api.php?action=wbgetclaims&entity={wd_id}&property=P225&format=json"
                wd_resp = requests.get(wd_url, headers=headers, timeout=10).json()
                claims = wd_resp.get('claims', {}).get('P225', [])
                
                if claims:
                    sci_name = claims[0]['mainsnak']['datavalue']['value']
                    return sci_name
    except Exception as e:
        print(f"  [WARN] Wikipedia API lookup failed for {korean_name}: {e}")
        
    return None

def fetch_gbif_coordinates(scientific_name, max_records=2000):
    """
    Fetches up to `max_records` global coordinates (decimalLatitude, decimalLongitude) for a given scientific name.
    """
    url = "https://api.gbif.org/v1/occurrence/search"
    params = {
        "scientificName": scientific_name,
        "hasCoordinate": "true",
        "hasGeospatialIssue": "false",
        "limit": min(300, max_records), # GBIF limit per page is 300
        "offset": 0
    }
    
    coords = []
    print(f"  Fetching GBIF occurrences for {scientific_name}...", end="")
    while params["offset"] < max_records:
        try:
            resp = requests.get(url, params=params, timeout=15).json()
            results = resp.get('results', [])
            if not results:
                break
                
            for r in results:
                lat = r.get('decimalLatitude')
                lon = r.get('decimalLongitude')
                if lat is not None and lon is not None:
                    coords.append((lon, lat)) # (x, y) for rasterio
                    
            if resp.get('endOfRecords'):
                break
                
            params["offset"] += params["limit"]
        except Exception as e:
            print(f" Error: {e}", end="")
            break
            
    print(f" Found {len(coords)} global coords.")
    return coords


def load_local_coordinates(target_species):
    """
    Loads local presence data from GPKG and transforms to WGS84 (EPSG:4326).
    Returns a dict mapping Korean Name -> list of (lon, lat) tuples.
    """
    print(f"Loading local geometries from {PLANT_GPKG_PATH}...")
    try:
        # Use pyogrio engine for blazing fast GPKG read
        gdf = gpd.read_file(PLANT_GPKG_PATH, engine="pyogrio")
    except Exception as e:
        print(f"  [WARN] Using default fiona for GPKG load. (pyogrio failed: {e})")
        gdf = gpd.read_file(PLANT_GPKG_PATH)
        
    print(f"  Loaded {len(gdf)} records. Reprojecting to ESPG:4326...")
    gdf = gdf.to_crs("EPSG:4326")
    
    local_coords = {name: [] for name in target_species}
    
    # Fast iteration over the geometry column
    valid_gdf = gdf.dropna(subset=['종명', 'geometry'])
    
    # Pre-filter for target species to save memory/time
    valid_gdf = valid_gdf[valid_gdf['종명'].isin(target_species)]
    
    for _, row in valid_gdf.iterrows():
        sp = row['종명']
        pt = row['geometry']
        local_coords[sp].append((pt.x, pt.y))
        
    # Remove empty lists
    local_coords = {k: v for k, v in local_coords.items() if len(v) > 0}
    print(f"  Found local coords for {len(local_coords)} target species.")
    return local_coords


def extract_thermal_limits(coords, src_b5, src_b6):
    """
    Samples bio5/bio6 rasters at coords and returns:
    (bio5 99th percentile, bio6 1st percentile)
    """
    if not coords:
        return None, None
        
    # Sample rasters
    try:
        samples_b5 = list(src_b5.sample(coords))
        samples_b6 = list(src_b6.sample(coords))
        
        vals_b5 = [s[0] for s in samples_b5 if s[0] != src_b5.nodata and not np.isnan(s[0])]
        vals_b6 = [s[0] for s in samples_b6 if s[0] != src_b6.nodata and not np.isnan(s[0])]
    except Exception as e:
        print(f"  Sampling error: {e}")
        return None, None
        
    # Exclude extreme anomalies (> 60C or < -80C)
    valid_b5 = [v for v in vals_b5 if -50 <= v <= 60]
    valid_b6 = [v for v in vals_b6 if -80 <= v <= 40]
    
    max_b5, min_b6 = None, None
    if valid_b5:
        max_b5 = round(float(np.percentile(valid_b5, 99)), 2)
    if valid_b6:
        min_b6 = round(float(np.percentile(valid_b6, 1)), 2)
        
    return max_b5, min_b6


# --- MAIN PIPELINE ---
def main():
    print("=== Step 1: Downloading Global Climate Data ===")
    download_worldclim_bio5_6()
    
    target_list = target_species_list
    print(f"Total target species: {len(target_species_list)}. Processing {len(target_list)} species...")
    
    local_coords_dict = load_local_coordinates(target_list)
    
    print("\n=== Step 2: Processing Species Thermal Limits ===")
    
    # Load existing limits if present to allow resuming
    limits_dict = {}
    if os.path.exists(LIMITS_JSON_PATH):
        with open(LIMITS_JSON_PATH, 'r') as f:
            limits_dict = json.load(f)
            
    try:
        src5 = rasterio.open(BIO5_TIFF)
        src6 = rasterio.open(BIO6_TIFF)
    except Exception as e:
        print(f"[ERROR] Failed to open WorldClim TIFFs: {e}")
        return

    processed = 0
    for sp_idx, kor_name in enumerate(target_list):
        if kor_name in limits_dict and "combined_bio5_max" in limits_dict[kor_name]:
            # Simple check if current format already includes combined stats
            print(f"[{sp_idx+1}/{len(target_list)}] {kor_name} -> Already processed ({limits_dict[kor_name]['combined_bio5_max']}°C). Skipping.")
            continue
            
        print(f"[{sp_idx+1}/{len(target_list)}] Processing {kor_name}...")
        
        # 1. Local Extraction
        local_coords = local_coords_dict.get(kor_name, [])
        l_b5, l_b6 = extract_thermal_limits(local_coords, src5, src6)
        
        # 2. GBIF Extraction
        sci_name = suggest_scientific_name(kor_name)
        gbif_coords = []
        g_b5, g_b6 = None, None
        if sci_name:
            gbif_coords = fetch_gbif_coordinates(sci_name, max_records=1500)
            g_b5, g_b6 = extract_thermal_limits(gbif_coords, src5, src6)
        else:
            print(f"  [WARN] Could not find scientific name for '{kor_name}'. Skipping GBIF.")
            
        # 3. Combine Limits
        # If one is None, max() or min() with None fails, so handle gracefully
        c_b5, c_b6 = None, None

        # Helper to compute max safely
        def safe_max(a, b):
            if a is None and b is None: return None
            if a is None: return b
            if b is None: return a
            return max(a, b)
            
        # Helper to compute min safely
        def safe_min(a, b):
            if a is None and b is None: return None
            if a is None: return b
            if b is None: return a
            return min(a, b)
            
        c_b5 = safe_max(g_b5, l_b5)
        c_b6 = safe_min(g_b6, l_b6)
        
        status = "success"
        if c_b5 is None or c_b6 is None:
            status = "failed_climate"
            c_b5 = c_b5 if c_b5 is not None else 40.0
            c_b6 = c_b6 if c_b6 is not None else -40.0
            
        print(f"  [SUCCESS] Limits for {kor_name} | bio5: G={g_b5},L={l_b5}->{c_b5}C | bio6: G={g_b6},L={l_b6}->{c_b6}C")
        
        limits_dict[kor_name] = {
            "sci_name": sci_name if sci_name else "UNKNOWN",
            "gbif_records": len(gbif_coords),
            "local_records": len(local_coords),
            "gbif_bio5_max": g_b5,
            "gbif_bio6_min": g_b6,
            "local_bio5_max": l_b5,
            "local_bio6_min": l_b6,
            "combined_bio5_max": c_b5,
            "combined_bio6_min": c_b6,
            "status": status
        }
            
        # Save incrementally
        with open(LIMITS_JSON_PATH, 'w') as f:
            json.dump(limits_dict, f, indent=4, ensure_ascii=False)
            
        time.sleep(0.5) # Be nice to GBIF API
        processed += 1
        
    src5.close()
    src6.close()
    print(f"\n=== Finished processing ===")
    print(f"Successfully updated {LIMITS_JSON_PATH} with combined (GBIF+Local) macroecological thermal constraints.")
    print("You can now apply these constraints in SDM_3_1_ProjectHabitatSuitability.py")

if __name__ == "__main__":
    main()
