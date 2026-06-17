import os
import json
import numpy as np
import rasterio
from pathlib import Path

def one_hot_encode_raster(input_raster_path, output_dir, categories):
    """
    Apply one-hot encoding to a categorical raster and save binary rasters for each category.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with rasterio.open(input_raster_path) as src:
        data = src.read(1)
        profile = src.profile
        
        # Update profile for output rasters (keep spatial info, ensure float or int)
        # Using int8 for binary masks to save space
        profile.update(dtype=rasterio.int8, count=1, nodata=0)
        
        for cat in categories:
            cat_val = int(cat)
            output_name = f"LULC_cat_{cat_val}.asc"
            output_path = os.path.join(output_dir, output_name)
            
            # Create binary mask
            binary_mask = (data == cat_val).astype(np.int8)
            
            # Handle NoData: if original was NoData, we might want to keep it or set to 0
            # Usually for SDM, we want 0 for "not this category" and 1 for "this category"
            # But we should respect the original mask if possible.
            # In this case, we just set 1 where it matches and 0 otherwise.
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(binary_mask, 1)
            
            print(f"Created binary raster for category {cat_val}: {output_path}")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    config_path = os.path.join('SDM', "../sdm_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Path to LULC raster
    # Based on SDM_2_BuildSDM.py: LULC_2020 = rioxarray.open_rasterio(Path(MODELINPUT_DIR) / "cov_all.0.asc", masked=True).squeeze(drop=True)
    # MODELINPUT_DIR = os.path.join(PRJ_ROOT, "Model")
    prj_root = "."
    lulc_path = os.path.join(prj_root, "Model", "cov_all.0.asc")
    
    output_dir = os.path.join(prj_root, "Model", "OneHotEncoded")
    
    lulc_categories = config.get('LULC_CATEGORIES', [0, 1, 2, 3, 4, 5, 6])
    
    print(f"Starting one-hot encoding for: {lulc_path}")
    print(f"Categories to process: {lulc_categories}")
    print(f"Output directory: {output_dir}")
    
    if not os.path.exists(lulc_path):
        print(f"Error: Input raster not found at {lulc_path}")
    else:
        one_hot_encode_raster(lulc_path, output_dir, lulc_categories)
        print("One-hot encoding complete.")
