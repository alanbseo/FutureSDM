# MaxEnt  
 
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import rasterio
import pandas as pd
import numpy as np
# import elapid





def train_maxent_model(species_name, presence_points, env_layers):
    """
    MaxEnt 모델 학습 및 검증 함수 (Pseudocode)
    """
    print(f"Training MaxEnt for {species_name}...")
    
    # 1. 환경 변수 샘플링 (Point sampling)
    # X = sample_env_at_points(presence_points, env_layers)
    # y = [1] * len(X) # Presence only
    # Backgound point 생성 필요
    
    # 2. 5-fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    
    # for train_idx, val_idx in kf.split(X):
    #     model = MaxentModel()
    #     model.fit(X_train, y_train)
    #     pred = model.predict(X_val)
    #     auc = roc_auc_score(y_val, pred)
    #     auc_scores.append(auc)
    
    # print(f"Average AUC: {np.mean(auc_scores):.3f}")
    return None # Return trained model



def extract_values_at_points(gdf, env_dict, transform, shape):
    """Extract raster values at point locations"""
    coords = [(x,y) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]
    rows, cols = rasterio.transform.rowcol(transform, [p[0] for p in coords], [p[1] for p in coords])
    
    data = []
    for i in range(len(rows)):
        r, c = rows[i], cols[i]
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            row_data = {}
            valid_point = True
            for name, layer in env_dict.items():
                val = layer[r, c]
                # Check for NoData (often -9999 or similar in ASC)
                if val == -9999 or val < -9000 or (isinstance(val, float) and np.isnan(val)):
                    valid_point = False
                    break
                row_data[name] = val
            if valid_point:
                data.append(row_data)
    return pd.DataFrame(data)