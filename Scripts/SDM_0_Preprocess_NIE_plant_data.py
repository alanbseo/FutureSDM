
# %%
import pandas as pd
import geopandas as gpd
import os
import numpy as np
import matplotlib.pyplot as plt


from datetime import datetime

# Define Paths
base_dir = '../Data/NIE Data/생태계서비스팀 공간정보/07. 전국자연환경조사'
meta_path = os.path.join(base_dir, 'Plant_metadata.xlsx')
coord_path = os.path.join(base_dir, 'Plant_coordinates.csv')
current_date = datetime.now().strftime('%d%b%Y')
output_path = os.path.join(base_dir, f'Plant_Spatial_Data_{current_date}.gpkg')

print("--- Starting Data Processing ---")


# 1. Load Data
print(f"Loading Metadata from: {meta_path}")
df_meta = pd.read_excel(meta_path)
print(f"  > Metadata loaded. Shape: {df_meta.shape}")
print(f"  > Metadata Columns: {df_meta.columns.tolist()}")


print(f"Loading Coordinates from: {coord_path}")

df_coord = pd.read_csv(coord_path, encoding='cp949')
print(f"  > Coordinates loaded. Shape: {df_coord.shape}")


target_path = os.path.join(base_dir, 'target_plant_species_5Feb2026.csv')
print(f"Loading target species list from: {target_path}")
df_target = pd.read_csv(target_path)
print(f"  > Target species list loaded. Shape: {df_target.shape}")
print(f"  > Target species list Columns: {df_target.columns.tolist()}")

# %%

# 2. Data Processing & Join
print("Preparing for Join...")

# Ensure join key 'ncode' is string in both to avoid mismatches
# Checking column names usually good practice, but assuming 'ncode' exists based on prior inspection
# Rename metadata column to match coordinates for easier merging
    

# rename some korean column names to English names
# df_meta.rename(columns={
#     '종명': 'Species_Name',
#     '종코드': 'Species_Code',
#     '경도': 'Longitude',
#     '위도': 'Latitude',
#     '적색목록': 'RedList',
#     '멸종위기 등급': 'Endangered', 
#     '기후변화지표종': 'ClimateIndicator',
#     '보정': 'RevisedYN'
# }, inplace=True)

# # Create ClimateSensitive column efficiently
# # Logic: If '남방계' is '2', then '2'. Else use '북방계'. 
# # finally replace '0' (str or int) with NaN.
# # Convert inputs to string to ensure safe comparison
# # String-based robust processing
# north = df_meta['북방계'].astype(str).str.strip().replace({'nan': np.nan, '0': np.nan, '0.0': np.nan}).str.replace(r'\.0$', '', regex=True)
# south = df_meta['남방계'].astype(str).str.strip().replace({'nan': np.nan, '0': np.nan, '0.0': np.nan}).str.replace(r'\.0$', '', regex=True)

# # ClimateSensitive logic:
# # If south == '2' → '2'
# # Else use north
# df_meta['ClimateSensitive'] = np.where(south == '2', '2', north)

# # Replace 0 with NaN (meaning "not climate sensitive")
# df_meta.loc[df_meta['ClimateSensitive'] == 0, 'ClimateSensitive'] = np.nan

# print(df_meta['ClimateSensitive'].value_counts(dropna=False))

df_meta.loc[df_meta['멸종위기 등급'] == '0', '멸종위기 등급'] = None

df_meta.loc[df_meta['멸종위기 등급'] == 'Ⅰ', '멸종위기 등급'] = '1'
df_meta.loc[df_meta['멸종위기 등급'] == 'Ⅱ', '멸종위기 등급'] = '2'

# # print(df_meta['ClimateIndicator'].value_counts(dropna=False))
# print(df_meta['Endangered'].value_counts(dropna=False))
# # print(df_meta['ClimateSensitive'].value_counts(dropna=False))

# print(df_meta['ClimateIndicator'].value_counts(dropna=False))
# print(df_meta['RedList'].value_counts(dropna=False))


# %%
# join target species list
df_merged = pd.merge(df_meta, df_target, on='nCode', how='left', suffixes=('', '_target'))
print(f"  > Merged DataFrame Columns: {df_merged.columns.tolist()}")
print(df_merged['기후변화지표종'].value_counts(dropna=False))



# %%
# crosstablulate Endangered and ClimateSensitive (including NA)
print(pd.crosstab(df_merged['기후변화.취약식물'], df_merged['기후변화지표종'], dropna=False))
print("df_merged.shape: ", df_merged.shape)



# %%
# can you plot the crosstablulate Endangered and ClimateSensitive
#pd.crosstab(df_meta['Endangered'], df_meta['ClimateSensitive']).plot(kind='bar', stacked=True)
plt.show()

# Fix nCode format mismatch (float in meta vs int in coord)
# Drop rows without nCode in metadata if any
df_merged = df_merged.dropna(subset=['nCode'])
df_merged['nCode'] = df_merged['nCode'].astype(int).astype(str)

# Create nCode in coord 
df_coord['nCode'] = df_coord['ncode'].astype(int).astype(str)
# DROP original 'ncode' to avoid case-insensitive collision in GPKG (ncode vs nCode)
if 'ncode' in df_coord.columns:
    df_coord.drop(columns=['ncode'], inplace=True)

print("Performing Left Join (Coordinates <- Metadata)...")
# We want spatial points, so we start with coordinates and join metadata to them.
# Note: If we want ALL metadata even without coords, we'd do right join, but usually for spatial data we need coords.
# The requirement was "Join these two files... Create spatial point data".

gdf_merged = pd.merge(df_coord, df_merged, on='nCode', how='left')
print(f"  > Join complete. Result shape: {gdf_merged.shape}")


# see how many data points are missing coordinates
missing_coords = gdf_merged[gdf_merged['경도'].isna()]
print(f"  > {len(missing_coords)} data points are missing coordinates") 
# how many missing in % 
print(f"  > {len(missing_coords) / len(gdf_merged) * 100:.2f}% of data points are missing coordinates")


print(gdf_merged.shape)
print(f"  > gdf_merged Columns: {gdf_merged.columns.tolist()}")

# %% 
# check the values of 기후변화지표종 and 기후변화.취약식물
# %%
gdf_merged['기후변화.취약식물'] = (
    gdf_merged['기후변화.취약식물']
    .astype('Int64')
)

gdf_merged['기후변화지표종'] = (
    gdf_merged['기후변화지표종']
    .astype('Int64')
)


# %%
print(gdf_merged['보정여부'].value_counts(dropna=False))
print(gdf_merged['기후변화지표종'].value_counts(dropna=False))
print(gdf_merged['기후변화.취약식물'].value_counts(dropna=False))
print(gdf_merged['멸종위기 등급'].value_counts(dropna=False))


# %%

# remove rows with 보정여부 = '1'
gdf_merged = gdf_merged[gdf_merged['보정여부'] == 0]

print(gdf_merged.shape)


# %%


# 3. Create Spatial Data
print("Creating GeoDataFrame...")
# Geometry: 경도 (Lon), 위도 (Lat)
geometry = gpd.points_from_xy(gdf_merged['경도'], gdf_merged['위도'])
gdf_merged2 = gpd.GeoDataFrame(gdf_merged, geometry=geometry, crs="EPSG:4326")

print(f"  > GeoDataFrame created. CRS: {gdf_merged2.crs}")

# 4. Export
print(f"Saving to GeoPackage: {output_path}")
try:
    gdf_merged2.to_file(output_path, driver='GPKG', layer='Plant_Data')
    print("  > Save successful!")
except Exception as e:
    print(f"Error saving GeoPackage: {e}")

print("--- Processing Complete ---")


# %%
