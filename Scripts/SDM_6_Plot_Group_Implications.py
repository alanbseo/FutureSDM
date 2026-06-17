"""
SDM_6_Plot_Group_Implications.py
=================================
Generates analytical plots showing the implications of climate scenarios
on different plant groups, highlighting the trade-offs (e.g. Northern vs Southern).
Outputs for both Constrained and Unconstrained modes.
"""
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

import json
try:
    with open("../sdm_config.json", "r", encoding="utf-8") as _f:
        _cfg = json.load(_f)
except:
    _cfg = {}


plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", rc={"font.family": "AppleGothic", "axes.unicode_minus": False})

OUTPUT_DIR = "../Output/SDM/Group_Implications"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PLANT_GPKG_PATH = _cfg.get("PLANT_GPKG_PATH", "../Data/Plant_Spatial_Data_05Feb2026.gpkg")

ALL_SCENARIOS = ['BAU-SSP585', 'Climate-SSP126', 'Biodiversity-SSP126', 'Biodiversity-SSP245']
PALETTE = {
    'BAU-SSP585': '#e74c3c', 
    'Climate-SSP126': '#3498db', 
    'Biodiversity-SSP126': '#2ecc71', 
    'Biodiversity-SSP245': '#f1c40f'
}
GROUP_ORDER = ['멸종위기 1급', '멸종위기 2급', '북방계 (기후민감)', '남방계 (기후민감)', '일반종']

def main():
    print("Loading GeoPackage...")
    gdf = gpd.read_file(PLANT_GPKG_PATH, ignore_geometry=True)
    
    # We need to know all modelled species to extract '일반종'
    df_temp = pd.read_csv("../Output/SDM/SDM_Analysis_AllSpecies_MaxEnt_Constrained.csv")
    all_species = set(df_temp['species'].unique())
    
    g1 = set(gdf[gdf['멸종위기 등급'] == '1']['종명'].dropna().unique())
    g2 = set(gdf[gdf['멸종위기 등급'] == '2']['종명'].dropna().unique())
    gn = set(gdf[gdf['기후변화.취약식물'] == 1]['종명'].dropna().unique())
    gs = set(gdf[gdf['기후변화.취약식물'] == 2]['종명'].dropna().unique())
    general_species = list(all_species - (g1 | g2 | gn | gs))
    
    groups_dict = {
        '멸종위기 1급': list(g1),
        '멸종위기 2급': list(g2),
        '북방계 (기후민감)': list(gn),
        '남방계 (기후민감)': list(gs),
        '일반종': general_species
    }
    
    modes = [
        ("Constrained", "../Output/SDM/SDM_Analysis_AllSpecies_MaxEnt_Constrained.csv"),
        ("Unconstrained", "../Output/SDM/SDM_Analysis_AllSpecies_MaxEnt_Unconstrained.csv")
    ]

    for mode, csv_path in modes:
        if not os.path.exists(csv_path):
            continue
            
        print(f"\nProcessing {mode}...")
        df = pd.read_csv(csv_path)
        
        group_mapping = []
        for g_name, sp_list in groups_dict.items():
            temp = df[df['species'].isin(sp_list)].copy()
            temp['Group'] = g_name
            group_mapping.append(temp)
            
        df_grouped = pd.concat(group_mapping, ignore_index=True)
        df_grouped = df_grouped[df_grouped['suitable_2020'] > 0].copy()
        df_grouped['net_pct'] = (df_grouped['suitable_2050'] - df_grouped['suitable_2020']) / df_grouped['suitable_2020'] * 100
        
        df_target = df_grouped[df_grouped['scenario'].isin(ALL_SCENARIOS)].copy()
        
        plot_diverging_bar(df_target, mode)
        plot_boxplot(df_target, mode)
        plot_scatter(df_target, mode)
        
    print(f"\nAll plots saved to {OUTPUT_DIR}")

def plot_diverging_bar(df, mode):
    summary = df.groupby(['Group', 'scenario'])['net_pct'].median().reset_index()
    
    plt.figure(figsize=(15, 7))
    ax = sns.barplot(data=summary, x='Group', y='net_pct', hue='scenario', 
                     order=GROUP_ORDER, hue_order=ALL_SCENARIOS, palette=PALETTE)
    
    plt.axhline(0, color='black', linewidth=1.5, linestyle='--')
    plt.title(f'기후변화 시나리오별 그룹 중앙값(Median) 서식지 증감률 ({mode})', fontsize=16, pad=15)
    plt.xlabel('식물 그룹', fontsize=12)
    plt.ylabel('중앙값 서식지 증감률 (%)', fontsize=12)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=10, fontweight='bold')
        
    plt.legend(title='시나리오', loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_Diverging_Bar_Chart_{mode}.png", dpi=300)
    plt.close()

def plot_boxplot(df, mode):
    PALETTE_LIGHT = {'BAU-SSP585': '#ff9999', 'Climate-SSP126': '#99ccff', 'Biodiversity-SSP126': '#99ff99', 'Biodiversity-SSP245': '#ffe680'}
    PALETTE_DARK = {'BAU-SSP585': '#cc0000', 'Climate-SSP126': '#005c99', 'Biodiversity-SSP126': '#009900', 'Biodiversity-SSP245': '#b38f00'}
    
    plt.figure(figsize=(15, 8))
    
    ax = sns.boxplot(data=df, x='Group', y='net_pct', hue='scenario', 
                order=GROUP_ORDER, hue_order=ALL_SCENARIOS, palette=PALETTE_LIGHT, fliersize=4, linewidth=1.2)
    
    sns.stripplot(data=df, x='Group', y='net_pct', hue='scenario',
                  order=GROUP_ORDER, hue_order=ALL_SCENARIOS, palette=PALETTE_DARK, dodge=True, alpha=0.4, size=4, jitter=True, legend=False)
                  
    plt.axhline(0, color='black', linewidth=1.5, linestyle='--')
    plt.title(f'식물 그룹별 종단위 서식지 증감률 분포 ({mode})', fontsize=16, pad=15)
    plt.xlabel('식물 그룹', fontsize=12)
    plt.ylabel('서식지 증감률 (%)', fontsize=12)
    
    handles, labels = ax.get_legend_handles_labels() if 'ax' in locals() else plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles[:4], labels=ALL_SCENARIOS, title='시나리오', loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_Distribution_Boxplot_{mode}.png", dpi=300)
    plt.close()

def plot_scatter(df, mode):
    # Include all 5 groups in scatter plot
    df_climate = df.copy()
    
    df_climate['loss_pct'] = df_climate['loss'] / df_climate['suitable_2020'] * 100
    df_climate['gain_pct'] = df_climate['gain'] / df_climate['suitable_2020'] * 100
    
    g = sns.FacetGrid(df_climate, col="Group", col_wrap=3, hue="scenario", height=5, aspect=1.1,
                      col_order=GROUP_ORDER, hue_order=ALL_SCENARIOS, palette=PALETTE,
                      hue_kws={'marker': ['o', 's', 'D', '^']})
    
    g.map(sns.scatterplot, "loss_pct", "gain_pct", s=80, alpha=0.7, edgecolor='k')
    
    max_val = max(df_climate['loss_pct'].max(), df_climate['gain_pct'].max()) * 1.1 if len(df_climate) > 0 else 100
    for ax in g.axes.flat:
        ax.plot([0, max_val], [0, max_val], ls='--', color='gray', zorder=0, label='Net Zero (Loss=Gain)')
        ax.set_xlabel('서식지 상실 비율 (Loss %)', fontsize=12)
        ax.set_ylabel('새로운 서식지 확보 비율 (Gain %)', fontsize=12)
        ax.fill_between([0, max_val], [0, max_val], max_val * 2, color='#e0f7fa', alpha=0.2, zorder=-1)
        ax.fill_between([0, max_val], 0, [0, max_val], color='#ffebee', alpha=0.2, zorder=-1)
        ax.set_xlim(-5, max_val)
        ax.set_ylim(-5, max_val)
        
    g.fig.suptitle(f'그룹별: 상실(Loss) 대비 확보(Gain) 산점도 ({mode})', y=1.05, fontsize=16)
    g.add_legend(title='시나리오')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_Quadrant_Scatter_Plot_{mode}.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()
