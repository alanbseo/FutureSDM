"""
SDM_6_Plot_Group_Implications_Count.py
======================================
Plots implications using the COUNT of species (Winners vs Losers)
instead of percentage changes.
"""
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import json
try:
    with open("../sdm_config.json", "r", encoding="utf-8") as _f:
        _cfg = json.load(_f)
except:
    _cfg = {}


plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", rc={"font.family": "AppleGothic", "axes.unicode_minus": False})

OUTPUT_DIR = "../Output/SDM/Group_Implications_Count"
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
    gdf = gpd.read_file(PLANT_GPKG_PATH, ignore_geometry=True)
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
            
        df = pd.read_csv(csv_path)
        group_mapping = []
        for g_name, sp_list in groups_dict.items():
            temp = df[df['species'].isin(sp_list)].copy()
            temp['Group'] = g_name
            group_mapping.append(temp)
            
        df_grouped = pd.concat(group_mapping, ignore_index=True)
        df_grouped = df_grouped[df_grouped['suitable_2020'] > 0].copy()
        
        df_grouped['net_area'] = df_grouped['suitable_2050'] - df_grouped['suitable_2020']
        df_grouped['status'] = np.where(df_grouped['net_area'] > 0, '증가(Gain)', 
                               np.where(df_grouped['net_area'] < 0, '감소(Loss)', '유지(Stable)'))
        
        df_target = df_grouped[df_grouped['scenario'].isin(ALL_SCENARIOS)].copy()
        
        plot_diverging_counts(df_target, mode)
        plot_stacked_counts(df_target, mode)
        
    print(f"Count plots saved to {OUTPUT_DIR}")

def plot_diverging_counts(df, mode):
    counts = df.groupby(['Group', 'scenario', 'status'])['species'].count().reset_index()
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    width = 0.2
    x = np.arange(len(GROUP_ORDER))
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    
    for i, scen in enumerate(ALL_SCENARIOS):
        scen_data = counts[counts['scenario'] == scen]
        
        gain_counts = []
        loss_counts = []
        for g in GROUP_ORDER:
            g_gain = scen_data[(scen_data['Group'] == g) & (scen_data['status'] == '증가(Gain)')]['species'].sum()
            g_loss = scen_data[(scen_data['Group'] == g) & (scen_data['status'] == '감소(Loss)')]['species'].sum()
            gain_counts.append(g_gain)
            loss_counts.append(-g_loss)
            
        offset = offsets[i]
        color = PALETTE[scen]
        
        ax.bar(x + offset, gain_counts, width, label=f'{scen} (증가)', color=color, alpha=0.9, edgecolor='black')
        ax.bar(x + offset, loss_counts, width, label=f'{scen} (감소)', color=color, alpha=0.4, edgecolor='black', hatch='//')
        
        for j, (gain, loss) in enumerate(zip(gain_counts, loss_counts)):
            if gain > 0:
                ax.text(x[j] + offset, gain + 1, str(gain), ha='center', va='bottom', fontsize=9, fontweight='bold')
            if loss < 0:
                ax.text(x[j] + offset, loss - 1, str(abs(loss)), ha='center', va='top', fontsize=9, fontweight='bold')

    ax.axhline(0, color='black', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_ORDER, fontsize=12)
    ax.set_ylabel('종 수 (Count)', fontsize=12)
    ax.set_title(f'기후변화 시나리오별 서식지 증가/감소 종수 비교 ({mode})', fontsize=16, pad=20)
    
    yticks = ax.get_yticks()
    ax.set_yticklabels([str(abs(int(y))) for y in yticks])
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_Diverging_Counts_{mode}.png", dpi=300)
    plt.close()

def plot_stacked_counts(df, mode):
    fig, axes = plt.subplots(1, 5, figsize=(22, 6), sharey=True)
    
    for i, g in enumerate(GROUP_ORDER):
        ax = axes[i]
        g_df = df[df['Group'] == g]
        
        counts = pd.crosstab(g_df['scenario'], g_df['status'])
        cols = []
        for st in ['감소(Loss)', '유지(Stable)', '증가(Gain)']:
            if st in counts.columns: cols.append(st)
        counts = counts[cols]
        
        props = counts.div(counts.sum(axis=1), axis=0) * 100
        props = props.reindex(ALL_SCENARIOS).fillna(0)
        
        colors = {'감소(Loss)': '#ff9999', '유지(Stable)': '#cccccc', '증가(Gain)': '#99ff99'}
        plot_colors = [colors[c] for c in props.columns]
        
        props.plot(kind='bar', stacked=True, ax=ax, color=plot_colors, edgecolor='black', legend=False)
        
        ax.set_title(f'{g} (n={len(g_df.species.unique())})', fontsize=14)
        ax.set_xlabel('')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        for c in ax.containers:
            labels = [f'{w:.0f}%' if w > 5 else '' for w in c.datavalues]
            ax.bar_label(c, labels=labels, label_type='center', fontsize=11, fontweight='bold')
            
    axes[0].set_ylabel('종 비율 (%)', fontsize=12)
    fig.suptitle(f'식물 그룹별 증가/감소 종 비율 ({mode})', fontsize=16)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.95, 0.95))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(f"{OUTPUT_DIR}/02_Stacked_Ratio_{mode}.png", dpi=300)
    plt.close()

if __name__ == '__main__':
    main()
