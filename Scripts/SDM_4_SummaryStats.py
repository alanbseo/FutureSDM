
import os
import glob
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

import sys
# Make sure we can import from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))



def main():
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) # Go up one level to root

    for run_mode in ["Constrained", "Unconstrained"]:
         process_stats_for_mode(run_mode, project_root)


def process_stats_for_mode(run_mode, project_root):
    output_dir = os.path.join(project_root, 'Output', 'SDM')
    input_pattern = os.path.join(output_dir, f"Stats_{run_mode}", "Scenario_Stats_MaxEnt_*.csv")
    
    print(f"\n{'='*50}")
    print(f"Aggregating Summary Stats for MODE: {run_mode.upper()}")
    print(f"{'='*50}")
    print(f"Looking for files in: {input_pattern}")

    # 1. List all Scenario_Stats files
    files = glob.glob(input_pattern)
    print(f"Found {len(files)} files to process.")

    all_data = []

    # 2. Iterate through files and read data
    for file_path in files:
        # Extract species name from filename: Scenario_Stats_{species}.csv
        filename = os.path.basename(file_path)
        species_name = filename.replace("Scenario_Stats_MaxEnt_", "").replace(".csv", "")
        
        try:
            df = pd.read_csv(file_path)
            df['species'] = species_name  # Add species column
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if not all_data:
        print("No data found.")
        return

    # 3. Concatenate all DataFrames
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Reorder columns to put species first for readability
    cols = ['species'] + [c for c in combined_df.columns if c != 'species']
    combined_df = combined_df[cols]

    # Save all species data
    all_species_output_path = os.path.join(output_dir, f"SDM_Analysis_AllSpecies_MaxEnt_{run_mode}.csv")
    combined_df.to_csv(all_species_output_path, index=False, encoding='utf-8-sig')
    print(f"Saved combined data to: {all_species_output_path}")

    # 4. Group by scenario and sum numeric columns
    # Identify numeric columns (excluding 'species' and 'scenario')
    numeric_cols = ['suitable_2020', 'suitable_2050', 'stable', 'gain', 'loss', 'unsuitable']
    
    # Ensure columns exist before determining numeric cols strictly
    existing_numeric_cols = [c for c in numeric_cols if c in combined_df.columns]
    
    # summary_df = combined_df.groupby('scenario')[existing_numeric_cols].sum().reset_index()
    #
    # # Save summary stats
    # summary_output_path = os.path.join(output_dir, "SDM_Analysis_MaxEnt_ByScenario.csv")
    # summary_df.to_csv(summary_output_path, index=False, encoding='utf-8-sig')
    # print(f"Saved summary stats to: {summary_output_path}")

    # 5. Generate Boxplots
    boxplot_dir = os.path.join(output_dir, f'Boxplots_{run_mode}')
    if not os.path.exists(boxplot_dir):
        os.makedirs(boxplot_dir)
        print(f"Created directory: {boxplot_dir}")

    # Set style
    sns.set(style="whitegrid")

    for col in existing_numeric_cols:
        plt.figure(figsize=(12, 8))
        
        # Create boxplot
        # We start with the full dataset (combined_df) because we want to see the distribution of species values per scenario
        sns.boxplot(x='scenario', y=col, data=combined_df)
        
        plt.title(f'Distribution of {col} by Scenario', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel(col)
        plt.tight_layout()
        
        plot_filename = f"Boxplot_{col}.png"
        plot_path = os.path.join(boxplot_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved boxplot: {plot_path}")

    # 6. Scenario Comparisons (Win Rates)
    print("Calculating scenario comparisons...")
    
    # Pivot for suitable_2050
    pivot_df = combined_df.pivot(index='species', columns='scenario', values='suitable_2050')
    
    def calculate_and_save_comparisons(df_pivot, out_path, group_name="All Species"):
        if df_pivot.empty:
            print(f"Skipping comparisons for {group_name} because no species data available.")
            return

        top_scenario = df_pivot.idxmax(axis=1)
        top_counts = top_scenario.value_counts().reset_index()
        top_counts.columns = ['Scenario', 'Top_Rank_Count']
        
        scenarios = df_pivot.columns.tolist()
        pairwise_wins = pd.DataFrame(index=scenarios, columns=scenarios, dtype=int)
        
        for s1 in scenarios:
            for s2 in scenarios:
                if s1 == s2:
                    pairwise_wins.loc[s1, s2] = 0
                else:
                    win_count = (df_pivot[s1] > df_pivot[s2]).sum()
                    pairwise_wins.loc[s1, s2] = win_count

        with open(out_path, 'w', encoding='utf-8-sig') as f:
            f.write("--- Top Rank Counts (Number of species where this scenario is best) ---\n")
            top_counts.to_csv(f, index=False)
            f.write("\n\n--- Pairwise Win Matrix (Row > Column count) ---\n")
            f.write("Scenario," + ",".join(scenarios) + "\n")
            for s1 in scenarios:
                row_vals = [str(pairwise_wins.loc[s1, s2]) for s2 in scenarios]
                f.write(f"{s1}," + ",".join(row_vals) + "\n")
                
        print(f"Saved scenario comparison summary to: {out_path}")

    # Calculate for All Species
    comparison_output_path = os.path.join(output_dir, f"SDM_Scenario_MaxEnt_Comparisons_{run_mode}.csv")
    calculate_and_save_comparisons(pivot_df, comparison_output_path, "All Species")
    
    # Calculate for each group
    try:
        import geopandas as gpd
        PLANT_GPKG_PATH = "../Data/Plant_Spatial_Data_05Feb2026.gpkg"
        print("Loading plant spatial data for group membership...")
        gdf = gpd.read_file(PLANT_GPKG_PATH)
        
        GROUP_CONFIG = [
            {"key": "endangered_1", "filter": ("멸종위기 등급", "1")},
            {"key": "endangered_2", "filter": ("멸종위기 등급", "2")},
            {"key": "climate_north", "filter": ("기후변화.취약식물", 1)},
            {"key": "climate_south", "filter": ("기후변화.취약식물", 2)},
        ]
        
        # In the GPKG the column for species is '종명' or whatever SDM_5 uses
        modelled_species = set(pivot_df.index.tolist())
        
        for cfg in GROUP_CONFIG:
            col, val = cfg["filter"]
            if col not in gdf.columns:
                continue
            # Some columns might have mixed types, standardizing
            sp_in_group = gdf.loc[gdf[col] == val, "종명"].dropna().unique().tolist()
            sp_in_group = [s for s in sp_in_group if s in modelled_species]
            
            if sp_in_group:
                group_pivot = pivot_df.loc[sp_in_group]
                group_out_path = os.path.join(output_dir, f"SDM_Scenario_MaxEnt_Comparisons_{cfg['key']}_{run_mode}.csv")
                calculate_and_save_comparisons(group_pivot, group_out_path, cfg['key'])
            else:
                print(f"No modelled species found for group {cfg['key']}.")

    except ImportError:
        print("geopandas not available. Skipping group-level comparisons.")
    except Exception as e:
        print(f"Error computing group-level comparisons: {e}")

if __name__ == "__main__":
    main()
