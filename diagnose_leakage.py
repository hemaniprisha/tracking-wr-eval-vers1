"""
Diagnostic Script: Identify Data Leakage in NFL Analysis
========================================================

Run this to find where the R² inflation is coming from.
"""

import pandas as pd
import sys

def diagnose(csv_path):
    print("="*70)
    print("DATA LEAKAGE DIAGNOSTIC")
    print("="*70)
    
    # Load the CSV
    df = pd.read_csv(csv_path)
    print(f"\n1. CSV STRUCTURE")
    print(f"   Total rows: {len(df)}")
    print(f"   Unique players: {df['player_name'].nunique()}")
    
    # Check for NFL performance columns
    print(f"\n2. NFL PERFORMANCE COLUMNS IN CSV")
    nfl_cols = ['targets_per_game', 'yards_per_game', 'receptions_per_game', 
                'recs_per_game', 'tds_per_game', 'catch_rate']
    
    for col in nfl_cols:
        if col in df.columns:
            non_null = df[col].notna().sum()
            print(f"   ✓ {col}: {non_null} non-null values")
        else:
            print(f"   ✗ {col}: NOT IN CSV")
    
    # KEY CHECK: Does targets_per_game exist in the raw CSV?
    print(f"\n3. CRITICAL CHECK: targets_per_game")
    if 'targets_per_game' in df.columns:
        non_null = df['targets_per_game'].notna().sum()
        print(f"   ⚠️  WARNING: targets_per_game EXISTS in CSV with {non_null} values!")
        print(f"   This is your problem! The CSV already has the target variable.")
        print(f"   The model is learning from data that includes the answer.")
        print(f"\n   SOLUTION: Remove targets_per_game from your CSV, or use a")
        print(f"   clean version of tracking_with_outcomes.csv that doesn't have NFL data.")
    else:
        print(f"   ✓ targets_per_game not in CSV (this is correct)")
    
    # Check what the prepare_modeling_data would see
    print(f"\n4. DATA AFTER min_plays FILTER (50+)")
    df_filtered = df[df['total_plays'] >= 50]
    print(f"   Rows with 50+ plays: {len(df_filtered)}")
    
    if 'targets_per_game' in df.columns:
        with_target = df_filtered['targets_per_game'].notna().sum()
        print(f"   Rows with targets_per_game: {with_target}")
    
    # Check for duplicate player names
    print(f"\n5. DUPLICATE PLAYER CHECK")
    duplicates = df['player_name'].value_counts()
    dups = duplicates[duplicates > 1]
    if len(dups) > 0:
        print(f"   Players with multiple rows: {len(dups)}")
        print(f"   Examples: {dups.head(5).to_dict()}")
    else:
        print(f"   ✓ No duplicate player names")
    
    print("\n" + "="*70)
    print("DIAGNOSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_leakage.py <path_to_csv>")
        print("Example: python diagnose_leakage.py data/tracking_with_outcomes.csv")
        sys.exit(1)
    
    diagnose(sys.argv[1])