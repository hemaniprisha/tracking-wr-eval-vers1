"""
Tracking-Derived Metrics Analysis for WR Profiling & Drafting

Analyzes college football tracking data to show why in-game metrics beat 
combine testing. Traditional scouting misses the real game: route discipline, 
separation ability, ball tracking, YAC creation. Tracking data shows the "why."
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class TrackingDataAnalyzer:
    """Analyzer for football tracking data with focus on NFL draft prediction."""
    
    def __init__(self, data_path=None):
        self.df = None
        self.feature_columns = []
        self.target_columns = []
        self.models = {}
        self.results = {}
        
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, path):
        """Load CSV and print basic dataset info."""
        self.df = pd.read_csv(path)
        print("="*80)
        print("Data Loaded Successfully")
        print("="*80)
        print(f"\nShape: {self.df.shape[0]} players × {self.df.shape[1]} metrics")
        print(f"Seasons: {self.df['season'].min()} - {self.df['season'].max()}")
        print(f"Players: {self.df['player_name'].nunique()}")
        print(f"Teams: {self.df['offense_team'].nunique()}")
        
        return self
    
    def explore_data(self):
        """Quick data quality check and summary stats."""
        print("\n" + "="*80)
        print("Data Quality Check")
        print("="*80)
        
        # Check for missing values
        print("\nMissing Data:")
        print("-" * 80)
        miss = self.df.isnull().sum()
        miss_pct = (miss / len(self.df)) * 100
        miss_df = pd.DataFrame({
            'Count': miss[miss > 0],
            'Pct': miss_pct[miss > 0]
        }).sort_values('Pct', ascending=False)
        
        if len(miss_df) > 0:
            print(f"\n{len(miss_df)} metrics with missing data:")
            print(miss_df.head(15))
            
            print("\nContext:")
            print("• Missing 'targeted' metrics → player wasn't thrown to much")
            print("• Missing 'separation_VMAN' → didn't face man or small sample")
            print("• Missing 'YACOE/CPOE' → no catch opps or tracking issues")
        else:
            print("✓ No missing data")
        
        # Key metrics summary
        print("\n\nKey Metrics")
        print("-" * 80)
        
        key = {
            'total_plays': 'Sample Size',
            'max_speed_99': 'Top Speed (mph)',
            'average_separation_99': 'Separation (yds)',
            'YACOE_MEAN': 'YAC Over Expected',
            'CPOE_MEAN': 'Catch % Over Expected',
            'cod_sep_generated_overall': 'Sep from Cuts'
        }
        
        for metric, desc in key.items():
            if metric in self.df.columns:
                d = self.df[metric].dropna()
                print(f"\n{desc} ({metric}):")
                print(f"  Mean: {d.mean():.2f} | Med: {d.median():.2f} | SD: {d.std():.2f}")
                print(f"  Range: [{d.min():.2f}, {d.max():.2f}]")
        
        # Volume tiers (super important for sample reliability)
        print("\n\nPlaying Time Distribution")
        print("-" * 80)
        bins = [0, 50, 100, 150, 200, 300, 1000]
        labels = ['<50 (Limited)', '50-100 (Rotational)', '100-150 (Starter)', 
                  '150-200 (Heavy)', '200-300 (Workhorse)', '300+ (Elite)']
        self.df['vol_tier'] = pd.cut(self.df['total_plays'], bins=bins, labels=labels)
        print(self.df['vol_tier'].value_counts().sort_index())
        
        print("\n→ Sample size matters! <50 plays = unreliable metrics")
        print("→ Elite prospects usually have 150+ plays")
        
        return self
    
    def engineer_features(self):
        """Create football-intelligent features for modeling."""
        print("\n" + "="*80)
        print("Feature Engineering")
        print("="*80)
        
        df = self.df.copy()
        
        # ATHLETICISM
        print("\n1. Athleticism")
        print("-" * 80)
        
        # Speed score - consistent top speed ability
        df['spd_score'] = (df['max_speed_99'] + df['max_speed_30_inf_yards_max']) / 2
        print("• Speed Score: ability to hit and sustain top speed")
        
        # Burst - acceleration events per play
        df['burst_rt'] = (df['high_acceleration_count_SUM'] / df['total_plays']) * 100
        print("• Burst Rate: high accel events per play")
        
        # First step quickness (0-10 yard speed)
        df['first_step'] = df['max_speed_0_10_yards_max']
        print("• First Step: speed in first 10 yards")
        
        # Decel ability for cuts
        df['brake_rt'] = (df['high_deceleration_count_SUM'] / df['total_plays']) * 100
        print("• Brake Rate: decel events per play")
        
        # ROUTE RUNNING
        print("\n\n2. Route Running")
        print("-" * 80)
        
        # Route tree versatility
        df['route_div'] = (
            (df['10ydplus_route_MEAN'] * 0.3) +
            (df['20ydplus_route_MEAN'] * 0.4) +
            (df['changedir_route_MEAN'] * 0.3)
        )
        print("• Route Diversity: versatility across route tree")
        
        # Consistent separation
        df['sep_consistency'] = df['average_separation_99']
        print("• Separation Consistency: 99th pct separation")
        
        # Man coverage
        df['man_win_rt'] = df['separation_at_throw_VMAN']
        print("• Man Win Rate: separation vs man coverage")
        
        # Ball tracking
        df['tracking'] = df['separation_change_postthrow_MEAN']
        print("• Tracking: separation change after throw")
        
        # CONTESTED CATCHES
        print("\n\n3. Contested Catches")
        print("-" * 80)
        
        # Tight window success
        df['contested_rt'] = np.where(
            df['tight_window_at_throw_SUM'] > 0,
            (df['targeted_tightwindow_catch_SUM'] / df['tight_window_at_throw_SUM']) * 100,
            np.nan
        )
        print("• Contested Rate: success in tight windows (<2 yds)")
        
        # Target share in traffic
        df['tight_tgt_pct'] = np.where(
            df['total_plays'] > 0,
            (df['tight_window_at_throw_SUM'] / df['total_plays']) * 100,
            np.nan
        )
        print("• Tight Target %: how often QB trusts them in traffic")
        
        # PLAYMAKING
        print("\n\n4. Playmaking")
        print("-" * 80)
        
        df['yac_ability'] = df['YACOE_MEAN']
        print("• YAC Ability: yards after catch over expected")
        
        df['qb_friendly'] = df['CPOE_MEAN']
        print("• QB-Friendly: completion % over expected")
        
        # CHANGE OF DIRECTION
        print("\n\n5. Change of Direction")
        print("-" * 80)
        
        # 90 degree cuts (slants, outs)
        df['sharp_cuts'] = (
            df['cod_top5_speed_entry_avg_90_'] + df['cod_top5_speed_exit_avg_90_']
        ) / 2
        print("• Sharp Cuts: speed through 90° cuts")
        
        # 180 degree (comebacks, curls)
        df['route_bend'] = (
            df['cod_top5_speed_entry_avg_180_'] + df['cod_top5_speed_exit_avg_180_']
        ) / 2
        print("• Route Bend: speed through 180° cuts")
        
        df['cut_sep'] = df['cod_sep_generated_overall']
        print("• Cut Separation: yards created from route breaks")
        
        # WORKLOAD
        print("\n\n6. Workload")
        print("-" * 80)
        
        df['high_vol'] = (df['total_plays'] >= 150).astype(int)
        print("• High Volume: 150+ plays (starter/feature player)")
        
        df['dist_per_play'] = df['play_distance_SUM'] / df['total_plays']
        print("• Distance/Play: route depth tendency")
        
        df['tot_explosive'] = (
            df['high_acceleration_count_SUM'] + df['high_deceleration_count_SUM']
        )
        df['explosive_rt'] = (df['tot_explosive'] / df['total_plays']) * 100
        print("• Explosive Rate: combined accel/decel events")
        
        # COMPOSITE SCORES
        print("\n\n7. Composite Ratings")
        print("-" * 80)
        
        # Athleticism score (0-100)
        spd_norm = (df['spd_score'] - df['spd_score'].mean()) / df['spd_score'].std()
        burst_norm = (df['burst_rt'] - df['burst_rt'].mean()) / df['burst_rt'].std()
        df['ath_score'] = ((spd_norm + burst_norm) / 2) * 10 + 50
        df['ath_score'] = df['ath_score'].clip(0, 100)
        print("• Athleticism Score: speed + burst (0-100)")
        
        # Route running grade
        rr_metrics = ['route_div', 'sep_consistency', 'man_win_rt']
        rr_avail = [c for c in rr_metrics if c in df.columns]
        if rr_avail:
            rr_norm = df[rr_avail].apply(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )
            df['rr_grade'] = (rr_norm.mean(axis=1) * 10 + 50).clip(0, 100)
            print("• Route Running Grade: separation + versatility (0-100)")
        
        # Overall prospect score
        key_f = ['ath_score', 'rr_grade', 'yac_ability']
        avail_f = [f for f in key_f if f in df.columns]
        if avail_f:
            df['prospect_score'] = df[avail_f].mean(axis=1)
            print("• Prospect Score: holistic composite")
        
        new_cols = len([c for c in df.columns if c not in self.df.columns])
        print(f"\n✓ Created {new_cols} new features")
        
        self.df = df
        return self
    
    def identify_archetypes(self, n_clusters=5):
        """Cluster players into receiver archetypes."""
        print("\n" + "="*80)
        print("Player Archetype Identification")
        print("="*80)
        
        # Use engineered features for clustering
        cluster_f = ['spd_score', 'burst_rt', 'sep_consistency']
        
        # Check what's actually available
        avail = []
        for f in cluster_f:
            if f in self.df.columns:
                n_valid = self.df[f].notna().sum()
                if n_valid > 0:
                    avail.append(f)
                    print(f"✓ {f}: {n_valid} values")
                else:
                    print(f"✗ {f}: no data")
            else:
                print(f"✗ {f}: not found")
        
        if len(avail) < 2:
            print("\n→ Need at least 2 features with data. Skipping.")
            return self
        
        # Get complete cases
        clust_data = self.df[avail + ['player_name']].dropna()
        
        print(f"\nClustering {len(clust_data)} players on {len(avail)} features...")
        
        if len(clust_data) < n_clusters:
            print(f"Only {len(clust_data)} players with complete data")
            n_clusters = min(3, max(2, len(clust_data) // 10))
            print(f"Reducing to {n_clusters} clusters")
        
        if len(clust_data) < 10:
            print("Too few players. Skipping clustering.")
            return self
        
        # Standardize and cluster
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(clust_data[avail])
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clust_data['archetype'] = kmeans.fit_predict(X_scaled)
        
        # Label archetypes
        print("\nReceiver Archetypes:\n")
        
        arch_names = {
            0: "Deep Threat Burners",
            1: "Route Technicians",
            2: "YAC Monsters",
            3: "Complete Receivers",
            4: "Possession Specialists"
        }
        
        for cid in range(n_clusters):
            grp = clust_data[clust_data['archetype'] == cid]
            
            print(f"\n{arch_names.get(cid, f'Archetype {cid}')} ({len(grp)} players)")
            print("-" * 60)
            
            for f in cluster_f:
                if f in grp.columns:
                    print(f"  {f:30s}: {grp[f].mean():6.2f}")
            
            samples = grp['player_name'].head(3).tolist()
            print(f"  Examples: {', '.join(samples)}")
        
        # Merge back
        self.df = self.df.merge(
            clust_data[['player_name', 'archetype']],
            on='player_name',
            how='left'
        )
        
        return self
    
    def prepare_modeling_data(self, target='auto', min_plays=50):
        """Prep clean dataset for modeling with draft/production targets."""
        print("\n" + "="*80)
        print("Preparing Modeling Data")
        print("="*80)
        
        # Filter by sample size
        df_mdl = self.df[self.df['total_plays'] >= min_plays].copy()
        print(f"\n✓ {len(df_mdl)} players with {min_plays}+ plays")
        
        # Define potential features (using new short names)
        pot_feats = [
            'spd_score', 'burst_rt', 'first_step', 'brake_rt',
            'max_speed_99', 'route_div', 'sep_consistency',
            'sharp_cuts', 'route_bend', 'cut_sep',
            'changedir_route_MEAN', 'cod_sep_generated_overall',
            'dist_per_play', 'explosive_rt', 'total_plays'
        ]
        
        # Check availability
        self.feature_columns = []
        for f in pot_feats:
            if f in df_mdl.columns:
                pct = df_mdl[f].notna().sum() / len(df_mdl) * 100
                if pct >= 30:
                    self.feature_columns.append(f)
                    print(f"✓ {f:35s}: {pct:5.1f}%")
        
        # Target priority: draft > production > fallback
        pot_tgts = [
            ('draft_capital', 'Draft Capital (best - tracking predicts draft)'),
            ('rec_yards', 'College Yards (good - tracking predicts production)'),
            ('production_score', 'Production Score (good - composite)'),
            ('cod_sep_generated_overall', 'Separation from Cuts (fallback)'),
        ]
        
        sel_tgt = None
        if target != 'auto' and target in df_mdl.columns:
            n = df_mdl[target].notna().sum()
            pct = n / len(df_mdl) * 100
            if pct >= 30:
                sel_tgt = target
                print(f"\nUsing specified target: {target} ({pct:.1f}%)")
        
        if sel_tgt is None:
            for tgt, desc in pot_tgts:
                if tgt in df_mdl.columns:
                    n = df_mdl[tgt].notna().sum()
                    pct = n / len(df_mdl) * 100
                    print(f"  {tgt:30s}: {n:4,} ({pct:5.1f}%)")
                    if pct >= 30 and sel_tgt is None:
                        sel_tgt = tgt
                        target = tgt
                        print(f"  → SELECTED: {desc}")
        
        if sel_tgt is None:
            print("\n✗ No valid target found")
            return None, None
        
        # Remove target from features (no leakage)
        if target in self.feature_columns:
            self.feature_columns.remove(target)
        
        # Prevent leakage for related features
        if target == 'sep_consistency' and 'average_separation_99' in self.feature_columns:
            self.feature_columns.remove('average_separation_99')
        
        print(f"\nFinal: {len(self.feature_columns)} features → {target}")
        
        # Impute missing values
        imp = SimpleImputer(strategy='median')
        
        X = df_mdl[self.feature_columns].copy()
        X_imp = pd.DataFrame(
            imp.fit_transform(X),
            columns=self.feature_columns,
            index=X.index
        )
        
        y = df_mdl[target].copy()
        y_imp = imp.fit_transform(y.values.reshape(-1, 1)).ravel()
        y = pd.Series(y_imp, index=y.index)
        
        # Filter to complete cases
        valid = ~(X_imp.isnull().any(axis=1) | y.isnull())
        X_clean = X_imp[valid]
        y_clean = y[valid]
        
        print(f"\n✓ Ready: {len(X_clean)} players")
        print(f"  Target: {target}")
        print(f"  Mean: {y_clean.mean():.3f} | Med: {y_clean.median():.3f}")
        
        return X_clean, y_clean
    
    def build_models(self, X, y, test_size=0.25):
        """Train and evaluate gradient boosting model."""
        print("\n" + "="*80)
        print("Building Models")
        print("="*80)
        
        # Split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"\nTrain: {len(X_tr)} | Test: {len(X_te)}")
        
        # GBM
        print("\nTraining Gradient Boosting...")
        gb = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
        gb.fit(X_tr, y_tr)
        
        # CV
        cv = cross_val_score(gb, X_tr, y_tr, cv=5, scoring='r2')
        print(f"CV R² (5-fold): {cv.mean():.3f} ± {cv.std():.3f}")
        
        # Test performance
        y_pred_tr = gb.predict(X_tr)
        y_pred_te = gb.predict(X_te)
        
        r2_tr = r2_score(y_tr, y_pred_tr)
        r2_te = r2_score(y_te, y_pred_te)
        mae = mean_absolute_error(y_te, y_pred_te)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred_te))
        
        print(f"\nPerformance:")
        print(f"  Train R²: {r2_tr:.3f}")
        print(f"  Test R²: {r2_te:.3f}")
        print(f"  MAE: {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        
        # Store
        self.models['gradient_boosting'] = gb
        self.results['predictions'] = {
            'X_train': X_tr, 'X_test': X_te,
            'y_train': y_tr, 'y_test': y_te,
            'y_pred_train': y_pred_tr, 'y_pred_test': y_pred_te
        }
        self.results['metrics'] = {
            'train_r2': r2_tr, 'test_r2': r2_te,
            'test_mae': mae, 'test_rmse': rmse,
            'cv_mean': cv.mean(), 'cv_std': cv.std()
        }
        
        # Feature importance
        feat_imp = pd.DataFrame({
            'feature': X.columns,
            'importance': gb.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.results['feature_importance'] = feat_imp
        
        print("\nTop 10 Features:")
        print(feat_imp.head(10).to_string(index=False))
        
        return self


if __name__ == "__main__":
    print("""
    College Tracking → Draft Prediction Pipeline
    """)
    
    analyzer = TrackingDataAnalyzer()
    
    print("\nLoad data:")
    print("  analyzer.load_data('data.csv')")
    print("\nRun pipeline:")
    print("  analyzer.explore_data()")
    print("  analyzer.engineer_features()")
    print("  analyzer.identify_archetypes()")
    print("  X, y = analyzer.prepare_modeling_data()")
    print("  analyzer.build_models(X, y)")