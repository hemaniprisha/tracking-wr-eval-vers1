"""
Tracking-Derived Metrics Analysis for WR Profiling & Drafting
=============================================================

This pipeline analyzes college football tracking data to demonstrate the 
predictive power of in-game metrics vs traditional combine testing.

Football Context:
- Traditional scouting relies on combine metrics (40-time, vertical, etc.)
- These static tests miss what matters: route discipline, separation ability, 
  ball tracking, YAC creation, and functional athleticism in live game situations
- Tracking data captures the "why" behind production
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

# Set style for professional visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class TrackingDataAnalyzer:
    """
    Comprehensive analyzer for football tracking data with football-context storytelling.
    """
    
    def __init__(self, data_path=None):
        """Initialize analyzer with optional data path."""
        self.df = None
        self.feature_columns = []
        self.target_columns = []
        self.models = {}
        self.results = {}
        
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path):
        """Load and perform initial data exploration."""
        self.df = pd.read_csv(data_path)
        print("="*80)
        print("DATA LOADED SUCCESSFULLY")
        print("="*80)
        print(f"\nDataset Shape: {self.df.shape[0]} players × {self.df.shape[1]} metrics")
        print(f"Seasons Covered: {self.df['season'].min()} - {self.df['season'].max()}")
        print(f"Unique Players: {self.df['player_name'].nunique()}")
        print(f"Teams Represented: {self.df['offense_team'].nunique()}")
        
        return self
    
    def explore_data(self):
        """
        Comprehensive data exploration with football context.
        """
        print("\n" + "="*80)
        print("DATA EXPLORATION & QUALITY CHECK")
        print("="*80)
        
        # Missing data analysis
        print("\n MISSING DATA ANALYSIS")
        print("-" * 80)
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing[missing > 0],
            'Percentage': missing_pct[missing > 0]
        }).sort_values('Percentage', ascending=False)
        
        if len(missing_df) > 0:
            print(f"\n {len(missing_df)} metrics have missing data:")
            print(missing_df.head(15))
            
            # Football context for missing data
            print("\nFOOTBALL CONTEXT:")
            print("Missing 'targeted' metrics → Player wasn't thrown to (limited targets)")
            print("Missing 'separation_VMAN' → Didn't face man coverage or insufficient sample")
            print("Missing 'YACOE/CPOE' → No catch opportunities or tracking limitations")
        else:
            print("No missing data detected!")
        
        # Summary statistics for key metrics
        print("\n\nKEY METRICS SUMMARY")
        print("-" * 80)
        
        key_metrics = {
            'total_plays': 'Opportunity/Sample Size',
            'max_speed_99': 'Consistent Top Speed (mph)',
            'average_separation_99': 'Separation Consistency (yards)',
            'YACOE_MEAN': 'Yards After Catch Over Expected',
            'CPOE_MEAN': 'Completion % Over Expected',
            'cod_sep_generated_overall': 'Separation from Cuts'
        }
        
        for metric, description in key_metrics.items():
            if metric in self.df.columns:
                data = self.df[metric].dropna()
                print(f"\n{description} ({metric}):")
                print(f"  Mean: {data.mean():.2f} | Median: {data.median():.2f} | Std: {data.std():.2f}")
                print(f"  Range: [{data.min():.2f}, {data.max():.2f}]")
        
        # Volume distribution
        print("\n\nPLAYING TIME DISTRIBUTION")
        print("-" * 80)
        play_bins = [0, 50, 100, 150, 200, 300, 1000]
        play_labels = ['<50 (Limited)', '50-100 (Rotational)', '100-150 (Starter)', 
                       '150-200 (Heavy)', '200-300 (Workhorse)', '300+ (Elite Volume)']
        self.df['volume_tier'] = pd.cut(self.df['total_plays'], bins=play_bins, labels=play_labels)
        print(self.df['volume_tier'].value_counts().sort_index())
        
        print("\nFOOTBALL CONTEXT:")
        print("Sample size matters! Players with <50 plays have unreliable metrics.")
        print("Elite prospects typically have 150+ plays (heavy usage in college).")
        
        return self
    
    def engineer_features(self):
        """
        Engineer football-intelligent features with clear strategic context.
        """
        print("\n" + "="*80)
        print("FEATURE ENGINEERING: BUILDING FOOTBALL-INTELLIGENT METRICS")
        print("="*80)
        
        df = self.df.copy()
        
        # ===================================================================
        # 1. ATHLETICISM PROFILE
        # ===================================================================
        print("\n1. ATHLETICISM PROFILE")
        print("-" * 80)
        
        # Speed Score (consistent elite speed)
        df['speed_score'] = (df['max_speed_99'] + df['max_speed_30_inf_yards_max']) / 2
        print("✓ Speed Score: Ability to reach and sustain top speed (deep threat indicator)")
        
        # Burst Efficiency (acceleration per play)
        df['burst_rate'] = (df['high_acceleration_count_SUM'] / df['total_plays']) * 100
        print("✓ Burst Rate: High acceleration events per play (route explosion)")
        
        # Top-End Acceleration (0-10 yard explosion)
        df['first_step_quickness'] = df['max_speed_0_10_yards_max']
        print("✓ First Step Quickness: Speed in first 10 yards (release & separation)")
        
        # Functional Agility (decel ability for cuts)
        df['brake_rate'] = (df['high_deceleration_count_SUM'] / df['total_plays']) * 100
        print("✓ Brake Rate: Deceleration events per play (cut sharpness)")
        
        # ===================================================================
        # 2. ROUTE RUNNING INTELLIGENCE
        # ===================================================================
        print("\n\n2. ROUTE RUNNING INTELLIGENCE")
        print("-" * 80)
        
        # Route Diversity Index (versatility)
        df['route_diversity'] = (
            (df['10ydplus_route_MEAN'] * 0.3) +  # Medium depth
            (df['20ydplus_route_MEAN'] * 0.4) +  # Deep threat
            (df['changedir_route_MEAN'] * 0.3)   # Route complexity
        )
        print("Route Diversity: Versatility across route tree (vs one-dimensional)")
        
        # Separation Consistency (can they get open reliably?)
        df['separation_consistency'] = df['average_separation_99']
        print("Separation Consistency: 99th percentile separation (elite vs lucky)")
        
        # Man Coverage Dominance
        df['man_coverage_win_rate'] = df['separation_at_throw_VMAN']
        print("Man Coverage Win Rate: Separation vs man (toughest coverage)")
        
        # Tracking Ability (closing separation post-throw)
        df['tracking_skill'] = df['separation_change_postthrow_MEAN']
        print("Tracking Skill: Separation change after throw (ball tracking)")
        
        # ===================================================================
        # 3. CONTESTED CATCH PROFILE
        # ===================================================================
        print("\n\n3. CONTESTED CATCH ABILITY")
        print("-" * 80)
        
        # Tight window success rate
        df['contested_catch_rate'] = np.where(
            df['tight_window_at_throw_SUM'] > 0,
            (df['targeted_tightwindow_catch_SUM'] / df['tight_window_at_throw_SUM']) * 100,
            np.nan
        )
        print("Contested Catch Rate: Success in tight coverage (<2 yards separation)")
        
        # Target share in traffic
        df['tight_window_target_pct'] = np.where(
            df['total_plays'] > 0,
            (df['tight_window_at_throw_SUM'] / df['total_plays']) * 100,
            np.nan
        )
        print("✓ Tight Window Target %: How often QB trusts them in traffic")
        
        # ===================================================================
        # 4. PLAYMAKING ABILITY (VALUE METRICS)
        # ===================================================================
        print("\n\n4. PLAYMAKING & VALUE CREATION")
        print("-" * 80)
        
        # YAC ability (already in data as YACOE)
        df['yac_ability'] = df['YACOE_MEAN']
        print("YAC Ability: Yards after catch over expected (playmaking)")
        
        # QB-Friendly (completion percentage over expected)
        df['qb_friendly'] = df['CPOE_MEAN']
        print("QB-Friendly Rating: Completion % over expected (reliable hands)")
        
        # ===================================================================
        # 5. CHANGE OF DIRECTION (COD) ANALYSIS
        # ===================================================================
        print("\n\n5. CHANGE OF DIRECTION MASTERY")
        print("-" * 80)
        
        # Sharp cut ability (90 degree cuts - slants, digs, outs)
        df['sharp_cut_ability'] = (
            df['cod_top5_speed_entry_avg_90_'] + df['cod_top5_speed_exit_avg_90_']
        ) / 2
        print("✓ Sharp Cut Ability: Speed through 90° cuts (slants, outs, digs)")
        
        # Route bending (180 degree - comebacks, curls)
        df['route_bend_ability'] = (
            df['cod_top5_speed_entry_avg_180_'] + df['cod_top5_speed_exit_avg_180_']
        ) / 2
        print("✓ Route Bend Ability: Speed through 180° cuts (comebacks, curls)")
        
        # Separation from cuts (already in data)
        df['cut_separation'] = df['cod_sep_generated_overall']
        print("✓ Cut Separation: Yards of separation created from route breaks")
        
        # ===================================================================
        # 6. WORKLOAD & DURABILITY
        # ===================================================================
        print("\n\n6. WORKLOAD & USAGE INDICATORS")
        print("-" * 80)
        
        # Volume tier (already created in exploration)
        df['high_volume_player'] = (df['total_plays'] >= 150).astype(int)
        print("✓ High Volume Flag: 150+ plays (starter/feature player)")
        
        # Distance per play (efficiency + usage type)
        df['distance_per_play'] = df['play_distance_SUM'] / df['total_plays']
        print("Distance per Play: Route depth tendency (deep vs short game)")
        
        # Total acceleration events (explosive play frequency)
        df['total_explosive_events'] = (
            df['high_acceleration_count_SUM'] + df['high_deceleration_count_SUM']
        )
        df['explosive_rate'] = (df['total_explosive_events'] / df['total_plays']) * 100
        print("Explosive Rate: Combined accel/decel events (dynamic playmaking)")
        
        # ===================================================================
        # 7. COMPOSITE SCORES (HOLISTIC RATINGS)
        # ===================================================================
        print("\n\n7. COMPOSITE RATING SYSTEMS")
        print("-" * 80)
        
        # Elite Athleticism Score (0-100 scale)
        speed_norm = (df['speed_score'] - df['speed_score'].mean()) / df['speed_score'].std()
        burst_norm = (df['burst_rate'] - df['burst_rate'].mean()) / df['burst_rate'].std()
        df['athleticism_score'] = ((speed_norm + burst_norm) / 2) * 10 + 50
        df['athleticism_score'] = df['athleticism_score'].clip(0, 100)
        print(" Athleticism Score: Combined speed + burst (0-100 scale)")
        
        # Route Running Grade
        route_metrics = ['route_diversity', 'separation_consistency', 'man_coverage_win_rate']
        route_cols_available = [col for col in route_metrics if col in df.columns]
        if route_cols_available:
            route_norm = df[route_cols_available].apply(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )
            df['route_running_grade'] = (route_norm.mean(axis=1) * 10 + 50).clip(0, 100)
            print("✓ Route Running Grade: Separation + versatility (0-100 scale)")
        
        # Overall Prospect Score
        key_features = ['athleticism_score', 'route_running_grade', 'yac_ability']
        available_features = [f for f in key_features if f in df.columns]
        if available_features:
            df['prospect_score'] = df[available_features].mean(axis=1)
            print("✓ Prospect Score: Holistic evaluation (weighted composite)")
        
        print("\n" + "="*80)
        print(f"FEATURE ENGINEERING COMPLETE: {len([c for c in df.columns if c not in self.df.columns])} new features created")
        print("="*80)
        
        self.df = df
        return self
    
    def identify_archetypes(self, n_clusters=5):
        """
        Identify distinct receiver archetypes using clustering.
        """
        print("\n" + "="*80)
        print("PLAYER ARCHETYPE IDENTIFICATION")
        print("="*80)
        
        # Select features for clustering - use only features we created
        cluster_features = [
            'speed_score', 'burst_rate', 'separation_consistency'
        ]
        
        # Check which features actually exist and have data
        available_features = []
        for feat in cluster_features:
            if feat in self.df.columns:
                non_null = self.df[feat].notna().sum()
                if non_null > 0:
                    available_features.append(feat)
                    print(f"{feat}: {non_null} non-null values")
                else:
                    print(f"{feat}: no data available")
            else:
                print(f"⚠️  {feat}: column not found")
        
        if len(available_features) < 2:
            print("\n Insufficient features for clustering. Skipping archetype analysis.")
            print("   (Need at least 2 features with data)")
            return self
        
        # Filter to complete cases for available features
        cluster_data = self.df[available_features + ['player_name']].dropna()
        
        print(f"\nClustering {len(cluster_data)} players with complete data on {len(available_features)} features...")
        
        if len(cluster_data) < n_clusters:
            print(f"\n Only {len(cluster_data)} players with complete data.")
            print(f"   Need at least {n_clusters} for {n_clusters} clusters.")
            print("   Reducing to {min(3, len(cluster_data))} clusters...")
            n_clusters = min(3, max(2, len(cluster_data) // 10))
        
        if len(cluster_data) < 10:
            print("\n Too few players for reliable clustering. Skipping.")
            return self
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(cluster_data[available_features])
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_data['archetype'] = kmeans.fit_predict(X_scaled)
        
        # Analyze each cluster
        print("\n RECEIVER ARCHETYPES IDENTIFIED:\n")
        
        archetype_names = {
            0: "Deep Threat Burners",
            1: "Route Technicians", 
            2: "YAC Monsters",
            3: "Complete Receivers",
            4: "Possession Specialists"
        }
        
        for cluster_id in range(n_clusters):
            cluster_players = cluster_data[cluster_data['archetype'] == cluster_id]
            
            print(f"\n{archetype_names.get(cluster_id, f'Archetype {cluster_id}')} ({len(cluster_players)} players)")
            print("-" * 60)
            
            # Cluster characteristics
            for feature in cluster_features:
                mean_val = cluster_players[feature].mean()
                print(f"  {feature:30s}: {mean_val:6.2f}")
            
            # Sample players
            sample_players = cluster_players['player_name'].head(3).tolist()
            print(f"  Example players: {', '.join(sample_players)}")
        
        # Merge back to main dataframe
        self.df = self.df.merge(
            cluster_data[['player_name', 'archetype']], 
            on='player_name', 
            how='left'
        )
        
        return self
    def prepare_modeling_data(self, target='auto', min_plays=50):
        """
        Prepare clean dataset for modeling with draft/production targets.
        """
        print("\n" + "="*80)
        print("PREPARING DATA FOR PREDICTIVE MODELING")
        print("="*80)
        
        # Filter to players with sufficient sample size
        df_model = self.df[self.df['total_plays'] >= min_plays].copy()
        print(f"\n✓ Filtered to {len(df_model)} players with {min_plays}+ plays")
        
        # Define feature groups
        potential_features = [
            'speed_score', 'burst_rate', 'first_step_quickness', 'brake_rate',
            'max_speed_99', 'route_diversity', 'separation_consistency', 
            'sharp_cut_ability', 'route_bend_ability', 'cut_separation',
            'changedir_route_MEAN', 'cod_sep_generated_overall',
            'distance_per_play', 'explosive_rate', 'total_plays'
        ]
        
        # Check which features have data
        self.feature_columns = []
        for feat in potential_features:
            if feat in df_model.columns:
                non_null_pct = df_model[feat].notna().sum() / len(df_model) * 100
                if non_null_pct >= 30:
                    self.feature_columns.append(feat)
                    print(f"✓ {feat:35s}: {non_null_pct:5.1f}% available")
        
        # PRIORITIZE DRAFT/PRODUCTION TARGETS
        potential_targets = [
            ('draft_capital', 'Draft Capital (BEST - shows tracking predicts draft)'),
            ('rec_yards', 'College Yards (GOOD - shows tracking predicts production)'),
            ('production_score', 'Production Score (GOOD - composite metric)'),
            ('cod_sep_generated_overall', 'Separation from Cuts (FALLBACK)'),
        ]
        
        selected_target = None
        if target != 'auto' and target in df_model.columns:
            non_null = df_model[target].notna().sum()
            pct = non_null / len(df_model) * 100
            if pct >= 30:
                selected_target = target
                print(f"\n Using specified target: {target} ({pct:.1f}% data)")
        
        if selected_target is None:
            for tgt, desc in potential_targets:
                if tgt in df_model.columns:
                    non_null = df_model[tgt].notna().sum()
                    pct = non_null / len(df_model) * 100
                    print(f"   {tgt:30s}: {non_null:4,} ({pct:5.1f}%)")
                    if pct >= 30 and selected_target is None:
                        selected_target = tgt
                        target = tgt
                        print(f"   → SELECTED: {desc}")
        
        if selected_target is None:
            print("\n No valid target found!")
            return None, None
        
        # Remove target from features
        if target in self.feature_columns:
            self.feature_columns.remove(target)
        
        # Prevent data leakage
        if target == 'separation_consistency' and 'average_separation_99' in self.feature_columns:
            self.feature_columns.remove('average_separation_99')
        
        print(f"\n Final: {len(self.feature_columns)} features → {target}")
        
        # Handle missing values
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        
        X = df_model[self.feature_columns].copy()
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=self.feature_columns,
            index=X.index
        )
        
        y = df_model[target].copy()
        y_imputed = imputer.fit_transform(y.values.reshape(-1, 1)).ravel()
        y = pd.Series(y_imputed, index=y.index)
        
        valid_idx = ~(X_imputed.isnull().any(axis=1) | y.isnull())
        X_clean = X_imputed[valid_idx]
        y_clean = y[valid_idx]
        
        print(f"\n Ready: {len(X_clean)} players")
        print(f"   Target: {target}")
        print(f"   Mean: {y_clean.mean():.3f} | Median: {y_clean.median():.3f}")
        
        return X_clean, y_clean    

    def build_models(self, X, y, test_size=0.25):
        """
        Build and evaluate predictive models.
        """
        print("\n" + "="*80)
        print("BUILDING PREDICTIVE MODELS")
        print("="*80)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"\n Training set: {len(X_train)} players")
        print(f" Test set: {len(X_test)} players")
        
        # Model 1: Gradient Boosting
        print("\n Training Gradient Boosting Regressor...")
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(gb_model, X_train, y_train, cv=5, scoring='r2')
        print(f"  Cross-validation R² (5-fold): {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        # Test set performance
        y_pred_train = gb_model.predict(X_train)
        y_pred_test = gb_model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        print(f"\n MODEL PERFORMANCE:")
        print(f"  Training R²: {train_r2:.3f}")
        print(f"  Test R²: {test_r2:.3f}")
        print(f"  Test MAE: {test_mae:.3f}")
        print(f"  Test RMSE: {test_rmse:.3f}")
        
        # Store results
        self.models['gradient_boosting'] = gb_model
        self.results['predictions'] = {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'y_pred_train': y_pred_train, 'y_pred_test': y_pred_test
        }
        self.results['metrics'] = {
            'train_r2': train_r2, 'test_r2': test_r2,
            'test_mae': test_mae, 'test_rmse': test_rmse,
            'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()
        }
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': gb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.results['feature_importance'] = feature_importance
        
        print("\n TOP 10 MOST IMPORTANT FEATURES:")
        print(feature_importance.head(10).to_string(index=False))
        
        return self


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                                                                          ║
    ║   TRACKING-DERIVED METRICS FOR WR EVALUATION & DRAFT STRATEGY            ║
    ║                                                                          ║
    ║   From Static Testing to Dynamic Performance                             ║
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize analyzer
    analyzer = TrackingDataAnalyzer()
    
    print("\n To begin analysis, load your data:")
    print("   analyzer.load_data('your_data.csv')")
    print("\n Then run the full pipeline:")
    print("   analyzer.explore_data()")
    print("   analyzer.engineer_features()")
    print("   analyzer.identify_archetypes()")
    print("   X, y = analyzer.prepare_modeling_data()")
    print("   analyzer.build_models(X, y)")