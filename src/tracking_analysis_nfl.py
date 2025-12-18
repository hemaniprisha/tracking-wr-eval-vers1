"""
Tracking-Derived College Football Metrics versus NFL Rookie On-Field Performance

This pipeline analyzes college football player tracking data and evaluates
how in-game movement, separation, and athletic metrics relate to early NFL
wide receiver performance.
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

# Set up matplotlib and seaborn for consistent visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class TrackingDataAnalyzer:
    """
    Analyzer for football tracking data that processes college metrics
    and evaluates their relationship to NFL rookie performance.
    """
    
    def __init__(self, data_path=None):
        """
        Set up the analyzer with empty containers for data and results.
        Loads data if path is provided.
        """
        self.df = None
        self.feature_columns = []
        self.target_columns = []
        self.models = {}
        self.results = {}
        
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path):
        """
        Load tracking data from CSV and display basic dataset information.
        Returns self to allow method chaining.
        """
        self.df = pd.read_csv(data_path)
        print("Dataset Loaded")
        print(f"Rows: {self.df.shape[0]}")
        print(f"Columns: {self.df.shape[1]}")
        print(f"Seasons: {self.df['season'].min()} to {self.df['season'].max()}")
        print(f"Unique players: {self.df['player_name'].nunique()}")
        print(f"Teams: {self.df['offense_team'].nunique()}")
        
        return self
    
    def explore_data(self):
        """
        Analyze data quality, missing values, and key metric distributions.
        Provides football context for understanding missing data patterns.
        """
        print("Data Exploration and Quality Check")
        
        # Calculate and display missing data statistics
        print("\n Summary of Missing Data")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing[missing > 0],
            'Percentage': missing_pct[missing > 0]
        }).sort_values('Percentage', ascending=False)
        
        if len(missing_df) > 0:
            print(f"\n{len(missing_df)} metrics have missing data:")
            print(missing_df.head(15))
            
            # Explanation on why certain metrics have missing values in football context
            print("\nFootball Context:")
            print("Missing 'targeted' metrics means the player wasn't thrown to (limited targets)")
            print("Missing 'separation_VMAN' means the player didn't face man coverage or insufficient sample")
            print("Missing 'YACOE/CPOE' means no catch opportunities or tracking limitations")
        else:
            print("No missing data detected")
        
        # Show summary statistics for key performance metrics
        print("\n\nKey Metrics Summary")
        
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
        
        # Categorize players by playing time volume
        print("\n\nPlaying Time Distribution")
        play_bins = [0, 50, 100, 150, 200, 300, 1000]
        play_labels = ['<50 (Limited)', '50-100 (Rotational)', '100-150 (Starter)', 
                       '150-200 (Heavy)', '200-300 (Workhorse)', '300+ (Elite Volume)']
        self.df['volume_tier'] = pd.cut(self.df['total_plays'], bins=play_bins, labels=play_labels)
        print(self.df['volume_tier'].value_counts().sort_index())
        
        print("\nFootball Context:")
        print("\nPlayers with low snap counts may produce unstable estimates")
        
        return self
    
    def load_rookie_nfl_performance(self, start_year=2023, end_year=2025, min_games=2):
        """
        Load NFL rookie performance data using nfl_data_py.
        Identifies true rookie seasons by finding each player's first NFL season.
        Filters to wide receivers who played minimum games in their rookie year.
        """
        print("\n\nLoading True NFL Rookie Performance Data\n")

        try:
            import nfl_data_py as nfl
        except ImportError:
            raise ImportError("Please install nfl_data_py: pip install nfl_data_py")

        # Import complete career data to identify true rookie seasons
        career = nfl.import_seasonal_data(years=range(2000, end_year + 1))
        rosters = nfl.import_seasonal_rosters(years=range(2000, end_year + 1))

        # Merge career stats with roster position data
        career = career.merge(
            rosters[['player_id', 'player_name', 'position']],
            on='player_id',
            how='left'
        )

        # Filter to wide receivers only
        career = career[career['position'] == 'WR'].copy()

        # Find each player's first NFL season
        rookie_season = (
            career.groupby('player_name')['season']
            .min()
            .reset_index()
            .rename(columns={'season': 'rookie_season'})
        )

        # Join rookie season back to career data and filter to rookie year only
        career = career.merge(rookie_season, on='player_name')
        rookie_wr = career[career['season'] == career['rookie_season']].copy()

        # Keep only rookies within the specified year range
        rookie_wr = rookie_wr[
            rookie_wr['rookie_season'].between(start_year, end_year)
        ]

        # Require minimum games played to filter out practice squad players
        rookie_wr = rookie_wr[rookie_wr['games'] >= min_games].copy()
        rookie_wr = rookie_wr.drop_duplicates(subset=['player_name'], keep='first')

        # Calculate per-game performance metrics
        rookie_wr['yards_per_game'] = rookie_wr['receiving_yards'] / rookie_wr['games']
        rookie_wr['targets_per_game'] = rookie_wr['targets'] / rookie_wr['games']
        rookie_wr['receptions_per_game'] = rookie_wr['receptions'] / rookie_wr['games']
        rookie_wr['catch_rate'] = rookie_wr['receptions'] / rookie_wr['targets'].replace(0, np.nan)

        print(f"True rookie WRs loaded: {len(rookie_wr)}")
        print(f"Rookie seasons: {start_year}–{end_year}")
        print(f"Minimum games: {min_games}")

        # Store only the performance metrics needed for analysis
        self.rookie_perf = rookie_wr[
            [
                'player_name',
                'rookie_season',
                'yards_per_game',
                'targets_per_game',
                'receptions_per_game',
                'catch_rate'
            ]
        ]

        return self
    
    def merge_tracking_with_rookie_performance(self):
        """
        Merge college tracking data with NFL rookie performance.
        Uses only the final college season for each player to prevent data leakage.
        Enforces temporal ordering: college season must occur before rookie NFL season.
        """
        print("Merging College Tracking and Rookie NFL Performance")

        if not hasattr(self, 'rookie_perf'):
            raise ValueError("Rookie performance not loaded. Call load_rookie_nfl_performance() first.")

        # Keep only the most recent college season for each player
        college_final = (
            self.df.sort_values('season')
            .groupby('player_name')
            .last()
            .reset_index()
        )

        # Inner join keeps only players with both college and NFL data
        merged = college_final.merge(
            self.rookie_perf,
            on='player_name',
            how='inner'
        )

        # Remove any cases where college season is not before NFL rookie season
        merged = merged[merged['season'] < merged['rookie_season']]

        print(f"Leakage-safe players with college → rookie NFL data: {len(merged)}")

        self.df = merged
        return self

    def engineer_features(self):
        """
        Create derived features from raw tracking metrics.
        Features are organized into categories: athleticism, route running,
        contested catches, playmaking, change of direction, workload, and composites.
        """
        print("\n\nFeature Engineering")
        
        df = self.df.copy()
        
        # Athleticism features combining speed and acceleration metrics
        print("\n1. Athleticism Profile")
        
        # Average of consistent top speed and deep route speed
        df['speed_score'] = (df['max_speed_99'] + df['max_speed_30_inf_yards_max']) / 2
        print("Speed Score: Ability to reach and sustain top speed (deep threat indicator)")
        
        # High acceleration events normalized by total plays
        df['burst_rate'] = (df['high_acceleration_count_SUM'] / df['total_plays']) * 100
        print("Burst Rate: High acceleration events per play (route explosion)")
        
        # Speed in the first 10 yards indicates release and initial separation ability
        df['first_step_quickness'] = df['max_speed_0_10_yards_max']
        print("First Step Quickness: Speed in first 10 yards (release & separation)")
        
        # Deceleration ability important for cutting and route running
        df['brake_rate'] = (df['high_deceleration_count_SUM'] / df['total_plays']) * 100
        print("Brake Rate: Deceleration events per play (cut sharpness)")
        
        # Route running features measuring versatility and separation
        print("\n\n2. Route Running Intelligence")
        
        # Weighted combination of route depths and complexity
        df['route_diversity'] = (
            (df['10ydplus_route_MEAN'] * 0.3) +
            (df['20ydplus_route_MEAN'] * 0.4) +
            (df['changedir_route_MEAN'] * 0.3)
        )
        print("Route Diversity: Versatility across route tree (vs one-dimensional)")
        
        # Rename for clarity - measures consistent separation ability
        df['separation_consistency'] = df['average_separation_99']
        print("Separation Consistency: 99th percentile separation (elite vs lucky)")
        
        # Performance against man coverage is a key NFL predictor
        df['man_coverage_win_rate'] = df['separation_at_throw_VMAN']
        print("Man Coverage Win Rate: Separation vs man (toughest coverage)")
        
        # Ability to adjust to ball flight and close separation after throw
        df['tracking_skill'] = df['separation_change_postthrow_MEAN']
        print("Tracking Skill: Separation change after throw (ball tracking)")
        
        # Contested catch features measuring performance in tight coverage
        print("\n\n3. Contested Catch Ability")
        
        # Success rate when targeted in tight windows, avoiding division by zero
        df['contested_catch_rate'] = np.where(
            df['tight_window_at_throw_SUM'] > 0,
            (df['targeted_tightwindow_catch_SUM'] / df['tight_window_at_throw_SUM']) * 100,
            np.nan
        )
        print("Contested Catch Rate: Success in tight coverage (<2 yards separation)")
        
        # How often quarterback trusts player to make contested catches
        df['tight_window_target_pct'] = np.where(
            df['total_plays'] > 0,
            (df['tight_window_at_throw_SUM'] / df['total_plays']) * 100,
            np.nan
        )
        print("Tight Window Target %: How often QB trusts them in traffic")
        
        # Playmaking features measuring value creation beyond expectation
        print("\n\n4. Playmaking and Value Creation")
        
        # Yards after catch over expected measures pure playmaking ability
        df['yac_ability'] = df['YACOE_MEAN']
        print("YAC Ability: Yards after catch over expected (playmaking)")
        
        # Completion percentage over expected indicates reliable hands
        df['qb_friendly'] = df['CPOE_MEAN']
        print("QB-Friendly Rating: Completion % over expected (reliable hands)")
        
        # Change of direction features measuring cutting and route bending ability
        print("\n\n5. Change of Direction Focus")
        
        # Average entry and exit speed through 90 degree cuts (slants, outs, digs)
        df['sharp_cut_ability'] = (
            df['cod_top5_speed_entry_avg_90_'] + df['cod_top5_speed_exit_avg_90_']
        ) / 2
        print("Sharp Cut Ability: Speed through 90° cuts (slants, outs, digs)")
        
        # Average entry and exit speed through 180 degree cuts (comebacks, curls)
        df['route_bend_ability'] = (
            df['cod_top5_speed_entry_avg_180_'] + df['cod_top5_speed_exit_avg_180_']
        ) / 2
        print("Route Bend Ability: Speed through 180° cuts (comebacks, curls)")
        
        # Total separation generated from change of direction events
        df['cut_separation'] = df['cod_sep_generated_overall']
        print("Cut Separation: Yards of separation created from route breaks")
        
        # Workload features measuring usage patterns and durability
        print("\n\n6. Workload and Usage")
        
        # Binary indicator for high-volume players
        df['high_volume_player'] = (df['total_plays'] >= 150).astype(int)
        print("High Volume Flag: 150+ plays (starter/feature player)")
        
        # Average route depth per play indicates usage type
        df['distance_per_play'] = df['play_distance_SUM'] / df['total_plays']
        print("Distance per Play: Route depth tendency (deep vs short game)")
        
        # Combined acceleration and deceleration events measure dynamic playmaking
        df['total_explosive_events'] = (
            df['high_acceleration_count_SUM'] + df['high_deceleration_count_SUM']
        )
        df['explosive_rate'] = (df['total_explosive_events'] / df['total_plays']) * 100
        print("Explosive Rate: Combined accel/decel events (dynamic playmaking)")
        
        # Composite scores combining multiple metrics into 0-100 scales
        print("\n\n7. Combined Rating Metrics")
        
        # Normalize speed and burst to z-scores then scale to 0-100
        speed_norm = (df['speed_score'] - df['speed_score'].mean()) / df['speed_score'].std()
        burst_norm = (df['burst_rate'] - df['burst_rate'].mean()) / df['burst_rate'].std()
        df['athleticism_score'] = ((speed_norm + burst_norm) / 2) * 10 + 50
        df['athleticism_score'] = df['athleticism_score'].clip(0, 100)
        print("Athleticism Score: Combined speed + burst (0-100 scale)")
        
        # Combine route running metrics into single grade
        route_metrics = ['route_diversity', 'separation_consistency', 'man_coverage_win_rate']
        route_cols_available = [col for col in route_metrics if col in df.columns]
        if route_cols_available:
            route_norm = df[route_cols_available].apply(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )
            df['route_running_grade'] = (route_norm.mean(axis=1) * 10 + 50).clip(0, 100)
            print("Route Running Grade: Separation + versatility (0-100 scale)")
        
        # Overall prospect evaluation averaging key composite scores
        key_features = ['athleticism_score', 'route_running_grade', 'yac_ability']
        available_features = [f for f in key_features if f in df.columns]
        if available_features:
            df['prospect_score'] = df[available_features].mean(axis=1)
            print("Prospect Score: Holistic evaluation (weighted composite)")
        
        print(f"Feature engineering successful: {len([c for c in df.columns if c not in self.df.columns])} new features created")
        
        self.df = df
        return self
    
    def identify_archetypes(self, n_clusters=5):
        """
        Group players into distinct archetypes using K-means clustering.
        Uses speed, burst, and separation as clustering features.
        """
        print("Player Archetype Identification")
        
        # Define features to use for clustering
        cluster_features = [
            'speed_score', 'burst_rate', 'separation_consistency'
        ]
        
        # Verify features exist and have sufficient data
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
                print(f"{feat}: column not found")
        
        # Need at least 2 features for meaningful clustering
        if len(available_features) < 2:
            print("\nInsufficient features for clustering. Skipping archetype analysis.")
            print("   (Need at least 2 features with data)")
            return self
        
        # Keep only rows with complete data for clustering
        cluster_data = self.df[available_features + ['player_name']].dropna()
        
        print(f"\nClustering {len(cluster_data)} players with complete data on {len(available_features)} features...")
        
        # Adjust number of clusters if insufficient data
        if len(cluster_data) < n_clusters:
            print(f"\nOnly {len(cluster_data)} players with complete data.")
            print(f"   Need at least {n_clusters} for {n_clusters} clusters.")
            print(f"   Reducing to {min(3, len(cluster_data))} clusters...")
            n_clusters = min(3, max(2, len(cluster_data) // 10))
        
        # Minimum data requirement for stable clustering
        if len(cluster_data) < 10:
            print("\nToo few players for reliable clustering. Skipping.")
            return self
        
        # Standardize features so all have equal influence on clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(cluster_data[available_features])
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_data['archetype'] = kmeans.fit_predict(X_scaled)
        
        # Display characteristics of each cluster
        print("\nReciever Archetypes:\n")
        
        # Assign descriptive names to clusters
        archetype_names = {
            0: "Deep Threats",
            1: "Route Technicians", 
            2: "YAC Focus",
            3: "Complete Receivers",
            4: "Possession Specialists"
        }
        
        for cluster_id in range(n_clusters):
            cluster_players = cluster_data[cluster_data['archetype'] == cluster_id]
            
            print(f"\n{archetype_names.get(cluster_id, f'Archetype {cluster_id}')} ({len(cluster_players)} players)")
            
            # Show mean values for each clustering feature
            for feature in cluster_features:
                mean_val = cluster_players[feature].mean()
                print(f"  {feature:30s}: {mean_val:6.2f}")
            
            # Show example players from this archetype
            sample_players = (
                cluster_players['player_name']
                .drop_duplicates()
                .head(3)
                .tolist()
            )
            print(f"  Example players: {', '.join(sample_players)}")
        
        # Add archetype assignments back to main dataframe
        self.df = self.df.merge(
            cluster_data[['player_name', 'archetype']], 
            on='player_name', 
            how='left'
        )
        
        return self
    
    def prepare_modeling_data(self, target='targets_per_game', min_plays=50):
        """
        Prepare features and target variable for machine learning models.
        Filters to players with sufficient NFL data and college playing time.
        Handles missing values using median imputation.
        """
        print("Preparing data for modeling")

        # Define valid NFL performance targets
        nfl_targets = [
            'yards_per_game',
            'targets_per_game',
            'receptions_per_game',
            'catch_rate'
        ]

        # Validate target selection
        if target not in nfl_targets:
            raise ValueError(
                f"Target must be a TRUE rookie NFL metric. Choose from: {nfl_targets}"
            )

        # Keep only players with valid target values
        df_model = self.df.copy()
        df_model = df_model[df_model[target].notna()]

        print(f"\nPlayers with rookie NFLdata ({target}): {len(df_model)}")

        # Filter to players with sufficient college sample size
        df_model = df_model[df_model['total_plays'] >= min_plays]
        print(f"After min_plays filter ({min_plays}+): {len(df_model)}")

        if len(df_model) < 40:
            print("\n Warning: Very small NFL-labeled sample size")
        
        # Define candidate features for modeling
        potential_features = [
            'speed_score', 'burst_rate', 'first_step_quickness', 'brake_rate',
            'max_speed_99', 'route_diversity', 'separation_consistency', 
            'sharp_cut_ability', 'route_bend_ability', 'cut_separation',
            'changedir_route_MEAN', 'cod_sep_generated_overall',
            'distance_per_play', 'explosive_rate', 'total_plays'
        ]
        
        # Select features with at least 30% non-missing data
        self.feature_columns = []
        for feat in potential_features:
            if feat in df_model.columns:
                non_null_pct = df_model[feat].notna().sum() / len(df_model) * 100
                if non_null_pct >= 30:
                    self.feature_columns.append(feat)
                    print(f"{feat:35s}: {non_null_pct:5.1f}% available")
        
        # List of possible targets with descriptions
        potential_targets = [
            ('yards_per_game', 'Rookie Yards/Game (PRIMARY)'),
            ('targets_per_game', 'Rookie Targets/Game'),
            ('receptions_per_game', 'Rookie Receptions/Game'),
            ('catch_rate', 'Rookie Catch Rate'),
        ]

        # Verify target has sufficient data
        selected_target = None
        if target != 'auto' and target in df_model.columns:
            non_null = df_model[target].notna().sum()
            pct = non_null / len(df_model) * 100
            if pct >= 30:
                selected_target = target
                print(f"\nUsing target: {target} ({pct:.1f}% data)")
        
        # Auto-select target if not specified or insufficient data
        if selected_target is None:
            for tgt, desc in potential_targets:
                if tgt in df_model.columns:
                    non_null = df_model[tgt].notna().sum()
                    pct = non_null / len(df_model) * 100
                    print(f"   {tgt:30s}: {non_null:4,} ({pct:5.1f}%)")
                    if pct >= 30 and selected_target is None:
                        selected_target = tgt
                        target = tgt
                        print(f"   selected: {desc}")
        
        if selected_target is None:
            print("\nNo valid target found!")
            return None, None
        
        # Remove target from feature list to prevent data leakage
        if target in self.feature_columns:
            self.feature_columns.remove(target)
        
        # Remove highly correlated features to prevent data leakage
        if target == 'separation_consistency' and 'average_separation_99' in self.feature_columns:
            self.feature_columns.remove('average_separation_99')
        
        print(f"\nFinal: {len(self.feature_columns)} features → {target}")
        
        # Impute missing values using median strategy
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        
        # Impute features
        X = df_model[self.feature_columns].copy()
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=self.feature_columns,
            index=X.index
        )
        
        # Impute target
        y = df_model[target].copy()
        y = pd.Series(y, index=y.index)
        
        # Remove any rows that still have missing values
        valid_idx = ~(X_imputed.isnull().any(axis=1) | y.isnull())
        X_clean = X_imputed[valid_idx]
        y_clean = y[valid_idx]
        
        print(f"\nReady: {len(X_clean)} players")
        print(f"   Target: {target}")
        print(f"   Mean: {y_clean.mean():.3f} | Median: {y_clean.median():.3f}")
        
        return X_clean, y_clean    

    def build_models(self, X, y, test_size=0.25):
        """
        Train and evaluate a gradient boosting regression model.
        Uses cross-validation and holdout test set for evaluation.
        Stores model, predictions, and performance metrics.
        """
        print("Building models")
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"\n Training set: {len(X_train)} players")
        print(f" Test set: {len(X_test)} players")
        
        # Initialize and train gradient boosting model
        print("\n Training Gradient Boosting Regressor...")
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        
        # Evaluate using 5-fold cross-validation on training set
        cv_scores = cross_val_score(gb_model, X_train, y_train, cv=5, scoring='r2')
        print(f"  Cross-validation R² (5-fold): {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        # Generate predictions for train and test sets
        y_pred_train = gb_model.predict(X_train)
        y_pred_test = gb_model.predict(X_test)
        
        # Calculate performance metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        print(f"\n Model Performance:")
        print(f"  Training R²: {train_r2:.3f}")
        print(f"  Test R²: {test_r2:.3f}")
        print(f"  Test MAE: {test_mae:.3f}")
        print(f"  Test RMSE: {test_rmse:.3f}")
        
        # Store model and results for later use
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
        
        # Extract and display feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': gb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.results['feature_importance'] = feature_importance
        
        print("\n Top 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        return self


if __name__ == "__main__":
    print("""
        College Tracking Data to NFL Rookie On-Field Performance                      
    """)
    
    # Initialize analyzer
    analyzer = TrackingDataAnalyzer()
    
    print("\n Then run the full pipeline:")
    print("   analyzer.explore_data()")
    print("   analyzer.load_rookie_nfl_performance()")
    print("   analyzer.merge_tracking_with_rookie_performance()")
    print("   analyzer.engineer_features()")
    print("   analyzer.identify_archetypes()")
    print("   X, y = analyzer.prepare_modeling_data()")
    print("   analyzer.build_models(X, y)")