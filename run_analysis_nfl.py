"""
Master Analysis Runner
=====================

Run this file to execute the full WR tracking analysis pipeline for tracking-NFL rookie metrics pipeline. 
Compatible with tracking_visuals.py visualization suite.
"""

import os
import sys
import argparse
from datetime import datetime
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# Robust imports
# ------------------------------------------------------------------------------

try:
    from src.tracking_analysis_nfl import TrackingDataAnalyzer
    from src.advanced_modeling import AdvancedModelingPipeline
    from src.tracking_visuals import create_all_visuals
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    from src.tracking_analysis_nfl import TrackingDataAnalyzer
    from src.advanced_modeling import AdvancedModelingPipeline
    from src.tracking_visuals import create_all_visuals


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║            WR TRACKING DATA ANALYSIS PIPELINE                             ║
╚══════════════════════════════════════════════════════════════════════════╝
""")
    print(f"Started: {datetime.now().strftime('%B %d, %Y %I:%M %p')}\n")


def create_directory_structure(base_dir):
    paths = {
        "visualizations": os.path.join(base_dir, "visualizations"),
        "reports": os.path.join(base_dir, "reports"),
        "data": os.path.join(base_dir, "data"),
        "models": os.path.join(base_dir, "models"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def compute_feature_importance(model, X):
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_)
    else:
        imp = np.ones(X.shape[1]) / X.shape[1]

    return (
        pd.DataFrame({"feature": X.columns, "importance": imp})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# ------------------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to tracking CSV")
    parser.add_argument("--output", default="results")
    parser.add_argument("--min-plays", type=int, default=50)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--rookie-start-year", type=int, default=2023)
    parser.add_argument("--rookie-end-year", type=int, default=2024)
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"\nData file not found: {args.data}")
        sys.exit(1)

    print_banner()
    dirs = create_directory_structure(args.output)

    # --------------------------------------------------------------------------
    # Load & feature engineering
    # --------------------------------------------------------------------------

    analyzer = TrackingDataAnalyzer(args.data)
    analyzer.explore_data()
    analyzer.engineer_features()

    analyzer.load_rookie_nfl_performance(
        start_year=args.rookie_start_year,
        end_year=args.rookie_end_year
    )
    analyzer.merge_tracking_with_rookie_performance()
    analyzer.identify_archetypes(n_clusters=5)



    # --------------------------------------------------------------------------
    # Target selection & modeling data
    # --------------------------------------------------------------------------

    X, y = analyzer.prepare_modeling_data(
        target="targets_per_game",
        min_plays=args.min_plays
    )

    if X is None or len(X) < 10:
        print("\nInsufficient data for modeling.")
        sys.exit(1)

    # --------------------------------------------------------------------------
    # Modeling
    # --------------------------------------------------------------------------

    pipeline = AdvancedModelingPipeline()
    pipeline.build_model_suite()
    pipeline.compare_models(X, y)

    best_model = pipeline.best_model

    if not args.quick:
        pipeline.build_ensemble(X, y)
        pipeline.analyze_feature_interactions(X, y)
        # Hyperparameter tuning
        best_model_name = pipeline.results['model_comparison'].iloc[0]['Model']
        if best_model_name in ['xgboost', 'lightgbm']:
            pipeline.hyperparameter_tuning(X, y, model_name=best_model_name)
            pipeline.compare_models(X, y)
        
        # SHAP analysis
        try:
            pipeline.shap_analysis(X, sample_size=min(100, len(X)))
        except Exception as e:
            print(f"SHAP analysis skipped: {str(e)}")
        
        # Uncertainty quantification  
        pipeline.quantify_uncertainty(X, y, n_iterations=50)

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        r2_score,
        mean_absolute_error,
        mean_squared_error,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    target_col = 'targets_per_game'  # Default
    if hasattr(y, 'name') and y.name:
        target_col = y.name
    elif isinstance(y, pd.Series) and y.name:
        target_col = y.name

    analyzer.results = {
        "model_comparison": pipeline.results["model_comparison"],
        "feature_importance": compute_feature_importance(best_model, X),
        "predictions": {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_pred_train": y_pred_train,
            "y_test": y_test,
            "y_pred_test": y_pred_test,
            "target_name": target_col,
        },
        "metrics": {
            "train_r2": r2_score(y_train, y_pred_train),
            "test_r2": r2_score(y_test, y_pred_test),
            "train_mae": mean_absolute_error(y_train, y_pred_train),
            "test_mae": mean_absolute_error(y_test, y_pred_test),
            "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "cv_mean": pipeline.results["model_comparison"].iloc[0]['CV R² Mean'] if 'CV R² Mean' in pipeline.results["model_comparison"].columns else 0,
            "cv_std": pipeline.results["model_comparison"].iloc[0]['CV R² Std'] if 'CV R² Std' in pipeline.results["model_comparison"].columns else 0,
        },
    }

    analyzer.models = pipeline.models

    # Visualizations
    create_all_visuals(analyzer, output_dir=dirs["visualizations"])

    # Exports

    analyzer.df.to_csv(
        os.path.join(dirs["data"], "processed_data_with_features.csv"),
        index=False,
    )

    pipeline.results["model_comparison"].to_csv(
        os.path.join(dirs["reports"], "model_comparison.csv"),
        index=False,
    )

    best = pipeline.results["model_comparison"].iloc[0]
    # Export for Streamlit Dashboard
    import pickle

    export = {
        'processed_data': analyzer.df,
        'feature_importance': analyzer.results.get('feature_importance', pd.DataFrame()),
        'predictions': analyzer.results.get('predictions', {}),
        'metrics': analyzer.results.get('metrics', {}),
        'model_comparison': analyzer.results.get('model_comparison', pd.DataFrame()),
        'analysis_type': 'nfl_rookie_performance',
        'target_variable': y.name if hasattr(y, 'name') else 'target',
        'sample_size': len(analyzer.df),
        'n_features': len(X.columns) if X is not None else 0
    }
    with open('tracking_nfl_export.pkl', 'wb') as f:
        pickle.dump(export, f)

    print("\nANALYSIS COMPLETE")
    print(f"Best Model: {best['Model'].upper()}")
    print(f"Test R²: {best['Test R²']:.3f}")
    print(f"Test MAE: {best['Test MAE']:.3f}")
    print(f"\nResults saved to: {args.output}/")


if __name__ == "__main__":
    main()
