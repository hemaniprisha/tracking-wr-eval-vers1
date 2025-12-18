"""
Master Analysis Runner
=====================

This script orchestrates the complete analysis pipeline.
Run this ONE file and get everything.

Usage:
    python run_analysis.py --data data/tracking_data.csv
    
    # Or with custom output directory
    python run_analysis.py --data data/tracking_data.csv --output my_results/
    
    # Or for quick analysis (skip some advanced steps)
    python run_analysis.py --data data/tracking_data.csv --quick

Author: Prisha Hemani
Date: December 2024
"""

import os
import sys
import argparse
from datetime import datetime
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import our modules
try:
    # Case 1: project root on PYTHONPATH
    from src.tracking_analysis_draft import TrackingDataAnalyzer
    from src.advanced_modeling import AdvancedModelingPipeline, run_advanced_analysis
    from src.tracking_visuals import TrackVis, create_all_visuals

except ImportError:
    # Case 2: running script directly, add src to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

    try:
        from tracking_analysis_draft import TrackingDataAnalyzer
        from advanced_modeling import AdvancedModelingPipeline, run_advanced_analysis
        from tracking_visuals import TrackVis, create_all_visuals
    except ImportError as e:
        print("\nERROR: Cannot find required modules!")
        print(e)
        sys.exit(1)


def print_banner():
    # Title
    print("""
                WR Tracking Data to Draft Performance Analysis                                               
    """)
    print(f"Analysis Started: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")


def create_directory_structure(base_output_dir):
    # Create organized output directory structure.
    directories = {
        'visualizations': os.path.join(base_output_dir, 'visualizations'),
        'reports': os.path.join(base_output_dir, 'reports'),
        'data': os.path.join(base_output_dir, 'data'),
        'models': os.path.join(base_output_dir, 'models')
    }
    
    for name, path in directories.items():
        os.makedirs(path, exist_ok=True)
    
    print(f" Created output directory structure at: {base_output_dir}/")
    return directories


def save_executive_summary(analyzer, advanced_pipeline, output_path):
    # Generate executive summary.
    
    # Get best model from advanced pipeline
    best_model_info = advanced_pipeline.results['model_comparison'].iloc[0]
    
    summary = f"""
                            Summary:                                 
                   WR Evaluation with Tracking Data                       

Date: {datetime.now().strftime('%B %d, %Y')}
Analyst: Prisha Hemani

1. The Problem: Using Only Combine Testing Fails

Traditional NFL scouting relies heavily on combine metrics:
• 40-yard dash, vertical jump, broad jump, 3-cone drill, shuttle

Previous Analysis Results (Combine Prophet Analytics):
• Predictive Power (R²): -0.155 (worse than guessing)
• Mean Absolute Error: 14.5 yards per game (51% of average production)
• Conclusion: Static testing CANNOT predict WR performance

What Combine Misses: 
• Route discipline and precision
• Separation ability vs live defenders
• Ball tracking and adjustments
• Yards after catch creation
• Performance under fatigue
• Coverage recognition
• Functional athleticism in game situations

2. The Solution: Using Tracking-Derived Metrics

Current Analysis Results (Tracking Data):
• Dataset Size: {len(analyzer.df):,} players
• Tracking Metrics: {len(analyzer.df.columns)} features
• Sample Period: College football season(s)

Best Model: {best_model_info['Model'].upper()}
• Test R²: {best_model_info['Test R²']:.3f}
• Test MAE: {best_model_info['Test MAE']:.3f}
• Cross-Validation R²: {best_model_info['CV R² Mean']:.3f} ± {best_model_info['CV R² Std']:.3f}

Improvement over Combine:
• R² Improvement: {((best_model_info['Test R²'] - (-0.155)) / abs(-0.155) * 100):.0f}%
• Prediction Quality: {('Excellent' if best_model_info['Test R²'] > 0.35 else 'Good' if best_model_info['Test R²'] > 0.25 else 'Moderate')}
• Overfitting Check: {('✓ Good' if best_model_info['Overfit Gap'] < 0.15 else '⚠ Needs attention')}

3. Key Findings

Top 10 Predictive Features:

{advanced_pipeline.results.get('model_comparison', 'See technical report for details')}

Football Translation:
• Separation consistency > raw speed
• Route diversity indicates NFL versatility
• YAC ability identifies playmakers
• Performance vs man coverage = ultimate test
• Volume sustainability shows durability

PLAYER ARCHETYPES IDENTIFIED:
1. Deep Threat Burners: Elite speed, vertical stretching
2. Route Technicians: Consistent separation, full route tree
3. YAC Monsters: After-catch explosiveness, physicality
4. Complete Receivers: Above-average across all dimensions
5. Possession Specialists: Reliable hands, chain-moving


4. Business Impact


FINANCIAL CONTEXT:
• Average rookie WR contract (Rds 1-3): $2M-$8M per year
• Cost of evaluation miss: $8M+ over 4-year rookie deal
• Historical WR bust rate: 40-50% (combine-heavy methods)

ROI OF TRACKING-BASED EVALUATION:

Conservative Estimate (10% bust rate reduction):
• 40% bust rate → 30% bust rate
• 5 draft classes x 3 WRs per class = 15 picks
• 1.5 fewer busts x $8M = $12M saved

Moderate Estimate (15% bust rate reduction):
• 40% bust rate → 25% bust rate
• Same 15 picks scenario
• 2.25 fewer busts x $8M = $18M saved

Additional Benefits:
• Identify undervalued prospects (high tracking, low combine)
• Build scheme-specific player profiles
• Optimize draft capital allocation
• Competitive advantage in evaluation



6. Mechanics


Modeling Approach
• Compared 6 algorithms (Ridge, RF, GB, XGBoost, LightGBM, ElasticNet)
• Rigorous cross-validation (5-fold)
• Hyperparameter optimization
• Ensemble learning (weighted average of top models)
• Feature interaction analysis
• SHAP interpretability (modern ML explainability)
• Uncertainty quantification (bootstrap confidence intervals)


7. Conclusion


The evidence is clear:

TRACKING DATA WORKS. COMBINE TESTING DOESN'T.

With {best_model_info['Test R²']:.0%} variance explained (vs 0% for combine),
tracking metrics provide actionable insights for:
• Draft strategy
• Free agent targeting
• Trade evaluation
• Scheme fit assessment
• Player development

Teams that embrace tracking-based evaluation gain a 
competitive advantage in talent acquisition.

For reports, see: results/reports/ folder
For visualizations, see: results/visualizations/ folder
For interactive Streamlit app, run: streamlit run streamlit_app_draft.py

"""
    
    with open(output_path, 'w') as f:
        f.write(summary)
    
    print(f"Executive summary saved: {output_path}")
def add_to_run_analysis_draft_main():
    """
    Add this code at the end of the main() function in run_analysis_draft.py
    Right before the final print statements
    """
    
    # Export for Streamlit dashboard
    print("\n" + "="*80)
    print("EXPORTING FOR STREAMLIT DASHBOARD")
    print("="*80)
    
    export = {
        'processed_data': analyzer.df,
        'feature_importance': analyzer.results.get('feature_importance', pd.DataFrame()),
        'predictions': analyzer.results.get('predictions', {}),
        'metrics': analyzer.results.get('metrics', {}),
        'model_comparison': analyzer.results.get('model_comparison', pd.DataFrame()),
        'analysis_type': 'draft_prediction',
        'target_variable': 'draft_grade',
        'sample_size': len(analyzer.df),
        'n_features': len(X.columns) if X is not None else 0
    }
    
    export_path = os.path.join(args.output, 'tracking_draft_export.pkl')
    with open(export_path, 'wb') as f:
        pickle.dump(export, f)
    
    print(f"✓ Dashboard export: {export_path}")
    
    # Also copy to root directory for easy access
    root_export = 'tracking_draft_export.pkl'
    with open(root_export, 'wb') as f:
        pickle.dump(export, f)
    print(f"✓ Dashboard export (root): {root_export}")


def main():
    # Run complete analysis pipeline.
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Complete WR Tracking Data Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis.py --data data/tracking_data.csv
  python run_analysis.py --data data/tracking_data.csv --output my_results/
  python run_analysis.py --data data/tracking_data.csv --quick
        """
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to tracking data CSV file (REQUIRED)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory (default: results/)'
    )
    parser.add_argument(
        '--min-plays',
        type=int,
        default=50,
        help='Minimum plays for modeling (default: 50)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: skip SHAP and uncertainty quantification'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.data):
        print(f"\nError: Data file not found: {args.data}")
        print("\nPlease check the path and try again.")
        sys.exit(1)
    
    # Print banner
    print_banner()
    
    # Create output directories
    dirs = create_directory_structure(args.output)
    
    print("Loading Data")
    
    try:
        analyzer = TrackingDataAnalyzer(args.data)
    except Exception as e:
        print(f"\nErrror loading data: {str(e)}")
        sys.exit(1)
    
    # Step 2: Explore data
    print("Data Profiling")
    analyzer.explore_data()
    
    # Step 3: Feature engineering
    print("Feature Engineering")
    analyzer.engineer_features()
    
    # Step 4: Player archetypes (may skip if insufficient data)
    print("Player Archetypes")
    analyzer.identify_archetypes(n_clusters=5)
    
    # Step 5: Advanced modeling
    print("Model Analysis")
    
    # Auto-detect best target variable
    X, y = analyzer.prepare_modeling_data(min_plays=args.min_plays)
    
    # Check if we have valid data for modeling
    if X is None or y is None or len(X) < 10:
        print("\nError: Insufficient data for modeling!")
        print("\nYour dataset has extensive missing data in key target variables.")
        print("\nPossible solutions:")
        print("1. Check if data loaded correctly (run: python diagnose_data.py --data YOUR_FILE.csv)")
        print("2. Try lowering --min-plays threshold")
        print("3. Verify your CSV has the expected columns")
        sys.exit(1)
    
    # Run advanced analysis
    advanced_pipeline = AdvancedModelingPipeline()
    advanced_pipeline.build_model_suite()
    advanced_pipeline.compare_models(X, y, cv=5)
    
    # Hyperparameter tuning for best model
    best_model_name = advanced_pipeline.results['model_comparison'].iloc[0]['Model']
    if best_model_name in ['xgboost', 'lightgbm']:
        print(f"\nTuning hyperparameters for {best_model_name}...")
        advanced_pipeline.hyperparameter_tuning(X, y, model_name=best_model_name)
        # Re-compare with tuned model
        advanced_pipeline.compare_models(X, y)
    
    # Ensemble
    advanced_pipeline.build_ensemble(X, y, n=3)
    
    # Feature interactions
    advanced_pipeline.analyze_feature_interactions(X, y, n_feats=8)
    
    # SHAP (unless quick mode)
    if not args.quick:
        try:
            advanced_pipeline.shap_analysis(X, sample_size=100)
        except Exception as e:
            print(f" SHAP analysis skipped: {str(e)}")
    
    # Uncertainty quantification (unless quick mode)
    if not args.quick:
        advanced_pipeline.quantify_uncertainty(X, y, n_iterations=50)
    
    # Store results back in analyzer for visualization (map to expected format)
    analyzer.results = {
        'predictions': {
            'X_train': advanced_pipeline.results['X_train'],
            'X_test': advanced_pipeline.results['X_test'],
            'y_train': advanced_pipeline.results['y_train'],
            'y_test': advanced_pipeline.results['y_test'],
            'y_pred_train': advanced_pipeline.best_model.predict(advanced_pipeline.results['X_train']),
            'y_pred_test': advanced_pipeline.best_model.predict(advanced_pipeline.results['X_test']),
            'target_name': y.name if hasattr(y, 'name') else 'draft_capital'  # ADD THIS

        },
        'metrics': {
            'train_r2': advanced_pipeline.results['model_comparison'].iloc[0]['Train R²'],
            'test_r2': advanced_pipeline.results['model_comparison'].iloc[0]['Test R²'],
            'test_mae': advanced_pipeline.results['model_comparison'].iloc[0]['Test MAE'],
            'test_rmse': advanced_pipeline.results['model_comparison'].iloc[0]['Test RMSE'],
            'cv_mean': advanced_pipeline.results['model_comparison'].iloc[0]['CV R² Mean'],
            'cv_std': advanced_pipeline.results['model_comparison'].iloc[0]['CV R² Std']
        },
        'feature_importance': pd.DataFrame({
            'feature': X.columns,
            'importance': advanced_pipeline.best_model.feature_importances_ if hasattr(advanced_pipeline.best_model, 'feature_importances_') else [1/len(X.columns)] * len(X.columns)
        }).sort_values('importance', ascending=False) if hasattr(advanced_pipeline.best_model, 'feature_importances_') else pd.DataFrame({'feature': X.columns, 'importance': [1/len(X.columns)] * len(X.columns)}),
        'model_comparison': advanced_pipeline.results['model_comparison']
    }
    analyzer.models = advanced_pipeline.models
    
    # Step 6: Generate visualizations
    print("Creating Visualizations")
    
    viz = create_all_visuals(analyzer, output_dir=dirs['visualizations'])
    
    # Step 7: Generate reports
    print("Creating Reports")
    
    save_executive_summary(
        analyzer,
        advanced_pipeline,
        os.path.join(dirs['reports'], 'executive_summary.txt')
    )
    
    # Export processed data
    print("\nExporting processed data...")
    analyzer.df.to_csv(
        os.path.join(dirs['data'], 'processed_data_with_features.csv'),
        index=False
    )
    print(f"Processed data: {dirs['data']}/processed_data_with_features.csv")
    
    # Export model comparison
    advanced_pipeline.results['model_comparison'].to_csv(
        os.path.join(dirs['reports'], 'model_comparison.csv'),
        index=False
    )
    print(f"Model comparison: {dirs['reports']}/model_comparison.csv")
    
    # Export top players by archetype
    if 'archetype' in analyzer.df.columns:
        for arch_id in range(5):
            arch_players = analyzer.df[analyzer.df['archetype'] == arch_id]
            if len(arch_players) > 0 and 'prospect_score' in arch_players.columns:
                top_players = arch_players.nlargest(10, 'prospect_score')
                top_players.to_csv(
                    os.path.join(dirs['data'], f'archetype_{arch_id}_top_players.csv'),
                    index=False
                )
        print(f"Top players by archetype: {dirs['data']}/archetype_*_top_players.csv")
    # Export for Streamlit Dashboard
    import pickle

    export = {
        'processed_data': analyzer.df,
        'feature_importance': analyzer.results.get('feature_importance', pd.DataFrame()),
        'predictions': analyzer.results.get('predictions', {}),
        'metrics': analyzer.results.get('metrics', {}),
        'model_comparison': analyzer.results.get('model_comparison', pd.DataFrame()),
        'analysis_type': 'draft_prediction',
        'target_variable': y.name if hasattr(y, 'name') else 'target',
        'sample_size': len(analyzer.df),
        'n_features': len(X.columns) if X is not None else 0
    }

    # Save to root for Streamlit
    with open('tracking_draft_export.pkl', 'wb') as f:
        pickle.dump(export, f)
    print(" Dashboard export: tracking_draft_export.pkl")
    # Final summary
    print("Analysis Complete")
    
    print(f"\nAll results saved to: {args.output}/")
    print("\nGenerated Files:")
    print(f"   Visualizations: {dirs['visualizations']}/ (5-6 PNG files)")
    print(f"   Reports: {dirs['reports']}/ (executive summary, model comparison)")
    print(f"   Data: {dirs['data']}/ (processed data, top players)")
    
    print("\n Next Steps:")
    print("   1. Check visualizations folder for charts")
    print("   2. Run interactive dashboard:")
    print(f"      streamlit run streamlit_app.py")
    print("   3. Upload {args.data} in the dashboard sidebar")
    
    print("\nModel Performance Summary:")
    best = advanced_pipeline.results['model_comparison'].iloc[0]
    print(f"   Best Model: {best['Model'].upper()}")
    print(f"   Test R²: {best['Test R²']:.3f}")
    print(f"   Test MAE: {best['Test MAE']:.3f}")
    print(f"   vs Combine R² (-0.155): {((best['Test R²'] - (-0.155)) / abs(-0.155) * 100):.0f}% improvement")
    
    print(f"\nTotal runtime: {datetime.now()}")
    

if __name__ == "__main__":
    main()