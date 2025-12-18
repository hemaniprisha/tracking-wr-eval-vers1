"""
Multi-model ML pipeline with ensemble methods, interaction analysis, and uncertainty estimates.

Supports common tree/boosting algorithms, hyperparameter search, SHAP interpretation,
and bootstrap-based confidence intervals.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, train_test_split
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.utils import resample

import xgboost as xgb
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not installed. Install with: pip install shap")


class AdvancedModelingPipeline:
    """Manages model training, evaluation, ensembling, and interpretation."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.ensemble_weights = {}
        
    def build_model_suite(self):
        """Initialize baseline and advanced models for comparison."""
        print("Building models... ")
        
        # Ridge: fast baseline, good for linear relationships
        self.models['ridge'] = Ridge(alpha=1.0)
        
        # RF: handles nonlinearity well, less prone to overfitting
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1
        )
        
        # GBM: sequential boosting, strong but slower to train
        self.models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=4,
            subsample=0.8,
            random_state=42
        )
        
        # XGBoost: regularized boosting with built-in overfitting protection
        self.models['xgboost'] = xgb.XGBRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )
        
        # LightGBM: histogram-based, very fast on large datasets
        self.models['lightgbm'] = lgb.LGBMRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        # ElasticNet: L1+L2 regularization, good for feature selection
        self.models['elastic_net'] = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        
        return self
    
    def compare_models(self, X, y, cv=5):
        """Train all models and compare performance using CV + holdout test set."""
        print(f"\n\nComparing Models ({cv}-fold CV)")
        print(f"Training samples: {len(X):,}, Features: {X.shape[1]}\n")
        
        # Fixed split for fair comparison across models
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
        
        results = []
        
        for name, mdl in self.models.items():
            print(f"\nTraining {name.upper()}...")
            
            # CV scores on training data only
            cv_scores = cross_val_score(mdl, X_tr, y_tr, cv=cv, scoring='r2', n_jobs=-1)
            
            mdl.fit(X_tr, y_tr)
            
            y_pred_tr = mdl.predict(X_tr)
            y_pred_te = mdl.predict(X_te)
            
            r2_tr = r2_score(y_tr, y_pred_tr)
            r2_te = r2_score(y_te, y_pred_te)
            mae_te = mean_absolute_error(y_te, y_pred_te)
            rmse_te = np.sqrt(mean_squared_error(y_te, y_pred_te))
            
            gap = r2_tr - r2_te  # overfitting indicator
            
            results.append({
                'Model': name,
                'CV R² Mean': cv_scores.mean(),
                'CV R² Std': cv_scores.std(),
                'Train R²': r2_tr,
                'Test R²': r2_te,
                'Test MAE': mae_te,
                'Test RMSE': rmse_te,
                'Overfit Gap': gap
            })
            
            print(f"  CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            print(f"  Test R²: {r2_te:.3f} | MAE: {mae_te:.3f}")
            if gap > 0.15:
                print(f"  Overfit Gap: {gap:.3f} (HIGH - watch out)")
            else:
                print(f"  Overfit Gap: {gap:.3f} (looks good)")
        
        df = pd.DataFrame(results).sort_values('Test R²', ascending=False)
        
        print("\n\nModel Comparison Summary")
        print(df.to_string(index=False))
        
        # Pick winner based on test R²
        winner_idx = df['Test R²'].idxmax()
        winner_name = df.loc[winner_idx, 'Model']
        self.best_model = self.models[winner_name]
        
        print(f"\nBest Model: {winner_name.upper()}")
        print(f"  Test R²: {df.loc[winner_idx, 'Test R²']:.3f}")
        print(f"  Test MAE: {df.loc[winner_idx, 'Test MAE']:.3f}")
        
        # Store everything for later use
        self.results['model_comparison'] = df
        self.results['X_train'] = X_tr
        self.results['X_test'] = X_te
        self.results['y_train'] = y_tr
        self.results['y_test'] = y_te
        
        return self
    
    def hyperparameter_tuning(self, X, y, model_name='xgboost'):
        """Run randomized hyperparameter search for XGBoost or LightGBM."""
        print(f"\nHyperparameter tuning: {model_name.upper()}")
        
        if model_name == 'xgboost':
            params = {
                'n_estimators': [100, 150, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
            base = xgb.XGBRegressor(random_state=42, n_jobs=-1)
            
        elif model_name == 'lightgbm':
            params = {
                'n_estimators': [100, 150, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'num_leaves': [15, 31, 63],
                'min_child_samples': [5, 10, 20]
            }
            base = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
        else:
            print(f"Tuning not supported for {model_name}")
            return self
        
        # Estimate search space size for user's benefit
        n_combos = np.prod([len(v) for v in params.values()])
        print(f"Searching {len(params)} hyperparameters (~{n_combos:,} total combinations)")
        
        from sklearn.model_selection import RandomizedSearchCV
        
        search = RandomizedSearchCV(
            base, param_distributions=params, n_iter=50,
            scoring='r2', cv=5, random_state=42, n_jobs=-1, verbose=0
        )
        
        search.fit(X, y)
        
        print(f"Best CV R²: {search.best_score_:.3f}")
        print("Best Parameters:")
        for p, v in search.best_params_.items():
            print(f"  {p}: {v}")
        
        # Add tuned version to model collection
        self.models[f'{model_name}_tuned'] = search.best_estimator_
        self.results[f'{model_name}_tuning'] = {
            'best_score': search.best_score_,
            'best_params': search.best_params_
        }
        
        return self
    
    def build_ensemble(self, X, y, n=3):
        """Create weighted ensemble from top N models based on inverse MAE."""
        print(f"\nBuilding Ensemble (top {n} models)")
        
        if 'model_comparison' not in self.results:
            print("ERROR: Run compare_models() first")
            return self
        
        top = self.results['model_comparison'].head(n)
        
        print("\nEnsemble Members:")
        print(top[['Model', 'Test R²', 'Test MAE']].to_string(index=False))
        
        X_te = self.results['X_test']
        y_te = self.results['y_test']
        
        # Get predictions from each model
        preds = {}
        for _, row in top.iterrows():
            nm = row['Model']
            preds[nm] = self.models[nm].predict(X_te)
        
        # Weight by inverse MAE (lower error = higher weight)
        wts = {}
        total_inv = 0
        for _, row in top.iterrows():
            inv = 1.0 / row['Test MAE']
            wts[row['Model']] = inv
            total_inv += inv
        
        # Normalize weights to sum to 1
        for nm in wts:
            wts[nm] /= total_inv
        
        print("\nWeights:")
        for nm, w in wts.items():
            print(f"  {nm:20s}: {w:.3f}")
        
        # Weighted average prediction
        ens_pred = np.zeros(len(y_te))
        for nm, pr in preds.items():
            ens_pred += pr * wts[nm]
        
        ens_r2 = r2_score(y_te, ens_pred)
        ens_mae = mean_absolute_error(y_te, ens_pred)
        
        print(f"\nEnsemble Performance:")
        print(f"  Test R²: {ens_r2:.3f}")
        print(f"  Test MAE: {ens_mae:.3f}")
        
        # Compare to best single model
        best_r2 = top.iloc[0]['Test R²']
        diff = ens_r2 - best_r2
        
        print(f"\nvs Best Single Model: {diff:+.3f} R²")
        if diff > 0:
            print("  → Ensemble wins!")
        else:
            print("  → Single model is better (just use that)")
        
        self.ensemble_weights = wts
        self.results['ensemble'] = {
            'predictions': ens_pred,
            'r2': ens_r2,
            'mae': ens_mae,
            'weights': wts
        }
        
        return self
    
    def analyze_feature_interactions(self, X, y, n_feats=10):
        """Test pairwise interactions among top features."""
        print("\n\nFeature Interaction Analysis")
        print("Checking pairwise interactions...")
        
        # Identify top features from best model (or use correlation as fallback)
        if hasattr(self.best_model, 'feature_importances_'):
            imp = self.best_model.feature_importances_
            top_f = X.columns[np.argsort(imp)[-n_feats:]]
        else:
            corrs = X.corrwith(y).abs()
            top_f = corrs.nlargest(n_feats).index
        
        print(f"\nTesting interactions among top {len(top_f)} features:")
        for f in top_f:
            print(f"  • {f}")
        
        # Build interaction features (multiplicative)
        X_int = X.copy()
        int_names = []
        
        from itertools import combinations
        for f1, f2 in combinations(top_f, 2):
            nm = f"{f1}×{f2}"
            X_int[nm] = X[f1] * X[f2]
            int_names.append(nm)
        
        X_tr, X_te, y_tr, y_te = train_test_split(X_int, y, test_size=0.25, random_state=42)
        
        # Train RF to rank interaction importance
        mdl = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        mdl.fit(X_tr, y_tr)
        
        # Extract importance for interaction terms only
        int_imp = {}
        for nm in int_names:
            idx = list(X_tr.columns).index(nm)
            int_imp[nm] = mdl.feature_importances_[idx]
        
        sorted_int = sorted(int_imp.items(), key=lambda x: x[1], reverse=True)
        
        print("\nTop 10 Interactions by Importance:\n")
        for i, (nm, imp) in enumerate(sorted_int[:10], 1):

            print(f"{i:2d}. {nm:40s}  {imp:.4f}")
        
        # Check if interactions help
        y_pred = mdl.predict(X_te)
        r2_w_int = r2_score(y_te, y_pred)
        
        orig_r2 = self.results['model_comparison'].iloc[0]['Test R²']
        print(f"\nImpact:")
        print(f"  Original R²: {orig_r2:.3f}")
        print(f"  With Interactions: {r2_w_int:.3f}")
        
        self.results['interactions'] = {
            'top_interactions': sorted_int[:10],
            'r2_improvement': r2_w_int - orig_r2
        }
        
        return self
    
    def shap_analysis(self, X, sample_size=100):
        """Compute SHAP values for the best model (tree-based only)."""
        print("\n\nSHAP Analysis")
        
        X_samp = X.sample(min(sample_size, len(X)), random_state=42)
        
        explainer = shap.TreeExplainer(self.best_model)
        shap_vals = explainer.shap_values(X_samp)
        
        # Average absolute SHAP = feature importance
        mean_abs = np.abs(shap_vals).mean(axis=0)
        
        imp_df = pd.DataFrame({
            'feature': X.columns,
            'shap_importance': mean_abs
        }).sort_values('shap_importance', ascending=False)
        
        print("\nTop 10 Features by SHAP Importance:")
        print(imp_df.head(10).to_string(index=False))
        
        self.results['shap'] = {
            'values': shap_vals,
            'mean_abs': mean_abs,
            'feature_names': X.columns.tolist(),
            'importance_df': imp_df,
            'sample_size': len(X_samp)
        }
        
        print(f"\nSHAP computed on {len(X_samp)} samples")
        return self
    
    def quantify_uncertainty(self, X, y, n_iterations=100):
        """Bootstrap resampling to estimate prediction intervals and R² confidence."""
        print("\n\nUncertainty Quantification")
        print(f"Running {n_iterations} bootstrap iterations...")
        
        X_te = self.results['X_test']
        y_te = self.results['y_test']
        
        boot_preds = []
        boot_r2s = []
        
        for i in range(n_iterations):
            if (i + 1) % 20 == 0:
                print(f"  Iteration {i + 1}/{n_iterations}")
            
            X_b, y_b = resample(self.results['X_train'], self.results['y_train'], random_state=i)
            
            # Clone model with same params
            mdl = type(self.best_model)(**self.best_model.get_params())
            mdl.fit(X_b, y_b)
            
            y_pred = mdl.predict(X_te)
            boot_preds.append(y_pred)
            boot_r2s.append(r2_score(y_te, y_pred))
        
        boot_preds = np.array(boot_preds)
        pred_mean = boot_preds.mean(axis=0)
        pred_lo = np.percentile(boot_preds, 2.5, axis=0)
        pred_hi = np.percentile(boot_preds, 97.5, axis=0)
        
        # What fraction of true values fall inside 95% PI?
        coverage = np.mean((y_te >= pred_lo) & (y_te <= pred_hi))
        
        r2_mean = np.mean(boot_r2s)
        r2_lo = np.percentile(boot_r2s, 2.5)
        r2_hi = np.percentile(boot_r2s, 97.5)
        
        print(f"\nResults:")
        print(f"  R² Mean: {r2_mean:.3f}")
        print(f"  R² 95% CI: [{r2_lo:.3f}, {r2_hi:.3f}]")
        print(f"  Prediction Coverage: {coverage:.1%}")
        
        self.results['uncertainty'] = {
            'bootstrap_predictions': boot_preds,
            'pred_mean': pred_mean,
            'pred_lower_95': pred_lo,
            'pred_upper_95': pred_hi,
            'coverage': coverage,
            'r2_distribution': boot_r2s,
            'r2_mean': r2_mean,
            'r2_ci_lower': r2_lo,
            'r2_ci_upper': r2_hi,
            'n_iterations': n_iterations
        }
        
        return self


def run_advanced_analysis(X, y):
    """Run full pipeline: model comparison, tuning, ensemble, interactions, SHAP, uncertainty."""
    pipe = AdvancedModelingPipeline()
    
    pipe.build_model_suite()
    pipe.compare_models(X, y)
    
    # Tune best model if it's XGB or LGB
    best_nm = pipe.results['model_comparison'].iloc[0]['Model']
    if best_nm in ['xgboost', 'lightgbm']:
        pipe.hyperparameter_tuning(X, y, model_name=best_nm)
        pipe.compare_models(X, y)  # Re-run comparison with tuned model
    
    pipe.build_ensemble(X, y, n=3)
    pipe.analyze_feature_interactions(X, y)
    
    if SHAP_AVAILABLE:
        pipe.shap_analysis(X)
    
    pipe.quantify_uncertainty(X, y, n_iterations=50)
    
    return pipe


if __name__ == "__main__":
    print("""
Advanced Multi-Model Analysis Pipeline

Includes:
- 6 model comparison (Ridge, RF, GBM, XGB, LGB, ElasticNet)
- Hyperparameter tuning
- Weighted ensemble
- Feature interactions
- SHAP interpretation
- Bootstrap uncertainty quantification

Usage:
    from advanced_modeling import run_advanced_analysis
    pipeline = run_advanced_analysis(X, y)
    results = pipeline.results
""")