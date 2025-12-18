import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, Patch
from matplotlib.gridspec import GridSpec
from scipy import stats

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


class TrackVis:
    def __init__(self, analyzer, analysis_type='nfl'):
        self.analyzer = analyzer
        self.df = analyzer.df
        self.analysis_type = analysis_type
        
        # color scheme we settled on after trying a bunch
        self.category = {
            'primary': '#1f77b4',
            'success': '#2ca02c', 
            'warning': '#ff7f0e',
            'danger': '#d62728',
            'neutral': '#7f7f7f',
            'elite': '#9467bd'
        }
        
        if analysis_type == 'nfl':
            self.target_label = 'NFL Rookie Performance'
            self.target_unit = 'targets/game'
        else:
            self.target_label = 'Draft Capital'
            self.target_unit = 'draft points'
    
    def plot_model_compare(self, combine_r2=-0.155, save_path='1_model_comparison.png'):
        # Side-by-side comparison: combine metrics vs our tracking-based approach
        # Spoiler alert: combine testing fails to predict performance, tracking data succeeds
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        tracking_r2 = self.analyzer.results['metrics']['test_r2']
        
        pred_data = self.analyzer.results.get('predictions', {})
        n_train = len(pred_data.get('y_train', []))
        n_test = len(pred_data.get('y_test', []))
        n_total = n_train + n_test
        
        # LEFT PANEL - Combine metrics
        ax1 = axes[0]
        ax1.text(0.5, 0.95, 'Combine Metrics', 
                ha='center', va='top', fontsize=20, fontweight='bold',
                transform=ax1.transAxes)
        ax1.text(0.5, 0.85, '40-time • Vertical • Broad Jump • 3-Cone • Shuttle',
                ha='center', va='top', fontsize=11, style='italic',
                transform=ax1.transAxes, color='gray')
        
        r2_color = self.category['danger']
        ax1.text(0.5, 0.55, f'R² = {combine_r2:.3f}',
                ha='center', va='center', fontsize=48, fontweight='bold',
                transform=ax1.transAxes, color=r2_color)
        
        ax1.text(0.5, 0.35, 'Worse than guessing\nthe average',
                ha='center', va='top', fontsize=14,
                transform=ax1.transAxes, color=r2_color)
        
        ax1.text(0.5, 0.15, '✗ Route discipline\n✗ Separation ability\n✗ Ball tracking\n✗ YAC creation',
                ha='center', va='top', fontsize=11,
                transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='#ffebee'))
        
        ax1.text(0.5, 0.02, 'n = 92 rookie WRs (2015-2023)',
                ha='center', va='bottom', fontsize=9, style='italic',
                transform=ax1.transAxes, color='gray')
        
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        
        # RIGHT PANEL - Tracking model performance
        ax2 = axes[1]
        ax2.text(0.5, 0.95, 'Tracking Metrics', 
                ha='center', va='top', fontsize=20, fontweight='bold',
                transform=ax2.transAxes)
        ax2.text(0.5, 0.85, 'Speed • Separation • Route Running • Change of Direction',
                ha='center', va='top', fontsize=11, style='italic',
                transform=ax2.transAxes, color='gray')
        
        r2_color = self.category['success'] if tracking_r2 > 0.2 else self.category['warning']
        ax2.text(0.5, 0.55, f'R² = {tracking_r2:.3f}',
                ha='center', va='center', fontsize=48, fontweight='bold',
                transform=ax2.transAxes, color=r2_color)
        
        if combine_r2 < 0:
            improvement_pct = ((tracking_r2-combine_r2)/abs(combine_r2)) * 100
        else:
            improvement_pct = ((tracking_r2-combine_r2)/max(combine_r2, 0.001)) * 100
        
        ax2.text(0.5, 0.35, f'{improvement_pct:.0f}% improvement\nover combine testing',
                ha='center', va='top', fontsize=14, fontweight='bold',
                transform=ax2.transAxes, color=r2_color)
        
        ax2.text(0.5, 0.15, 'Real game speed\nSeparation skills\nRoute precision\nPlaymaking ability',
                ha='center', va='top', fontsize=11,
                transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='#e8f5e9'))
        
        sample_note = f'n = {n_total} players' if n_total > 0 else 'College tracking data'
        ax2.text(0.5, 0.02, sample_note,
                ha='center', va='bottom', fontsize=9, style='italic',
                transform=ax2.transAxes, color='gray')
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        plt.suptitle('WR Evaluation Comparison',
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
        
        return fig
    
    def plot_feature_importance(self, top_n=15, outfile='2_feature_importance.png'):
        # Ranked feature importance - shows which metrics drive the model's predictions
        feature_imp = self.analyzer.results['feature_importance'].head(top_n).copy()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Translate technical feature names into readable football terminology
        football_labels = {
            'speed_score': 'Top-End Speed',
            'separation_consistency': 'Get-Open Ability',
            'route_diversity': 'Route Tree Versatility',
            'yac_ability': 'Yards After Catch',
            'sharp_cut_ability': 'Cutting Speed',
            'man_coverage_win_rate': 'vs Man Coverage',
            'burst_rate': 'Acceleration Bursts',
            'qb_friendly': 'Reliable Hands',
            'cut_separation': 'Separation from Cuts',
            'route_bend_ability': 'Route Bending',
            'explosive_rate': 'Explosive Plays',
            'first_step_quickness': 'Release Speed',
            'distance_per_play': 'Route Depth',
            'brake_rate': 'Deceleration Control',
            'total_plays': 'Volume/Sample Size',
            'max_speed_99': 'Max Speed 99',
            'cod_sep_generated_overall': 'Cod Sep Generated Overall',
            'changedir_route_MEAN': 'Changedir Route Mean'
        }
        
        feature_imp['football_label'] = feature_imp['feature'].map(
            lambda x: football_labels.get(x, x.replace('_', ' ').title())
        )
        
        # Assign colors by feature category for visual grouping
        category = []
        for feat in feature_imp['feature']:
            if any(x in feat.lower() for x in ['speed', 'burst', 'quickness', 'brake', 'max_speed']):
                category.append(self.category['elite'])
            elif any(x in feat.lower() for x in ['separation', 'route', 'man_coverage', 'changedir']):
                category.append(self.category['primary'])
            elif any(x in feat.lower() for x in ['yac', 'qb_friendly', 'explosive', 'cpoe', 'yacoe']):
                category.append(self.category['success'])
            elif any(x in feat.lower() for x in ['cut', 'bend', 'cod']):
                category.append(self.category['warning'])
            else:
                category.append(self.category['neutral'])
        
        bars = ax.barh(feature_imp['football_label'], feature_imp['importance'], color=category)
        
        ax.set_xlabel('Feature Importance (Impact on Prediction)', fontsize=12, fontweight='bold')
        ax.set_title(f'What Actually Predicts WR Success?\nTracking Metrics Ranked by Predictive Power',
                    fontsize=14, fontweight='bold', pad=20)
        
        for bar, val in zip(bars, feature_imp['importance']):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
        
        legend = [
            Patch(facecolor=self.category['elite'], label='Athleticism'),
            Patch(facecolor=self.category['primary'], label='Route Running'),
            Patch(facecolor=self.category['success'], label='Playmaking'),
            Patch(facecolor=self.category['warning'], label='Change of Direction'),
        ]
        ax.legend(handles=legend, loc='lower right', frameon=True, fontsize=10)
        
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        print(f"Saved: {outfile}")
        plt.close()
        
        return fig
    
    def plot_act_vs_pred(self, fname='3_actual_vs_predicted.png'):
        # Actual vs predicted scatter plots - validates model accuracy on train/test splits
        pred_data = self.analyzer.results['predictions']
        metrics = self.analyzer.results['metrics']
        
        target_name = pred_data.get('target_name')
        if not target_name or target_name is None:
            target_name = 'targets_per_game' if self.analysis_type == 'nfl' else 'draft_value'
        
        if self.analysis_type == 'nfl':
            xlabel = f'Actual NFL Rookie {target_name.replace("_", " ").title()}'
            ylabel = f'Predicted NFL Rookie {target_name.replace("_", " ").title()}'
        else:
            xlabel = 'Actual Draft Value'
            ylabel = 'Predicted Draft Value'
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Training set performance
        ax1 = axes[0]
        ax1.scatter(pred_data['y_train'], pred_data['y_pred_train'], 
                   alpha=0.6, s=80, color=self.category['primary'],
                   edgecolors='black', linewidth=0.5)
        
        min_val = min(pred_data['y_train'].min(), pred_data['y_pred_train'].min())
        max_val = max(pred_data['y_train'].max(), pred_data['y_pred_train'].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2, label='Perfect Prediction')
        
        ax1.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax1.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax1.set_title(f'Training Set (n={len(pred_data["y_train"])})\nR² = {metrics["train_r2"]:.3f}',
                     fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)
        
        # Test set - the real measure of predictive power
        ax2 = axes[1]
        ax2.scatter(pred_data['y_test'], pred_data['y_pred_test'],
                   alpha=0.6, s=80, color=self.category['success'],
                   edgecolors='black', linewidth=0.5)
        
        min_val = min(pred_data['y_test'].min(), pred_data['y_pred_test'].min())
        max_val = max(pred_data['y_test'].max(), pred_data['y_pred_test'].max())
        ax2.plot([min_val, max_val], [min_val, max_val],
                'r--', linewidth=2, label='Perfect Prediction')
        
        ax2.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax2.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax2.set_title(f'Test Set (n={len(pred_data["y_test"])})\nR² = {metrics["test_r2"]:.3f} | MAE = {metrics["test_mae"]:.3f}',
                     fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
        
        plt.suptitle('Model Prediction Quality: Capturing Real Performance',
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"Saved: {fname}")
        plt.close()
        
        return fig
    
    def plot_archetypes(self, output='4_player_archetypes.png'):
        # Visualize the five distinct receiver archetypes discovered through clustering
        if 'archetype' not in self.df.columns:
            print("No archetype data available - skipping this visualization")
            return None
        
        df_archetypes = self.df.dropna(subset=['archetype'])
        
        if len(df_archetypes) < 5:
            print("Insufficient archetype data for meaningful visualization")
            return None
        
        fig = plt.figure(figsize=(16, 10))
        gspec = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        archetypes = {
            0: {'name': 'Deep Threat Burners', 'color': '#e74c3c'},
            1: {'name': 'Route Technicians', 'color': '#3498db'},
            2: {'name': 'YAC Monsters', 'color': '#2ecc71'},
            3: {'name': 'Complete Receivers', 'color': '#9b59b6'},
            4: {'name': 'Possession Specialists', 'color': '#f39c12'}
        }
        
        # Main scatter plot showing archetype separation in feature space
        ax_main = fig.add_subplot(gspec[0, :])
        
        x_col = 'speed_score' if 'speed_score' in df_archetypes.columns else 'max_speed_99'
        y_col = 'separation_consistency' if 'separation_consistency' in df_archetypes.columns else 'average_separation_99'
        
        for arch_id, arch_info in archetypes.items():
            arch_data = df_archetypes[df_archetypes['archetype'] == arch_id]
            if len(arch_data) > 0 and x_col in arch_data.columns and y_col in arch_data.columns:
                plot_data = arch_data[[x_col, y_col]].dropna()
                if len(plot_data) > 0:
                    ax_main.scatter(
                        plot_data[x_col], 
                        plot_data[y_col],
                        c=arch_info['color'], 
                        label=arch_info['name'],
                        s=100, 
                        alpha=0.6,
                        edgecolors='black',
                        linewidth=0.5
                    )
        
        ax_main.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax_main.set_ylabel(y_col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax_main.set_title('5 Distinct Receiver Archetypes: Speed vs Separation', 
                         fontsize=14, fontweight='bold')
        ax_main.legend(loc='best', frameon=True, fontsize=10)
        ax_main.grid(alpha=0.3)
        
        # Individual profile charts for each archetype
        positions = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]
        
        metrics_to_check = ['speed_score', 'separation_consistency', 'yac_ability', 
                          'route_diversity', 'explosive_rate',
                          'max_speed_99', 'average_separation_99', 'YACOE_MEAN',
                          'changedir_route_MEAN', 'burst_rate']
        avail_m = [m for m in metrics_to_check if m in df_archetypes.columns][:5]
        
        for i, (arch_id, arch_info) in enumerate(archetypes.items()):
            if i >= len(positions):
                break
                
            row, col = positions[i]
            ax = fig.add_subplot(gspec[row, col])
            
            arch_data = df_archetypes[df_archetypes['archetype'] == arch_id]
            
            if len(arch_data) > 0 and avail_m:
                values = []
                for m in avail_m:
                    if m in arch_data.columns:
                        val = arch_data[m].mean()
                        col_data = df_archetypes[m].dropna()
                        if len(col_data) > 0 and col_data.std() > 0:
                            norm_val = ((val - col_data.mean()) / col_data.std()) * 15 + 50
                            values.append(max(0, min(100, norm_val)))
                        else:
                            values.append(50)
                    else:
                        values.append(50)
                
                x = np.arange(len(avail_m))
                ax.bar(x, values, color=arch_info['color'], alpha=0.7)
                ax.set_xticks(x)
                ax.set_xticklabels([m.replace('_', '\n')[:15] for m in avail_m], 
                                  rotation=0, fontsize=7)
                ax.set_ylim(0, 100)
                ax.set_title(f"{arch_info['name']}\n({len(arch_data)} players)", 
                           fontsize=11, fontweight='bold')
                ax.axhline(50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
                ax.grid(axis='y', alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Insufficient\nData', 
                       ha='center', va='center', fontsize=12,
                       transform=ax.transAxes)
                ax.set_title(arch_info['name'], fontsize=11, fontweight='bold')
                ax.axis('off')
        
        plt.suptitle('Understanding Receiver Diversity: One Size Does NOT Fit All',
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig(output, dpi=300, bbox_inches='tight')
        print(f"Saved: {output}")
        plt.close()
        
        return fig
    
    def plot_tracking_outcome(self, savename='5_tracking_to_outcome.png'):
        # Correlation analysis: do tracking metrics actually predict performance outcomes?
        # Shows relationship between top predictive features and the target variable
        fig = plt.figure(figsize=(16, 10))
        gspec = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
        
        feat_imp = self.analyzer.results.get('feature_importance', pd.DataFrame())
        
        if len(feat_imp) < 2:
            print("Not enough feature importance data")
            return None
        
        top_features = feat_imp.head(4)['feature'].tolist()
        
        pred_data = self.analyzer.results.get('predictions', {})
        target_col = pred_data.get('target_name', None)
        
        # Fallback logic if target name wasn't stored in prediction results
        if target_col is None or target_col not in self.df.columns:
            if self.analysis_type == 'nfl':
                target_candidates = ['targets_per_game', 'yards_per_game', 'receptions_per_game', 'catch_rate']
            else:
                target_candidates = ['draft_capital', 'rec_yards', 'production_score', 'draft_pick']
            
            for tc in target_candidates:
                if tc in self.df.columns and self.df[tc].notna().sum() > 10:
                    target_col = tc
                    break
        
        if target_col is None:
            print("Unable to identify valid target column for visualization")
            return None
        
        if self.analysis_type == 'nfl':
            target_label = 'NFL Rookie Performance'
        else:
            if target_col == 'draft_capital':
                target_label = 'Draft Value'
            elif target_col == 'rec_yards':
                target_label = 'College Production'
            else:
                target_label = target_col.replace('_', ' ').title()
        
        # Create four scatter plots: top features vs outcome variable
        for i, feature in enumerate(top_features[:4]):
            row, col = divmod(i, 2)
            ax = fig.add_subplot(gspec[row, col])
            
            if feature not in self.df.columns:
                ax.text(0.5, 0.5, f'{feature}\nNot Available', 
                       ha='center', va='center', fontsize=12,
                       transform=ax.transAxes)
                ax.axis('off')
                continue
            
            plot_data = self.df[[feature, target_col]].dropna()
            
            if len(plot_data) < 5:
                ax.text(0.5, 0.5, f'{feature}\nInsufficient Data', 
                       ha='center', va='center', fontsize=12,
                       transform=ax.transAxes)
                ax.axis('off')
                continue
            
            ax.scatter(plot_data[feature], plot_data[target_col],
                      alpha=0.5, s=60, color=self.category['primary'],
                      edgecolors='black', linewidth=0.3)
            
            corr = plot_data[feature].corr(plot_data[target_col])
            
            # Fit trend line to visualize relationship
            z = np.polyfit(plot_data[feature], plot_data[target_col], 1)
            p = np.poly1d(z)
            x_line = np.linspace(plot_data[feature].min(), plot_data[feature].max(), 100)
            ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7)
            
            feat_importance = feat_imp[feat_imp['feature'] == feature]['importance'].values
            imp_val = feat_importance[0] if len(feat_importance) > 0 else 0
            
            # Convert technical names to football-friendly labels
            football_labels = {
                'speed_score': 'Top-End Speed',
                'separation_consistency': 'Get-Open Ability',
                'route_bend_ability': 'Route Bending',
                'explosive_rate': 'Explosive Plays',
                'burst_rate': 'Acceleration Bursts',
                'max_speed_99': 'Consistent Top Speed',
                'cod_sep_generated_overall': 'Separation from Cuts',
                'distance_per_play': 'Route Depth',
                'first_step_quickness': 'Release Speed',
                'route_diversity': 'Route Versatility'
            }
            
            feature_label = football_labels.get(feature, feature.replace('_', ' ').title())
            
            ax.set_xlabel(f'{feature_label}', fontsize=11, fontweight='bold')
            ax.set_ylabel(f'{target_col.replace("_", " ").title()}', fontsize=11, fontweight='bold')
            ax.set_title(f'{feature_label} → {target_label}\nr = {corr:.3f} | Importance: {imp_val:.3f}',
                        fontsize=11, fontweight='bold')
            ax.grid(alpha=0.3)
            
            # Color-code background by correlation strength
            if abs(corr) > 0.3:
                ax.patch.set_facecolor('#e8f5e9')
                ax.patch.set_alpha(0.3)
            elif abs(corr) < 0.1:
                ax.patch.set_facecolor('#ffebee')
                ax.patch.set_alpha(0.3)
        
        if self.analysis_type == 'nfl':
            title = 'Tracking Metrics to NFL Rookie Production\nDo College Skills Translate to the Pros?'
        else:
            title = 'Tracking Metrics to Draft Capital\nWhat Do Scouts Really Value?'
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig(savename, dpi=300, bbox_inches='tight')
        print(f"Saved: {savename}")
        plt.close()
        
        return fig
    
    def plot_value(self, path='6_value_discovery.png'):
        # Residual analysis to identify over/underperformers relative to tracking profile
        # Green zone = outperformed expectations, Red zone = underperformed
        pred_data = self.analyzer.results.get('predictions', {})
        
        if 'y_test' not in pred_data or 'y_pred_test' not in pred_data:
            print("Prediction data required for value analysis")
            return None
        
        y_test = pred_data['y_test']
        y_pred = pred_data['y_pred_test']
        
        residuals = np.array(y_test) - np.array(y_pred)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # LEFT: Distribution of prediction errors
        ax1 = axes[0]
        ax1.hist(residuals, bins=20, color=self.category['primary'], 
                alpha=0.7, edgecolor='black')
        ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
        ax1.axvline(residuals.mean(), color='green', linestyle='-', linewidth=2, 
                   label=f'Mean Error: {residuals.mean():.2f}')
        
        ax1.set_xlabel('Prediction Error (Actual - Predicted)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Players', fontsize=12, fontweight='bold')
        ax1.set_title('Prediction Error Distribution\nPositive = Outperformed | Negative = Underperformed',
                     fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)
        
        ax1.axvspan(residuals.min(), -residuals.std(), alpha=0.1, color='red', 
                   label='Underperformers')
        ax1.axvspan(residuals.std(), residuals.max(), alpha=0.1, color='green',
                   label='Overperformers')
        
        # RIGHT: Actual vs predicted with value zone highlighting
        ax2 = axes[1]
        
        category = ['green' if r > residuals.std() else 'red' if r < -residuals.std() else 'gray' 
                 for r in residuals]
        
        ax2.scatter(y_pred, y_test, c=category, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
        
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Fit')
        
        ax2.set_xlabel('Predicted Performance (from Tracking)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Actual Performance', fontsize=12, fontweight='bold')
        ax2.set_title('Overperforming vs Undeperforming Players\nGreen = Value Pick | Red = Bust Risk',
                     fontsize=13, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        legend = [
            Patch(facecolor='green', label='Outperformed (Value)'),
            Patch(facecolor='red', label='Underperformed (Risk)'),
            Patch(facecolor='gray', label='As Expected'),
        ]
        ax2.legend(handles=legend, loc='upper left', fontsize=10)
        
        overperformers = sum(1 for r in residuals if r > residuals.std())
        underperformers = sum(1 for r in residuals if r < -residuals.std())
        as_expected = len(residuals) - overperformers - underperformers
        
        summary = f"Overperformers: {overperformers} | As Expected: {as_expected} | Underperformers: {underperformers}"
        fig.text(0.5, 0.02, summary, ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Finding Value: Tracking Profile vs Actual Outcome',
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved: {path}")
        plt.close()
        
        return fig
    
    def create_player_card(self, player_name, save_path=None):
        # Generate individual scouting report with radar chart and key metrics
        player_data = self.df[self.df['player_name'] == player_name]
        
        if len(player_data) == 0:
            print(f"Player not found: {player_name}")
            return None
        
        player = player_data.iloc[0]
        fig = plt.figure(figsize=(12, 10))
        gspec = GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)
        
        # Player name and basic info header
        ax_header = fig.add_subplot(gspec[0, :])
        ax_header.text(0.5, 0.7, player_name.upper(),
                      ha='center', va='center', fontsize=24, fontweight='bold',
                      transform=ax_header.transAxes)
        ax_header.text(0.5, 0.3, 
                      f"{player.get('offense_team', 'N/A')} | {player.get('season', 'N/A')} | {player.get('total_plays', 0):.0f} plays",
                      ha='center', va='center', fontsize=14,
                      transform=ax_header.transAxes, color='gray')
        ax_header.axis('off')
        
        # Radar chart showing percentile rankings across key metrics
        ax_radar = fig.add_subplot(gspec[1:3, 0], projection='polar')
        
        metrics_to_check = ['speed_score', 'separation_consistency', 'yac_ability', 
                          'route_diversity', 'qb_friendly',
                          'max_speed_99', 'average_separation_99', 'YACOE_MEAN']
        avail_m = [m for m in metrics_to_check if m in player.index and pd.notna(player.get(m))][:5]
        
        if avail_m:
            percentiles = []
            for m in avail_m:
                col_data = self.df[m].dropna()
                if len(col_data) > 0:
                    percentile = (player[m] > col_data).sum() / len(col_data) * 100
                    percentiles.append(percentile)
                else:
                    percentiles.append(50)
            
            angles = np.linspace(0, 2 * np.pi, len(avail_m), endpoint=False).tolist()
            percentiles_closed = percentiles + percentiles[:1]
            angles_closed = angles + angles[:1]
            
            ax_radar.plot(angles_closed, percentiles_closed, 'o-', linewidth=2, color=self.category['primary'])
            ax_radar.fill(angles_closed, percentiles_closed, alpha=0.25, color=self.category['primary'])
            ax_radar.set_xticks(angles)
            ax_radar.set_xticklabels([m.replace('_', '\n')[:12] for m in avail_m], fontsize=9)
            ax_radar.set_ylim(0, 100)
            ax_radar.set_yticks([25, 50, 75, 100])
            ax_radar.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=8)
            ax_radar.grid(True)
            ax_radar.set_title('Performance Percentiles', fontsize=12, fontweight='bold', pad=20)
        
        # Table of raw statistics
        ax_stats = fig.add_subplot(gspec[1:3, 1])
        ax_stats.axis('tight')
        ax_stats.axis('off')
        
        stats_data = [
            ['Metric', 'Value'],
            ['Top Speed', f"{player.get('max_speed_99', player.get('max_speed_max', 0)):.1f} mph"],
            ['Separation (99th%)', f"{player.get('average_separation_99', 0):.2f} yds"],
            ['YAC Over Expected', f"{player.get('YACOE_MEAN', player.get('yac_ability', 0)):.2f}"],
            ['Route Complexity', f"{player.get('changedir_route_MEAN', player.get('route_diversity', 0)):.1f}"],
            ['Cut Separation', f"{player.get('cod_sep_generated_overall', 0):.2f} yds"],
            ['Total Plays', f"{player.get('total_plays', 0):.0f}"],
        ]
        
        table = ax_stats.table(cellText=stats_data, cellLoc='left',
                              colWidths=[0.6, 0.4], loc='center',
                              bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        for i in range(2):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Evaluate strengths and weaknesses based on percentile thresholds
        ax_notes = fig.add_subplot(gspec[3, :])
        ax_notes.axis('off')
        
        strengths = []
        weaknesses = []
        
        speed_col = 'speed_score' if 'speed_score' in self.df.columns else 'max_speed_99'
        sep_col = 'separation_consistency' if 'separation_consistency' in self.df.columns else 'average_separation_99'
        
        if speed_col in player.index and pd.notna(player.get(speed_col)):
            if player[speed_col] > self.df[speed_col].quantile(0.75):
                strengths.append("Elite speed threat")
            elif player[speed_col] < self.df[speed_col].quantile(0.25):
                weaknesses.append("Limited top-end speed")
        
        if sep_col in player.index and pd.notna(player.get(sep_col)):
            if player[sep_col] > self.df[sep_col].quantile(0.75):
                strengths.append("Consistent separator")
            elif player[sep_col] < self.df[sep_col].quantile(0.25):
                weaknesses.append("Struggles to separate")
        
        yac_col = 'yac_ability' if 'yac_ability' in player.index else 'YACOE_MEAN'
        if yac_col in player.index and pd.notna(player.get(yac_col)):
            if player[yac_col] > 0.5:
                strengths.append("YAC creator")
            elif player[yac_col] < -0.5:
                weaknesses.append("Limited after catch")
        
        if not strengths:
            strengths = ["Well-rounded profile"]
        if not weaknesses:
            weaknesses = ["No major weaknesses found"]
        
        notes_text = "Strengths:\n" + "\n".join(f"• {s}" for s in strengths[:3])
        notes_text += "\n\nWeaknesses:\n" + "\n".join(f"• {w}" for w in weaknesses[:2])
        
        ax_notes.text(0.5, 0.5, notes_text,
                     ha='center', va='center', fontsize=11,
                     transform=ax_notes.transAxes,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle('Player Scouting Report', fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
        return fig


def create_all_visuals(analyzer, output_dir='.', analysis_type=None):
    # Batch generate the full visualization suite for the analysis
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Auto-detect analysis type based on target variable if not specified
    if analysis_type is None:
        target_name = analyzer.results.get('predictions', {}).get('target_name', '')
        
        nfl_targets = ['targets_per_game', 'yards_per_game', 'receptions_per_game', 'catch_rate']
        draft_targets = ['draft_capital', 'draft_pick', 'draft_round', 'rec_yards', 'production_score']
        
        if target_name in nfl_targets:
            analysis_type = 'nfl'
        elif target_name in draft_targets:
            analysis_type = 'draft'
        else:
            # Check which target columns have sufficient data coverage
            nfl_data = sum(1 for col in nfl_targets if col in analyzer.df.columns and analyzer.df[col].notna().sum() > 10)
            draft_data = analyzer.df.get('draft_capital', pd.Series()).notna().sum()
            
            if nfl_data > 0 and analyzer.df.get('targets_per_game', pd.Series()).notna().sum() > 20:
                analysis_type = 'nfl'
            else:
                analysis_type = 'draft'
    
    prefix = 'nfl' if analysis_type == 'nfl' else 'draft'
    print(f"Generating visualization suite - {analysis_type.upper()} analysis mode")
    visualization = TrackVis(analyzer, analysis_type=analysis_type)
    
    visualization.plot_model_compare(save_path=f'{output_dir}/{prefix}_1_model_comparison.png')
    visualization.plot_feature_importance(outfile=f'{output_dir}/{prefix}_2_feature_importance.png')
    visualization.plot_act_vs_pred(fname=f'{output_dir}/{prefix}_3_actual_vs_predicted.png')
    visualization.plot_archetypes(output=f'{output_dir}/{prefix}_4_player_archetypes.png')
    visualization.plot_tracking_outcome(savename=f'{output_dir}/{prefix}_5_tracking_to_outcome.png')
    visualization.plot_value(path=f'{output_dir}/{prefix}_6_value_discovery.png')
    
    print(f"\nVisualization suite complete - all files saved to {output_dir}/")
    
    return visualization