"""
Phase 6: Ensemble Model
========================
Combine Statistical Baseline (40%) + Random Forest (60%) to predict labels
for the 236 unlabeled functions.

Approach:
1. Load statistical scores from Phase 4
2. Load Random Forest model from Phase 5
3. Load all 291 functions (55 labeled + 236 unlabeled)
4. Generate predictions for unlabeled functions:
   - Statistical score (inverted - lower = utility)
   - RF probability (higher = utility)
   - Ensemble: 0.4 × (1 - stat_score) + 0.6 × rf_prob
5. Apply threshold to classify (ensemble_score > 0.5 = utility)
6. Evaluate on labeled subset (validation)
7. Rank all functions by ensemble score
8. Save final predictions

Expected Outcome:
- Combine interpretability (statistical) with accuracy (RF)
- Better performance than either model alone
- Final utility scores for all 291 functions
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

print("\n" + "="*80)
print("PHASE 6: ENSEMBLE MODEL")
print("="*80 + "\n")

# =============================================================================
# 1. LOAD MODELS AND DATA
# =============================================================================
print("📂 Loading models and data...")

# Load statistical baseline
with open('outputs/04_statistical_scores.pkl', 'rb') as f:
    stat_data = pickle.load(f)
print(f"✓ Statistical scores loaded: {len(stat_data)} functions")

# Load Random Forest model
with open('models/random_forest.pkl', 'rb') as f:
    rf_artifacts = pickle.load(f)
    rf_model = rf_artifacts['model']
    feature_columns = rf_artifacts['feature_columns']
print(f"✓ Random Forest model loaded: {len(feature_columns)} features")

# Load all data (labeled + unlabeled)
with open('outputs/03_labeled_data.pkl', 'rb') as f:
    all_data = pickle.load(f)
print(f"✓ All data loaded: {len(all_data)} functions")

# Split labeled vs unlabeled
labeled_data = all_data[all_data['ml_label'].notna()].copy()
unlabeled_data = all_data[all_data['ml_label'].isna()].copy()
print(f"  - Labeled: {len(labeled_data)} (for validation)")
print(f"  - Unlabeled: {len(unlabeled_data)} (for prediction)")

# =============================================================================
# 2. PREPARE FEATURES FOR RF PREDICTION
# =============================================================================
print("\n🔧 Preparing features for Random Forest...")

# Extract features for all data
X_all = all_data[feature_columns].values
X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
print(f"✓ Features prepared: {X_all.shape}")

# =============================================================================
# 3. GENERATE PREDICTIONS
# =============================================================================
print("\n🔮 Generating predictions...")

# --- 3.1: Statistical scores ---
# Statistical model: higher score = more utility-like (from Phase 4)
# No need to invert
all_data['stat_score'] = stat_data['statistical_score'].values
all_data['stat_utility_prob'] = all_data['stat_score']  # Already utility probability
print(f"✓ Statistical scores computed")
print(f"  Range: {all_data['stat_utility_prob'].min():.3f} - {all_data['stat_utility_prob'].max():.3f}")

# --- 3.2: Random Forest probabilities ---
rf_proba = rf_model.predict_proba(X_all)
# RF: class 0 = core, class 1 = utility
# So rf_proba[:, 1] gives utility probability
all_data['rf_utility_prob'] = rf_proba[:, 1]
print(f"✓ Random Forest probabilities computed")
print(f"  Range: {all_data['rf_utility_prob'].min():.3f} - {all_data['rf_utility_prob'].max():.3f}")

# --- 3.3: Ensemble scores ---
# Weighted combination: 40% statistical + 60% RF
STAT_WEIGHT = 0.4
RF_WEIGHT = 0.6

all_data['ensemble_score'] = (
    STAT_WEIGHT * all_data['stat_utility_prob'] + 
    RF_WEIGHT * all_data['rf_utility_prob']
)
print(f"✓ Ensemble scores computed: {STAT_WEIGHT:.0%} Stat + {RF_WEIGHT:.0%} RF")
print(f"  Range: {all_data['ensemble_score'].min():.3f} - {all_data['ensemble_score'].max():.3f}")

# --- 3.4: Classifications ---
# Default threshold: 0.5
THRESHOLD = 0.5
all_data['ensemble_prediction'] = (all_data['ensemble_score'] > THRESHOLD).astype(int)
print(f"✓ Predictions created (threshold={THRESHOLD})")
print(f"  Predicted Core (0): {(all_data['ensemble_prediction'] == 0).sum()}")
print(f"  Predicted Utility (1): {(all_data['ensemble_prediction'] == 1).sum()}")

# =============================================================================
# 4. EVALUATE ON LABELED SUBSET
# =============================================================================
print("\n📊 Evaluating ensemble on labeled subset...")

# Filter to labeled data only
labeled_results = all_data[all_data['ml_label'].notna()].copy()
y_true = labeled_results['ml_label'].values.astype(int)
y_pred = labeled_results['ensemble_prediction'].values
y_score = labeled_results['ensemble_score'].values

# Compute metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_score)

print(f"\n=== Ensemble Performance (Labeled Data: {len(labeled_results)} samples) ===")
print(f"Accuracy:  {accuracy:.4f} ({accuracy:.1%})")
print(f"Precision: {precision:.4f} ({precision:.1%})")
print(f"Recall:    {recall:.4f} ({recall:.1%})")
print(f"F1 Score:  {f1:.4f} ({f1:.1%})")
print(f"ROC AUC:   {roc_auc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print(f"\nConfusion Matrix:")
print(f"  TN={cm[0,0]:2d}  FP={cm[0,1]:2d}")
print(f"  FN={cm[1,0]:2d}  TP={cm[1,1]:2d}")

# Classification report
print(f"\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Core', 'Utility']))

# =============================================================================
# 5. COMPARE WITH INDIVIDUAL MODELS
# =============================================================================
print("\n📈 Comparing Ensemble vs Individual Models...")

# Load Phase 4 results (statistical baseline on labeled data)
with open('outputs/05_rf_predictions.pkl', 'rb') as f:
    rf_labeled = pickle.load(f)

# Get RF-only performance on labeled data
rf_pred = rf_labeled['rf_prediction'].values
rf_acc = accuracy_score(y_true, rf_pred)
rf_prec = precision_score(y_true, rf_pred)
rf_rec = recall_score(y_true, rf_pred)
rf_f1 = f1_score(y_true, rf_pred)

# Statistical baseline performance (from Phase 4 summary)
stat_acc = 0.309
stat_prec = 0.191
stat_rec = 1.0
stat_f1 = 0.321

print("\nMetric          | Ensemble | RF Only  | Statistical | Best Model")
print("----------------|----------|----------|-------------|------------")
print(f"Accuracy        | {accuracy:.1%}   | {rf_acc:.1%}   | {stat_acc:.1%}      | {'Ensemble' if accuracy >= rf_acc else 'RF'}")
print(f"Precision       | {precision:.1%}   | {rf_prec:.1%}   | {stat_prec:.1%}      | {'Ensemble' if precision >= rf_prec else 'RF'}")
print(f"Recall          | {recall:.1%}   | {rf_rec:.1%}   | {stat_rec:.1%}      | {'Ensemble' if recall >= rf_rec else 'Statistical'}")
print(f"F1 Score        | {f1:.1%}   | {rf_f1:.1%}   | {stat_f1:.1%}      | {'Ensemble' if f1 >= rf_f1 else 'RF'}")

# =============================================================================
# 6. ANALYZE PREDICTIONS ON UNLABELED DATA
# =============================================================================
print("\n🔍 Analyzing predictions on unlabeled data...\n")

unlabeled_results = all_data[all_data['ml_label'].isna()].copy()
print(f"=== Unlabeled Data Predictions ({len(unlabeled_results)} functions) ===")
print(f"Predicted Core (0):    {(unlabeled_results['ensemble_prediction'] == 0).sum()}")
print(f"Predicted Utility (1): {(unlabeled_results['ensemble_prediction'] == 1).sum()}")

print(f"\nEnsemble Score Statistics (Unlabeled):")
print(f"  Mean:   {unlabeled_results['ensemble_score'].mean():.3f}")
print(f"  Median: {unlabeled_results['ensemble_score'].median():.3f}")
print(f"  Std:    {unlabeled_results['ensemble_score'].std():.3f}")
print(f"  Min:    {unlabeled_results['ensemble_score'].min():.3f}")
print(f"  Max:    {unlabeled_results['ensemble_score'].max():.3f}")

# =============================================================================
# 7. TOP UTILITY AND CORE FUNCTIONS
# =============================================================================
print("\n🏆 Top 20 Highest Utility Functions (ALL DATA)...")
print("Rank | Function Name                          | Type     | Ensemble | Stat  | RF    | Label")
print("-----|----------------------------------------|----------|----------|-------|-------|--------")
top_utility = all_data.nlargest(20, 'ensemble_score')
for rank, (idx, row) in enumerate(top_utility.iterrows(), 1):
    label = "✓" if pd.notna(row['ml_label']) and row['ml_label'] == 1 else ("✗" if pd.notna(row['ml_label']) else "-")
    print(f"{rank:4d} | {row['function_name']:38s} | {row['type']:8s} | {row['ensemble_score']:.3f}    | {row['stat_utility_prob']:.3f} | {row['rf_utility_prob']:.3f} | {label:6s}")

print("\n🏗️  Top 20 Highest Core Functions (ALL DATA)...")
print("Rank | Function Name                          | Type     | Ensemble | Stat  | RF    | Label")
print("-----|----------------------------------------|----------|----------|-------|-------|--------")
top_core = all_data.nsmallest(20, 'ensemble_score')
for rank, (idx, row) in enumerate(top_core.iterrows(), 1):
    label = "✓" if pd.notna(row['ml_label']) and row['ml_label'] == 0 else ("✗" if pd.notna(row['ml_label']) else "-")
    print(f"{rank:4d} | {row['function_name']:38s} | {row['type']:8s} | {row['ensemble_score']:.3f}    | {row['stat_utility_prob']:.3f} | {row['rf_utility_prob']:.3f} | {label:6s}")

# =============================================================================
# 8. SCORE DISTRIBUTION PLOTS
# =============================================================================
print("\n📊 Creating score distribution plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Ensemble score distribution by label (labeled data only)
ax = axes[0, 0]
labeled_results['label_name'] = labeled_results['ml_label'].map({0: 'Core', 1: 'Utility'})
sns.boxplot(data=labeled_results, x='label_name', y='ensemble_score', ax=ax, palette='Set2')
ax.axhline(THRESHOLD, color='red', linestyle='--', label=f'Threshold={THRESHOLD}')
ax.set_title('Ensemble Score Distribution (Labeled Data)', fontweight='bold')
ax.set_xlabel('True Label')
ax.set_ylabel('Ensemble Score')
ax.legend()

# Plot 2: Statistical vs RF scatter (labeled data)
ax = axes[0, 1]
colors = labeled_results['ml_label'].map({0: 'blue', 1: 'red'})
ax.scatter(labeled_results['stat_utility_prob'], labeled_results['rf_utility_prob'], 
           c=colors, alpha=0.6, s=100, edgecolors='black')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.set_title('Statistical vs RF Probabilities (Labeled)', fontweight='bold')
ax.set_xlabel('Statistical Utility Probability')
ax.set_ylabel('RF Utility Probability')
ax.legend(['Diagonal', 'Core', 'Utility'])

# Plot 3: Ensemble score histogram (all data)
ax = axes[1, 0]
ax.hist(all_data['ensemble_score'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
ax.axvline(THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Threshold={THRESHOLD}')
ax.set_title('Ensemble Score Distribution (All 291 Functions)', fontweight='bold')
ax.set_xlabel('Ensemble Score')
ax.set_ylabel('Frequency')
ax.legend()

# Plot 4: Model contribution (labeled data)
ax = axes[1, 1]
contributions = pd.DataFrame({
    'Statistical': labeled_results['stat_utility_prob'] * STAT_WEIGHT,
    'Random Forest': labeled_results['rf_utility_prob'] * RF_WEIGHT
})
contributions.plot(kind='box', ax=ax)
ax.set_title(f'Model Contributions (40% Stat + 60% RF)', fontweight='bold')
ax.set_ylabel('Weighted Contribution to Ensemble')

plt.tight_layout()
plt.savefig('outputs/06_ensemble_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Plots saved: outputs/06_ensemble_analysis.png")

# =============================================================================
# 9. SAVE OUTPUTS
# =============================================================================
print("\n💾 Saving outputs...")

# Save all predictions
with open('outputs/06_ensemble_predictions.pkl', 'wb') as f:
    pickle.dump(all_data, f)
all_data.to_csv('outputs/06_ensemble_predictions.csv', index=False)
print("✓ All predictions saved: outputs/06_ensemble_predictions.pkl/csv")

# Save unlabeled predictions only
with open('outputs/06_unlabeled_predictions.pkl', 'wb') as f:
    pickle.dump(unlabeled_results, f)
unlabeled_results.to_csv('outputs/06_unlabeled_predictions.csv', index=False)
print("✓ Unlabeled predictions saved: outputs/06_unlabeled_predictions.pkl/csv")

# Save performance metrics
ensemble_metrics = {
    'labeled_performance': {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist()
    },
    'unlabeled_stats': {
        'total': len(unlabeled_results),
        'predicted_core': (unlabeled_results['ensemble_prediction'] == 0).sum(),
        'predicted_utility': (unlabeled_results['ensemble_prediction'] == 1).sum(),
        'score_mean': float(unlabeled_results['ensemble_score'].mean()),
        'score_median': float(unlabeled_results['ensemble_score'].median()),
        'score_std': float(unlabeled_results['ensemble_score'].std())
    },
    'weights': {
        'statistical': STAT_WEIGHT,
        'random_forest': RF_WEIGHT
    },
    'threshold': THRESHOLD
}

with open('outputs/06_ensemble_metrics.pkl', 'wb') as f:
    pickle.dump(ensemble_metrics, f)
print("✓ Metrics saved: outputs/06_ensemble_metrics.pkl")

# =============================================================================
# 10. SUMMARY
# =============================================================================
print("\n" + "="*80)
print("PHASE 6 COMPLETE! 🎉")
print("="*80)
print(f"\n✅ Ensemble model created: {STAT_WEIGHT:.0%} Statistical + {RF_WEIGHT:.0%} Random Forest")
print(f"\n📊 Performance on {len(labeled_results)} Labeled Samples:")
print(f"   Accuracy:  {accuracy:.1%}")
print(f"   Precision: {precision:.1%}")
print(f"   Recall:    {recall:.1%}")
print(f"   F1 Score:  {f1:.1%}")
print(f"   ROC AUC:   {roc_auc:.3f}")
print(f"\n🔮 Predictions for {len(unlabeled_results)} Unlabeled Functions:")
print(f"   Predicted Core:    {(unlabeled_results['ensemble_prediction'] == 0).sum()}")
print(f"   Predicted Utility: {(unlabeled_results['ensemble_prediction'] == 1).sum()}")
print(f"\n📁 Output Files:")
print(f"   outputs/06_ensemble_predictions.pkl/csv (all 291 functions)")
print(f"   outputs/06_unlabeled_predictions.pkl/csv (236 unlabeled)")
print(f"   outputs/06_ensemble_metrics.pkl")
print(f"   outputs/06_ensemble_analysis.png")
print(f"\n➡️  Next: Phase 7 - Final Validation & Quality Checks")
print("="*80 + "\n")
