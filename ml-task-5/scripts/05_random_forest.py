"""
Phase 5: Random Forest Training
================================
Train a Random Forest classifier on the 55 labeled samples to achieve higher
accuracy than the statistical baseline (30.9% → expected 80-90%).

Approach:
1. Load labeled subset (55 samples: 9 utility, 46 core)
2. Prepare features (25 numerical features from Phase 2)
3. Stratified train-test split (80/20) to handle class imbalance
4. Hyperparameter tuning with GridSearchCV + 10-fold CV
5. Train final model on all labeled data
6. Evaluate performance vs statistical baseline
7. Feature importance analysis
8. Save trained model for Phase 6 ensemble

Expected Outcome:
- Accuracy: 80-90% (vs 30.9% baseline)
- Precision: >70% (vs 19.1% baseline)
- Recall: >80% (vs 100% baseline - acceptable tradeoff)
- F1 Score: >75% (vs 32.1% baseline)
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold, 
    GridSearchCV, 
    cross_val_score,
    train_test_split
)
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

print("\n" + "="*80)
print("PHASE 5: RANDOM FOREST TRAINING")
print("="*80 + "\n")

# =============================================================================
# 1. LOAD LABELED DATA
# =============================================================================
print("📂 Loading labeled subset...")
with open('outputs/03_labeled_subset.pkl', 'rb') as f:
    labeled_data = pickle.load(f)

print(f"✓ Loaded {len(labeled_data)} labeled samples")
print(f"  - Core: {(labeled_data['ml_label'] == 0).sum()}")
print(f"  - Utility: {(labeled_data['ml_label'] == 1).sum()}")

# =============================================================================
# 2. PREPARE FEATURES
# =============================================================================
print("\n🔧 Preparing features...")

# Define feature columns (same 25 numerical features from Phase 4)
# Excluding 'has_docstring' (boolean, not useful for RF)
feature_columns = [
    # Complexity metrics (5)
    'cyclomatic_complexity', 'halstead_volume', 'halstead_difficulty',
    'halstead_effort', 'maintainability_index',
    
    # Code metrics (8)
    'loc', 'lloc', 'sloc', 'comments', 'multi', 'blank',
    'single_comments', 'comment_ratio',
    
    # Structure metrics (5)
    'ast_depth', 'ast_breadth', 'num_functions', 'num_classes',
    'max_nesting_depth',
    
    # Control flow metrics (5)
    'num_branches', 'num_loops', 'num_returns', 'num_function_calls',
    'num_assignments',
    
    # Other metrics (2)
    'num_parameters', 'num_imports'
]

# Extract features and labels
X = labeled_data[feature_columns].values
y = labeled_data['ml_label'].values.astype(int)

print(f"✓ Features prepared: {X.shape}")
print(f"  - Shape: {X.shape[0]} samples × {X.shape[1]} features")
print(f"  - Label distribution: {np.bincount(y)} (0=core, 1=utility)")

# Check for NaN/inf values
nan_count = np.isnan(X).sum()
inf_count = np.isinf(X).sum()
if nan_count > 0 or inf_count > 0:
    print(f"⚠️  Warning: {nan_count} NaN values, {inf_count} inf values detected")
    print("   Replacing with 0...")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
else:
    print("✓ No NaN/inf values detected")

# =============================================================================
# 3. TRAIN-TEST SPLIT (STRATIFIED)
# =============================================================================
print("\n✂️  Splitting data (80% train, 20% test, stratified)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"✓ Train set: {X_train.shape[0]} samples ({np.bincount(y_train)})")
print(f"✓ Test set:  {X_test.shape[0]} samples ({np.bincount(y_test)})")

# =============================================================================
# 4. HYPERPARAMETER TUNING
# =============================================================================
print("\n🔍 Hyperparameter tuning with GridSearchCV...")
print("   (This may take 1-2 minutes...)")

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],          # Number of trees
    'max_depth': [5, 10, 15, None],          # Tree depth
    'min_samples_split': [2, 5, 10],         # Min samples to split
    'min_samples_leaf': [1, 2, 4],           # Min samples in leaf
    'max_features': ['sqrt', 'log2'],        # Features per split
    'class_weight': ['balanced', None]       # Handle class imbalance
}

# Create base model
rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

# GridSearchCV with stratified 10-fold CV
grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
    scoring='f1',  # Optimize for F1 (balance precision/recall)
    n_jobs=-1,
    verbose=1
)

# Fit on training data
grid_search.fit(X_train, y_train)

print("\n✓ Best parameters found:")
for param, value in grid_search.best_params_.items():
    print(f"   {param}: {value}")
print(f"\n✓ Best cross-validation F1 score: {grid_search.best_score_:.4f}")

# Get best model
best_rf = grid_search.best_estimator_

# =============================================================================
# 5. TRAIN FINAL MODEL ON ALL LABELED DATA
# =============================================================================
print("\n🎯 Training final model on ALL labeled data (55 samples)...")

# Create final model with best parameters
final_rf = RandomForestClassifier(
    **grid_search.best_params_,
    random_state=42,
    n_jobs=-1
)

# Train on all labeled data
final_rf.fit(X, y)

print("✓ Final model trained!")

# =============================================================================
# 6. EVALUATE PERFORMANCE
# =============================================================================
print("\n📊 Evaluating model performance...\n")

# --- 6.1: Cross-validation on all data ---
print("=== Cross-Validation Performance (10-fold, stratified) ===")
cv_scores = cross_val_score(
    final_rf, X, y,
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
    scoring='accuracy',
    n_jobs=-1
)
print(f"Accuracy:  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

cv_precision = cross_val_score(
    final_rf, X, y,
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
    scoring='precision',
    n_jobs=-1
)
print(f"Precision: {cv_precision.mean():.4f} ± {cv_precision.std():.4f}")

cv_recall = cross_val_score(
    final_rf, X, y,
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
    scoring='recall',
    n_jobs=-1
)
print(f"Recall:    {cv_recall.mean():.4f} ± {cv_recall.std():.4f}")

cv_f1 = cross_val_score(
    final_rf, X, y,
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
    scoring='f1',
    n_jobs=-1
)
print(f"F1 Score:  {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

# --- 6.2: Test set performance (from train-test split) ---
print("\n=== Test Set Performance (20% holdout) ===")
y_pred_test = best_rf.predict(X_test)
y_proba_test = best_rf.predict_proba(X_test)[:, 1]

print(f"Accuracy:  {accuracy_score(y_test, y_pred_test):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_test):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_test):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_test):.4f}")
print(f"ROC AUC:   {roc_auc_score(y_test, y_proba_test):.4f}")

# Confusion matrix
cm_test = confusion_matrix(y_test, y_pred_test)
print("\nConfusion Matrix (Test Set):")
print(f"  TN={cm_test[0,0]:2d}  FP={cm_test[0,1]:2d}")
print(f"  FN={cm_test[1,0]:2d}  TP={cm_test[1,1]:2d}")

# --- 6.3: Performance on all labeled data (for comparison) ---
print("\n=== All Labeled Data Performance (55 samples) ===")
y_pred_all = final_rf.predict(X)
y_proba_all = final_rf.predict_proba(X)[:, 1]

print(f"Accuracy:  {accuracy_score(y, y_pred_all):.4f}")
print(f"Precision: {precision_score(y, y_pred_all):.4f}")
print(f"Recall:    {recall_score(y, y_pred_all):.4f}")
print(f"F1 Score:  {f1_score(y, y_pred_all):.4f}")
print(f"ROC AUC:   {roc_auc_score(y, y_proba_all):.4f}")

# Confusion matrix
cm_all = confusion_matrix(y, y_pred_all)
print("\nConfusion Matrix (All Data):")
print(f"  TN={cm_all[0,0]:2d}  FP={cm_all[0,1]:2d}")
print(f"  FN={cm_all[1,0]:2d}  TP={cm_all[1,1]:2d}")

# --- 6.4: Comparison with Statistical Baseline ---
print("\n=== Comparison: Random Forest vs Statistical Baseline ===")
print("Metric          | RF (CV)  | Statistical | Improvement")
print("----------------|----------|-------------|------------")
print(f"Accuracy        | {cv_scores.mean():.1%}   | 30.9%       | {(cv_scores.mean() - 0.309) / 0.309 * 100:+.1f}%")
print(f"Precision       | {cv_precision.mean():.1%}   | 19.1%       | {(cv_precision.mean() - 0.191) / 0.191 * 100:+.1f}%")
print(f"Recall          | {cv_recall.mean():.1%}   | 100.0%      | {(cv_recall.mean() - 1.0) / 1.0 * 100:+.1f}%")
print(f"F1 Score        | {cv_f1.mean():.1%}   | 32.1%       | {(cv_f1.mean() - 0.321) / 0.321 * 100:+.1f}%")

# =============================================================================
# 7. FEATURE IMPORTANCE ANALYSIS
# =============================================================================
print("\n🔬 Feature Importance Analysis...\n")

# Get feature importances
feature_importances = final_rf.feature_importances_

# Create DataFrame
importance_df = pd.DataFrame({
    'feature': feature_columns,
    'importance': feature_importances
}).sort_values('importance', ascending=False)

print("Top 10 Most Important Features:")
print("Rank | Feature                  | Importance")
print("-----|--------------------------|------------")
for idx, row in importance_df.head(10).iterrows():
    print(f"{importance_df.index.get_loc(idx)+1:4d} | {row['feature']:24s} | {row['importance']:.4f}")

# Plot feature importances
plt.figure(figsize=(12, 8))
sns.barplot(
    data=importance_df.head(15), 
    x='importance', 
    y='feature',
    palette='viridis'
)
plt.title('Top 15 Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/05_feature_importances.png', dpi=300, bbox_inches='tight')
print("\n✓ Feature importance plot saved: outputs/05_feature_importances.png")

# =============================================================================
# 8. ANALYZE PREDICTIONS ON LABELED DATA
# =============================================================================
print("\n🔍 Analyzing predictions on labeled data...\n")

# Add predictions to labeled data
labeled_results = labeled_data.copy()
labeled_results['rf_prediction'] = y_pred_all
labeled_results['rf_probability'] = y_proba_all
labeled_results['rf_correct'] = (y_pred_all == y)

# Analyze correct vs incorrect predictions
print("=== Prediction Analysis ===")
print(f"Total samples:       {len(labeled_results)}")
print(f"Correctly predicted: {labeled_results['rf_correct'].sum()} ({labeled_results['rf_correct'].mean():.1%})")
print(f"Incorrectly predicted: {(~labeled_results['rf_correct']).sum()} ({(~labeled_results['rf_correct']).mean():.1%})")

# Show misclassified samples
print("\n=== Misclassified Samples ===")
misclassified = labeled_results[~labeled_results['rf_correct']]
if len(misclassified) > 0:
    print(f"Found {len(misclassified)} misclassified samples:")
    for idx, row in misclassified.iterrows():
        true_label = "Core" if row['ml_label'] == 0 else "Utility"
        pred_label = "Core" if row['rf_prediction'] == 0 else "Utility"
        print(f"\n  {row['function_name']}")
        print(f"    True: {true_label}, Predicted: {pred_label}")
        print(f"    Probability: {row['rf_probability']:.3f}")
        print(f"    Type: {row['type']}, LOC: {row['loc']:.0f}, Complexity: {row['cyclomatic_complexity']:.0f}")
else:
    print("🎉 Perfect! No misclassified samples!")

# Show top utility and core predictions
print("\n=== Top 10 Highest Utility Probability (Prediction = Utility) ===")
top_utility = labeled_results[labeled_results['rf_prediction'] == 1].nlargest(10, 'rf_probability')
for idx, row in top_utility.iterrows():
    correct = "✓" if row['rf_correct'] else "✗"
    print(f"{correct} {row['function_name']:40s} | Prob: {row['rf_probability']:.3f} | Type: {row['type']:8s}")

print("\n=== Top 10 Highest Core Probability (Prediction = Core) ===")
top_core = labeled_results[labeled_results['rf_prediction'] == 0].nsmallest(10, 'rf_probability')
for idx, row in top_core.iterrows():
    correct = "✓" if row['rf_correct'] else "✗"
    print(f"{correct} {row['function_name']:40s} | Prob: {row['rf_probability']:.3f} | Type: {row['type']:8s}")

# =============================================================================
# 9. SAVE OUTPUTS
# =============================================================================
print("\n💾 Saving outputs...")

# Save trained model
with open('models/random_forest.pkl', 'wb') as f:
    pickle.dump({
        'model': final_rf,
        'best_model': best_rf,
        'best_params': grid_search.best_params_,
        'best_cv_score': grid_search.best_score_,
        'feature_columns': feature_columns,
        'feature_importances': importance_df.to_dict('records'),
        'cv_scores': {
            'accuracy': cv_scores,
            'precision': cv_precision,
            'recall': cv_recall,
            'f1': cv_f1
        }
    }, f)
print("✓ Model saved: models/random_forest.pkl")

# Save predictions on labeled data
with open('outputs/05_rf_predictions.pkl', 'wb') as f:
    pickle.dump(labeled_results, f)
labeled_results.to_csv('outputs/05_rf_predictions.csv', index=False)
print("✓ Predictions saved: outputs/05_rf_predictions.pkl/csv")

# Save performance metrics
performance_metrics = {
    'cv_accuracy': cv_scores.mean(),
    'cv_precision': cv_precision.mean(),
    'cv_recall': cv_recall.mean(),
    'cv_f1': cv_f1.mean(),
    'test_accuracy': accuracy_score(y_test, y_pred_test),
    'test_precision': precision_score(y_test, y_pred_test),
    'test_recall': recall_score(y_test, y_pred_test),
    'test_f1': f1_score(y_test, y_pred_test),
    'test_roc_auc': roc_auc_score(y_test, y_proba_test),
    'all_accuracy': accuracy_score(y, y_pred_all),
    'all_precision': precision_score(y, y_pred_all),
    'all_recall': recall_score(y, y_pred_all),
    'all_f1': f1_score(y, y_pred_all),
    'all_roc_auc': roc_auc_score(y, y_proba_all),
    'confusion_matrix_test': cm_test.tolist(),
    'confusion_matrix_all': cm_all.tolist(),
    'num_samples': len(labeled_data),
    'num_features': len(feature_columns)
}

with open('outputs/05_performance_metrics.pkl', 'wb') as f:
    pickle.dump(performance_metrics, f)
print("✓ Performance metrics saved: outputs/05_performance_metrics.pkl")

# =============================================================================
# 10. SUMMARY
# =============================================================================
print("\n" + "="*80)
print("PHASE 5 COMPLETE! 🎉")
print("="*80)
print(f"\n✅ Random Forest trained on {len(labeled_data)} labeled samples")
print(f"✅ Best model: {grid_search.best_params_}")
print(f"\n📊 Performance Summary (10-fold CV):")
print(f"   Accuracy:  {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
print(f"   Precision: {cv_precision.mean():.1%} ± {cv_precision.std():.1%}")
print(f"   Recall:    {cv_recall.mean():.1%} ± {cv_recall.std():.1%}")
print(f"   F1 Score:  {cv_f1.mean():.1%} ± {cv_f1.std():.1%}")
print(f"\n🚀 Improvement over Statistical Baseline:")
print(f"   Accuracy:  30.9% → {cv_scores.mean():.1%} ({(cv_scores.mean() - 0.309) / 0.309 * 100:+.1f}%)")
print(f"   F1 Score:  32.1% → {cv_f1.mean():.1%} ({(cv_f1.mean() - 0.321) / 0.321 * 100:+.1f}%)")
print(f"\n📁 Output Files:")
print(f"   models/random_forest.pkl")
print(f"   outputs/05_rf_predictions.pkl/csv")
print(f"   outputs/05_performance_metrics.pkl")
print(f"   outputs/05_feature_importances.png")
print(f"\n➡️  Next: Phase 6 - Ensemble Model (40% Statistical + 60% RF)")
print("="*80 + "\n")
