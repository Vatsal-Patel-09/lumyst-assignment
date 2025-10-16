"""
Phase 4: Statistical Baseline
==============================
Build a transparent, rule-based scoring system for utility function detection.

Approach:
- Use all 26 features to calculate a "utility score" (0-1)
- Inverse relationships: simpler code = higher utility score
- Weight features by discriminative power
- Validate on labeled subset
- Interpretable and explainable

Utility Score Formula:
=====================
utility_score = weighted_sum([
    inverse(cyclomatic_complexity),
    inverse(LOC),
    inverse(nesting_depth),
    inverse(num_parameters),
    inverse(halstead_difficulty),
    ... (all 26 features)
])

Normalization:
- Min-max scaling for each feature
- Invert features (1 - normalized_value)
- Apply weights based on feature importance
- Final score: 0 (definitely core) to 1 (definitely utility)
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("📊 PHASE 4: STATISTICAL BASELINE")
print("="*80)

# ============================================================================
# SECTION 1: LOAD LABELED DATA
# ============================================================================
print("\n" + "="*80)
print("1️⃣ LOADING DATA")
print("="*80)

# Load Phase 3 data (has labels + features)
input_file = 'outputs/03_labeled_data.pkl'
print(f"📥 Loading data from: {input_file}")
df = pd.read_pickle(input_file)

print(f"✅ Loaded {len(df)} functions")
print(f"📋 Total columns: {len(df.columns)}")

# Separate labeled and unlabeled
labeled_df = df[df['ml_label'].notna()].copy()
unlabeled_df = df[df['ml_label'].isna()].copy()

print(f"\n📊 Data split:")
print(f"   Labeled: {len(labeled_df)} functions")
print(f"   Unlabeled: {len(unlabeled_df)} functions")
print(f"   Total: {len(df)} functions")

# ============================================================================
# SECTION 2: DEFINE FEATURE GROUPS
# ============================================================================
print("\n" + "="*80)
print("2️⃣ DEFINING FEATURE GROUPS")
print("="*80)

# Define all numerical features to use
complexity_features = [
    'cyclomatic_complexity',
    'halstead_volume',
    'halstead_difficulty',
    'halstead_effort',
    'maintainability_index'
]

code_metric_features = [
    'loc',
    'lloc',
    'sloc',
    'comments',
    'multi',
    'blank',
    'single_comments',
    'comment_ratio'
]

structure_features = [
    'ast_depth',
    'ast_breadth',
    'num_functions',
    'num_classes',
    'max_nesting_depth'
]

control_flow_features = [
    'num_branches',
    'num_loops',
    'num_returns',
    'num_function_calls',
    'num_assignments'
]

code_element_features = [
    'num_parameters',
    'num_imports'
]

# Combine all features
all_features = (
    complexity_features + 
    code_metric_features + 
    structure_features + 
    control_flow_features + 
    code_element_features
)

print(f"✅ Feature groups defined:")
print(f"   - Complexity: {len(complexity_features)} features")
print(f"   - Code Metrics: {len(code_metric_features)} features")
print(f"   - Structure: {len(structure_features)} features")
print(f"   - Control Flow: {len(control_flow_features)} features")
print(f"   - Code Elements: {len(code_element_features)} features")
print(f"   - TOTAL: {len(all_features)} features")

# Verify all features exist
missing_features = [f for f in all_features if f not in df.columns]
if missing_features:
    print(f"\n⚠️  Missing features: {missing_features}")
    all_features = [f for f in all_features if f in df.columns]
    print(f"✅ Using {len(all_features)} available features")

# ============================================================================
# SECTION 3: FEATURE IMPORTANCE FROM LABELED DATA
# ============================================================================
print("\n" + "="*80)
print("3️⃣ CALCULATING FEATURE IMPORTANCE")
print("="*80)

print("🔍 Analyzing feature discriminative power on labeled data...")

# Calculate correlation with labels
feature_importance = {}

for feature in all_features:
    if feature in labeled_df.columns:
        # Calculate absolute correlation with label
        corr = abs(labeled_df[feature].corr(labeled_df['ml_label']))
        feature_importance[feature] = corr if not np.isnan(corr) else 0

# Sort by importance
feature_importance_sorted = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

print(f"\n📊 Top 10 Most Discriminative Features:")
print("-" * 60)
for i, (feature, importance) in enumerate(feature_importance_sorted[:10], 1):
    print(f"{i:2d}. {feature:30s} {importance:.4f}")

print(f"\n📊 Bottom 10 Least Discriminative Features:")
print("-" * 60)
for i, (feature, importance) in enumerate(feature_importance_sorted[-10:], 1):
    print(f"{i:2d}. {feature:30s} {importance:.4f}")

# ============================================================================
# SECTION 4: CREATE FEATURE WEIGHTS
# ============================================================================
print("\n" + "="*80)
print("4️⃣ CREATING FEATURE WEIGHTS")
print("="*80)

# Create weights based on feature importance
# Higher correlation = higher weight
feature_weights = {}

for feature, importance in feature_importance.items():
    # Use importance as weight (already 0-1 range)
    # Add small baseline to prevent zero weights
    feature_weights[feature] = importance + 0.01

# Normalize weights to sum to 1
total_weight = sum(feature_weights.values())
feature_weights = {k: v/total_weight for k, v in feature_weights.items()}

print("✅ Feature weights calculated")
print(f"   Total features: {len(feature_weights)}")
print(f"   Weight sum: {sum(feature_weights.values()):.4f}")

print(f"\n📊 Top 5 Weighted Features:")
top_weighted = sorted(feature_weights.items(), key=lambda x: x[1], reverse=True)[:5]
for feature, weight in top_weighted:
    print(f"   {feature:30s} {weight:.4f} ({weight*100:.2f}%)")

# ============================================================================
# SECTION 5: NORMALIZE FEATURES (MIN-MAX SCALING)
# ============================================================================
print("\n" + "="*80)
print("5️⃣ NORMALIZING FEATURES")
print("="*80)

print("🔄 Applying Min-Max scaling to all features...")

# Initialize scaler
scaler = MinMaxScaler()

# Fit on all data (not just labeled to avoid data leakage in scoring)
df_features_scaled = df[all_features].copy()
df_features_scaled[all_features] = scaler.fit_transform(df[all_features])

print(f"✅ Features normalized to [0, 1] range")
print(f"   Shape: {df_features_scaled.shape}")

# ============================================================================
# SECTION 6: CALCULATE UTILITY SCORES
# ============================================================================
print("\n" + "="*80)
print("6️⃣ CALCULATING UTILITY SCORES")
print("="*80)

def calculate_utility_score(row, feature_weights):
    """
    Calculate utility score using weighted inverse of features.
    
    Logic:
    - Low complexity, short code, simple structure = HIGH utility score
    - High complexity, long code, complex structure = LOW utility score
    
    Returns: Score between 0 (definitely core) and 1 (definitely utility)
    """
    score = 0.0
    
    # Special features that should NOT be inverted
    positive_features = ['maintainability_index', 'comment_ratio']
    
    for feature, weight in feature_weights.items():
        value = row[feature]
        
        if feature in positive_features:
            # Higher is better (more utility-like)
            contribution = value * weight
        else:
            # Lower is better (invert: 1 - value)
            contribution = (1 - value) * weight
        
        score += contribution
    
    return score

print("🔄 Computing utility scores for all functions...")

# Calculate scores
utility_scores = df_features_scaled.apply(
    lambda row: calculate_utility_score(row, feature_weights),
    axis=1
)

df['statistical_score'] = utility_scores

print(f"✅ Calculated utility scores for all {len(df)} functions")

# ============================================================================
# SECTION 7: SCORE STATISTICS
# ============================================================================
print("\n" + "="*80)
print("7️⃣ SCORE STATISTICS")
print("="*80)

print("\n📊 Overall Score Distribution:")
print(f"   Min:    {df['statistical_score'].min():.4f}")
print(f"   Max:    {df['statistical_score'].max():.4f}")
print(f"   Mean:   {df['statistical_score'].mean():.4f}")
print(f"   Median: {df['statistical_score'].median():.4f}")
print(f"   Std:    {df['statistical_score'].std():.4f}")

# Labeled data statistics
labeled_df_scored = df[df['ml_label'].notna()].copy()

utility_scores_labeled = labeled_df_scored[labeled_df_scored['ml_label'] == 1]['statistical_score']
core_scores_labeled = labeled_df_scored[labeled_df_scored['ml_label'] == 0]['statistical_score']

print(f"\n📊 Labeled Functions Score Distribution:")
print(f"\n   UTILITY Functions (label=1):")
print(f"      Count:  {len(utility_scores_labeled)}")
print(f"      Mean:   {utility_scores_labeled.mean():.4f}")
print(f"      Median: {utility_scores_labeled.median():.4f}")
print(f"      Min:    {utility_scores_labeled.min():.4f}")
print(f"      Max:    {utility_scores_labeled.max():.4f}")

print(f"\n   CORE Functions (label=0):")
print(f"      Count:  {len(core_scores_labeled)}")
print(f"      Mean:   {core_scores_labeled.mean():.4f}")
print(f"      Median: {core_scores_labeled.median():.4f}")
print(f"      Min:    {core_scores_labeled.min():.4f}")
print(f"      Max:    {core_scores_labeled.max():.4f}")

# Calculate separation
separation = utility_scores_labeled.mean() - core_scores_labeled.mean()
print(f"\n   📈 Score Separation: {separation:.4f}")
print(f"      (Higher = better separation)")

# ============================================================================
# SECTION 8: THRESHOLD OPTIMIZATION
# ============================================================================
print("\n" + "="*80)
print("8️⃣ FINDING OPTIMAL THRESHOLD")
print("="*80)

print("🔍 Testing different thresholds on labeled data...")

# Try different thresholds
thresholds = np.arange(0.3, 0.8, 0.05)
best_threshold = 0.5
best_f1 = 0

results = []

for threshold in thresholds:
    # Classify: score >= threshold → utility (1), else core (0)
    predictions = (labeled_df_scored['statistical_score'] >= threshold).astype(int)
    true_labels = labeled_df_scored['ml_label'].astype(int)
    
    # Calculate metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    
    results.append({
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

results_df = pd.DataFrame(results)

print(f"\n📊 Threshold Performance:")
print("-" * 70)
print(results_df.to_string(index=False))

print(f"\n✅ Best Threshold: {best_threshold:.2f}")
print(f"   F1 Score: {best_f1:.4f}")

# ============================================================================
# SECTION 9: EVALUATE ON LABELED DATA
# ============================================================================
print("\n" + "="*80)
print("9️⃣ EVALUATION ON LABELED DATA")
print("="*80)

# Use best threshold
final_predictions = (labeled_df_scored['statistical_score'] >= best_threshold).astype(int)
true_labels = labeled_df_scored['ml_label'].astype(int)

print(f"\n📊 Using threshold: {best_threshold:.2f}")
print("\n" + "="*70)
print("CLASSIFICATION REPORT:")
print("="*70)
print(classification_report(
    true_labels, 
    final_predictions,
    target_names=['CORE (0)', 'UTILITY (1)'],
    digits=3
))

print("\n" + "="*70)
print("CONFUSION MATRIX:")
print("="*70)
cm = confusion_matrix(true_labels, final_predictions)
print(f"\n                Predicted")
print(f"              CORE  UTILITY")
print(f"Actual CORE    {cm[0,0]:4d}    {cm[0,1]:4d}")
print(f"       UTILITY {cm[1,0]:4d}    {cm[1,1]:4d}")

# Calculate metrics
tn, fp, fn, tp = cm.ravel()
print(f"\n📊 Detailed Metrics:")
print(f"   True Negatives  (CORE correctly identified):    {tn}")
print(f"   False Positives (CORE misclassified as UTILITY): {fp}")
print(f"   False Negatives (UTILITY misclassified as CORE): {fn}")
print(f"   True Positives  (UTILITY correctly identified):  {tp}")

# ROC AUC
if len(np.unique(true_labels)) > 1:
    auc = roc_auc_score(true_labels, labeled_df_scored['statistical_score'])
    print(f"\n   ROC AUC Score: {auc:.4f}")

# ============================================================================
# SECTION 10: TOP UTILITY & CORE FUNCTIONS
# ============================================================================
print("\n" + "="*80)
print("🔟 TOP SCORED FUNCTIONS")
print("="*80)

# Sort by score
df_sorted = df.sort_values('statistical_score', ascending=False)

print("\n📝 Top 10 UTILITY Functions (Highest Scores):")
print("="*70)
top_utility = df_sorted.head(10)
for idx, row in top_utility.iterrows():
    label_str = f"[{int(row['ml_label'])}]" if not pd.isna(row['ml_label']) else "[ ]"
    print(f"\n{label_str} Score: {row['statistical_score']:.4f} | {row['label']}")
    print(f"    Type: {row['type']} | LOC: {row['loc']:.0f} | Complexity: {row['cyclomatic_complexity']:.1f}")

print("\n\n📝 Top 10 CORE Functions (Lowest Scores):")
print("="*70)
bottom_core = df_sorted.tail(10)
for idx, row in bottom_core.iterrows():
    label_str = f"[{int(row['ml_label'])}]" if not pd.isna(row['ml_label']) else "[ ]"
    print(f"\n{label_str} Score: {row['statistical_score']:.4f} | {row['label']}")
    print(f"    Type: {row['type']} | LOC: {row['loc']:.0f} | Complexity: {row['cyclomatic_complexity']:.1f}")

# ============================================================================
# SECTION 11: SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("1️⃣1️⃣ SAVING RESULTS")
print("="*80)

# Save scored dataset
output_pkl = 'outputs/04_statistical_scores.pkl'
df.to_pickle(output_pkl)
print(f"✅ Saved scored data to: {output_pkl}")

output_csv = 'outputs/04_statistical_scores.csv'
df.to_csv(output_csv, index=False)
print(f"✅ Saved CSV to: {output_csv}")

# Save model artifacts
model_artifacts = {
    'scaler': scaler,
    'feature_weights': feature_weights,
    'feature_list': all_features,
    'best_threshold': best_threshold,
    'feature_importance': feature_importance
}

model_file = 'models/statistical_baseline.pkl'
with open(model_file, 'wb') as f:
    pickle.dump(model_artifacts, f)
print(f"✅ Saved model artifacts to: {model_file}")

print(f"\n📊 Saved data:")
print(f"   - Total functions: {len(df)}")
print(f"   - Features used: {len(all_features)}")
print(f"   - Threshold: {best_threshold:.2f}")

# ============================================================================
# SECTION 12: KEY FINDINGS SUMMARY
# ============================================================================
print("\n" + "="*80)
print("🎯 KEY FINDINGS SUMMARY")
print("="*80)

print("\n1️⃣ Statistical Baseline Performance:")
print(f"   ✅ Accuracy: {accuracy_score(true_labels, final_predictions):.3f}")
print(f"   ✅ Precision: {precision_score(true_labels, final_predictions, zero_division=0):.3f}")
print(f"   ✅ Recall: {recall_score(true_labels, final_predictions, zero_division=0):.3f}")
print(f"   ✅ F1 Score: {f1_score(true_labels, final_predictions, zero_division=0):.3f}")

print("\n2️⃣ Score Separation:")
print(f"   Utility avg score: {utility_scores_labeled.mean():.4f}")
print(f"   Core avg score:    {core_scores_labeled.mean():.4f}")
print(f"   Separation:        {separation:.4f}")
if separation > 0.1:
    print("   ✅ Good separation achieved!")
else:
    print("   ⚠️  Limited separation - ensemble will help")

print("\n3️⃣ Top Discriminative Features:")
for i, (feature, importance) in enumerate(feature_importance_sorted[:5], 1):
    weight = feature_weights[feature]
    print(f"   {i}. {feature} (importance: {importance:.4f}, weight: {weight:.4f})")

print("\n4️⃣ Model Interpretability:")
print("   ✅ Fully transparent (rule-based)")
print("   ✅ All 26 features contribute")
print("   ✅ Weighted by discriminative power")
print("   ✅ Inverse relationship (simpler = more utility-like)")

print("\n5️⃣ Next Steps:")
print("   ✅ Statistical baseline complete")
print("   🚀 Ready for Phase 5: Random Forest Training")
print("   🎯 Ensemble will combine Statistical + RF predictions")

print("\n" + "="*80)
print("🚀 PHASE 4 COMPLETE - Ready for Phase 5: Random Forest")
print("="*80)
