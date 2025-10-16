"""Quick verification of Phase 5 results and data split"""
import pickle
import pandas as pd

print("\n" + "="*80)
print("MODEL VERIFICATION")
print("="*80 + "\n")

# Load RF predictions on labeled data
print("📊 Phase 5: Random Forest Predictions on Labeled Data")
print("-" * 60)
rf_data = pickle.load(open('outputs/05_rf_predictions.pkl', 'rb'))
print(f"Total labeled samples: {len(rf_data)}")
print(f"\nTrue labels:")
print(f"  Core (0):    {(rf_data['ml_label'] == 0).sum()}")
print(f"  Utility (1): {(rf_data['ml_label'] == 1).sum()}")
print(f"\nPredictions:")
print(f"  Predicted Core (0):    {(rf_data['rf_prediction'] == 0).sum()}")
print(f"  Predicted Utility (1): {(rf_data['rf_prediction'] == 1).sum()}")
print(f"\n✅ Accuracy: {rf_data['rf_correct'].sum()}/{len(rf_data)} = {rf_data['rf_correct'].mean():.2%}")

# Show misclassified
print(f"\n❌ Misclassified:")
mis = rf_data[~rf_data['rf_correct']]
print(f"  Count: {len(mis)}")
if len(mis) > 0:
    for idx, row in mis.iterrows():
        true_label = "Utility" if row['ml_label'] == 1 else "Core"
        pred_label = "Utility" if row['rf_prediction'] == 1 else "Core"
        print(f"  - {row['function_name']:30s} | True: {true_label:7s} | Pred: {pred_label:7s} | Prob: {row['rf_probability']:.3f}")

# Load all data to check split
print(f"\n" + "="*80)
print("📂 Dataset Split (Phase 3 output)")
print("-" * 60)
all_data = pickle.load(open('outputs/03_labeled_data.pkl', 'rb'))
print(f"Total functions in dataset: {len(all_data)}")

labeled = all_data[all_data['ml_label'].notna()]
unlabeled = all_data[all_data['ml_label'].isna()]

print(f"\nLabeled (high-confidence): {len(labeled)}")
print(f"  - Used for training RF in Phase 5")
print(f"  - Core (0):    {(labeled['ml_label'] == 0).sum()}")
print(f"  - Utility (1): {(labeled['ml_label'] == 1).sum()}")

print(f"\nUnlabeled (ambiguous): {len(unlabeled)}")
print(f"  - Need prediction in Phase 6")
print(f"  - These will be scored by ensemble model")

# Show some examples
print(f"\n" + "="*80)
print("📝 Examples of Unlabeled Functions (first 10)")
print("-" * 60)
for idx, row in unlabeled.head(10).iterrows():
    print(f"  {row['function_name']:40s} | Type: {row['type']:8s} | LOC: {row['loc']:.0f}")

print(f"\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\n✅ Phase 5 Model Performance: {rf_data['rf_correct'].mean():.1%} accuracy on {len(rf_data)} labeled samples")
print(f"✅ Only {len(mis)} misclassification(s) - excellent!")
print(f"\n➡️  Phase 6 Task: Predict labels for {len(unlabeled)} unlabeled functions")
print(f"   Using ensemble: 40% Statistical + 60% Random Forest")
print("="*80 + "\n")
