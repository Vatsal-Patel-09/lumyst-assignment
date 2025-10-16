# 📚 Notebook Workflow - Utility Function Detection

## 🎯 Approach: Statistical Analysis + Random Forest Ensemble

**Goal:** Identify and rank utility vs core logic functions in FastAPI codebase

---

## 📂 Notebook Structure

### ✅ Phase 1: Data Exploration
**Notebook:** `01_data_exploration.ipynb`  
**Status:** ✅ READY TO RUN  
**Objectives:**
- Load 291 functions from `analysis-with-code.json`
- Understand data structure and distributions
- Analyze naming patterns and code characteristics
- Identify signals for feature engineering

**Expected Outputs:**
- `outputs/01_explored_data.pkl` - Processed DataFrame
- Visualizations of data distributions
- Key insights for feature engineering

---

### 🔄 Phase 2: Feature Engineering
**Notebook:** `02_feature_engineering.ipynb`  
**Status:** ⏳ PENDING  
**Objectives:**
- Extract **15-20 pure statistical features**
- Calculate cyclomatic complexity (radon)
- Compute Halstead metrics
- Extract AST-based structural features
- No decorator/naming features (those are for labeling)

**Features to Extract:**
1. Cyclomatic Complexity
2. Lines of Code (LOC)
3. Halstead Volume/Difficulty/Effort
4. AST Depth/Breadth
5. Number of: parameters, returns, branches, loops
6. Comment ratio
7. Nesting depth
8. And more...

**Expected Outputs:**
- `outputs/02_feature_matrix.pkl` - Feature DataFrame
- `outputs/02_feature_descriptions.json` - Feature metadata

---

### 🏷️ Phase 3: Label Creation
**Notebook:** `03_label_creation.ipynb`  
**Status:** ⏳ PENDING  
**Objectives:**
- Create high-confidence training labels
- Use **multiple independent signals**:
  - Route decorators → Core Logic
  - Utility naming + low complexity → Utility
  - High complexity + long code → Core Logic
- Label only when signals agree (avoid ambiguous cases)

**Labeling Strategy:**
```python
# High confidence labels only
if has_route_decorator:
    label = "core_logic"
elif has_utility_name AND complexity < 3 AND loc < 10:
    label = "utility"
else:
    label = None  # Will be predicted
```

**Expected Outputs:**
- `outputs/03_labeled_data.pkl` - DataFrame with labels
- Label distribution analysis
- Confidence scores for labels

---

### 📊 Phase 4: Statistical Baseline Model
**Notebook:** `04_statistical_baseline.ipynb`  
**Status:** ⏳ PENDING  
**Objectives:**
- Build pure rule-based scoring system
- Define weighted scoring formula
- Transparent, interpretable baseline
- Score all 291 functions

**Baseline Formula:**
```python
score = 0.5  # baseline (uncertain)
if complexity >= 10: score += 0.25
if loc >= 50: score += 0.20
if has_many_branches: score += 0.15
# etc...
score = clip(score, 0, 1)
```

**Expected Outputs:**
- `outputs/04_statistical_scores.pkl` - Baseline scores
- Performance metrics (if we have labels)
- Interpretable scoring breakdown

---

### 🌲 Phase 5: Random Forest Training
**Notebook:** `05_random_forest_training.ipynb`  
**Status:** ⏳ PENDING  
**Objectives:**
- Train Random Forest on labeled subset
- Cross-validation & hyperparameter tuning
- Feature importance analysis
- Predict probability scores for all functions

**Model Configuration:**
```python
RandomForestClassifier(
    n_estimators=100-500,
    max_depth=10-30,
    min_samples_split=5-20,
    class_weight='balanced'
)
```

**Expected Outputs:**
- `outputs/05_random_forest_model.pkl` - Trained model
- `outputs/05_rf_scores.pkl` - RF probability scores
- `outputs/05_feature_importance.json` - Feature rankings
- Cross-validation results

---

### 🔀 Phase 6: Ensemble Model
**Notebook:** `06_ensemble_model.ipynb`  
**Status:** ⏳ PENDING  
**Objectives:**
- Combine Statistical + Random Forest scores
- Optimize ensemble weights
- Final importance score (0-1) for all functions
- Compare ensemble vs individual models

**Ensemble Strategy:**
```python
# Weighted average
final_score = (w1 * statistical_score) + (w2 * rf_score)

# Or voting
# Or stacking
```

**Expected Outputs:**
- `outputs/06_ensemble_scores.pkl` - Final scores
- `outputs/06_ranked_functions.json` - Ranked output
- Performance comparison table

---

### ✅ Phase 7: Final Validation
**Notebook:** `07_final_validation.ipynb`  
**Status:** ⏳ PENDING  
**Objectives:**
- Manual inspection of top/bottom ranked functions
- Edge case analysis
- False positive/negative investigation
- Final quality assurance

**Validation Checks:**
1. Top 20 highest scores - Should be core logic
2. Bottom 20 lowest scores - Should be utility
3. Mid-range scores - Mixed/uncertain
4. Short but important functions - No false positives

**Expected Outputs:**
- Validation report
- Final adjustments (if needed)
- Production-ready scorer

---

## 🔧 Production Module

After notebooks are validated, create production code:

**File:** `src/ensemble_scorer.py`

```python
class EnsembleUtilityScorer:
    def __init__(self):
        self.statistical_scorer = StatisticalScorer()
        self.rf_model = load_model('outputs/05_random_forest_model.pkl')
    
    def score_function(self, code, name):
        # Extract features
        # Score with both models
        # Ensemble
        return score, breakdown
```

---

## 📈 Success Metrics

### Model Performance:
- ✅ Accuracy > 85% on labeled subset
- ✅ Precision > 90% (minimize false positives)
- ✅ Recall > 80% (catch most utilities)
- ✅ F1-Score > 0.85

### Qualitative:
- ✅ Top 20 functions make sense as "core logic"
- ✅ Bottom 20 functions make sense as "utility"
- ✅ No obvious false positives on important functions
- ✅ Scores are interpretable and explainable

---

## 🚀 Next Steps

1. **Run Notebook 1** - Data Exploration
2. Review results and insights
3. Proceed to Notebook 2 - Feature Engineering
4. Continue step-by-step with validation at each phase

---

## 📝 Notes

- **No data leakage:** Decorator/naming used ONLY for labeling, NOT as features
- **Professional workflow:** Each notebook is self-contained and documented
- **Reproducible:** All outputs saved for inspection
- **Transparent:** Every decision explained and justified

---

**Status Last Updated:** October 17, 2025  
**Current Phase:** Phase 1 - Data Exploration ✅ READY
