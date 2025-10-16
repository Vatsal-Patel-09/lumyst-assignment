# Task 5: Utility Function Detection with Ensemble ML

## 🎯 Objective
Build an ML system to detect and filter utility/helper functions from core business logic in the FastAPI codebase using **Statistical Analysis + Random Forest Ensemble**.

## 📊 Dataset
- **Source**: FastAPI codebase analysis
- **Functions**: 291 total functions
- **Data File**: `analysis-with-code.json`
- **Types**: Methods (99), Classes (86), Functions (66), Files (35), Folders (4), Workspace (1)

## 🧠 Approach: Modular ML Pipeline

### 🎯 Strategy
**Ensemble Method**: Statistical Baseline + Random Forest
- **Statistical Component**: Transparent, rule-based scoring
- **Random Forest Component**: Learns complex patterns from labeled data
- **No Data Leakage**: Decorators/naming used for labeling only, NOT as features

---

## 📁 Project Structure

```
ml-task-5/
├── analysis-with-code.json      # Input: 291 FastAPI functions
├── requirements.txt             # Dependencies (pandas, numpy, radon, sklearn)
├── README.md                    # This file
├── .gitignore                   # Git ignore rules
│
├── notebooks/                   # 🎓 Jupyter notebooks for exploration
│   ├── README.md               # Pipeline documentation
│   └── 01_data_exploration.ipynb  # ✅ Phase 1: EDA
│
├── scripts/                     # 🐍 Python scripts (Jupyter alternative)
│   └── 01_data_exploration.py  # ✅ Phase 1: Data exploration
│
├── outputs/                     # 📊 Generated artifacts
│   ├── .gitkeep
│   ├── 01_explored_data.pkl    # ✅ Processed data from Phase 1
│   └── 01_explored_data.csv    # ✅ Human-readable export
│
├── models/                      # 🤖 Saved models (future)
│   └── .gitkeep
│
└── venv/                        # Virtual environment
```

---

## 🚀 7-Phase Modular Pipeline

### ✅ **Phase 1: Data Exploration** (COMPLETE)
**Objective**: Load and understand the 291 functions

**Key Outputs**:
- Dataset: 291 functions loaded
- Type distribution: Methods (34%), Classes (30%), Functions (23%)
- Code metrics: Average 79 lines, median 11 lines
- Pattern detection: 16 utility patterns, 8 core patterns, 29 route decorators

**Files**:
- `notebooks/01_data_exploration.ipynb`
- `scripts/01_data_exploration.py`
- `outputs/01_explored_data.pkl`

---

### 📋 **Phase 2: Feature Engineering** (PENDING)
**Objective**: Extract 15-20 pure statistical features

**Features to Extract**:
1. **Complexity Metrics**:
   - Cyclomatic complexity (decision points)
   - Halstead metrics (volume, difficulty, effort)
   
2. **Structure Metrics**:
   - AST depth/breadth
   - Nesting depth
   - Number of branches/loops
   
3. **Code Metrics**:
   - Lines of code (LOC)
   - Number of parameters
   - Number of return statements
   - Comment ratio

**Tools**: `radon`, `ast`, custom parsers

**Output**: `outputs/02_feature_matrix.pkl`

---

### 🏷️ **Phase 3: Label Creation** (PENDING)
**Objective**: Create high-confidence labels for training

**Labeling Strategy**:
- **High confidence utility**: Functions with decorators like `@lru_cache`, `@property`, `@staticmethod`
- **High confidence core**: Functions with route decorators (`@app.get`, `@app.post`)
- **Multiple signals**: Combine naming patterns + decorators + complexity
- **Quality over quantity**: Only label when confident (expect ~40-60 labeled functions)

**Output**: `outputs/03_labeled_data.pkl`

---

### 📊 **Phase 4: Statistical Baseline** (PENDING)
**Objective**: Build transparent rule-based scoring system

**Scoring Rules**:
```
Utility Score = weighted_sum([
    inverse(cyclomatic_complexity),
    inverse(LOC),
    inverse(num_parameters),
    inverse(halstead_difficulty),
    ...
])
```

**Validation**: Manual inspection of top/bottom 20 ranked functions

**Output**: `outputs/04_statistical_scores.pkl`

---

### 🌲 **Phase 5: Random Forest Training** (PENDING)
**Objective**: Train ML model on labeled subset

**Approach**:
- Train on high-confidence labeled functions only
- Use pure structural features (no decorators/names)
- Hyperparameter tuning with cross-validation
- Feature importance analysis

**Output**: `models/random_forest.pkl`

---

### 🎯 **Phase 6: Ensemble Model** (PENDING)
**Objective**: Combine Statistical + Random Forest predictions

**Ensemble Strategy**:
```
Final Score = (0.4 × Statistical Score) + (0.6 × RF Score)
```

**Validation**:
- Check top 50 utility functions manually
- Check bottom 50 core logic functions manually
- Precision/Recall on labeled subset

**Output**: `outputs/06_final_predictions.pkl`

---

### ✅ **Phase 7: Final Validation** (PENDING)
**Objective**: Quality assurance and documentation

**Tasks**:
- Generate classification report
- Document edge cases
- Final manual review
- Export final rankings

**Output**: `outputs/07_final_report.md`

---

## 🚀 Quick Start

### 1. Setup Environment
```powershell
# Create virtual environment
python -m venv venv

# Activate
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Phase 1 (Completed ✅)
```powershell
# Option 1: Run script
python scripts/01_data_exploration.py

# Option 2: Use Jupyter notebook
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 3. Check Outputs
```powershell
# View generated files
Get-ChildItem outputs/

# View CSV
Import-Csv outputs/01_explored_data.csv | Select -First 10
```

---

## 📊 Phase 1 Results

### Key Statistics:
- ✅ **Total Functions**: 291
- ✅ **Types Breakdown**:
  - Methods: 99 (34%)
  - Classes: 86 (30%)
  - Functions: 66 (23%)
  - Files: 35 (12%)
  - Other: 5 (2%)

### Code Metrics:
- **Average Lines**: 79.4
- **Median Lines**: 11.0
- **Max Lines**: 3,972 (FastAPI class)
- **Min Lines**: 0 (empty files)

### Pattern Detection:
- **Utility Patterns**: 16 functions (5.5%) - names like `get_`, `create_`, `read_`
- **Core Patterns**: 8 functions (2.7%) - names like `execute`, `process`, `handle`
- **Route Decorators**: 29 functions (10%) - average 543 lines vs 11 median

### Key Insights:
✅ Clear separation exists between utility and core logic  
✅ Route-decorated functions are significantly longer  
✅ Strong signals available for labeling  
✅ Data quality is good (None values handled)

---

## 📦 Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
radon>=6.0.0
scikit-learn>=1.3.0
jupyter>=1.0.0
```

---

## 🎯 Success Criteria

1. **Accuracy**: >85% on labeled subset
2. **Precision**: >90% (minimize false positives)
3. **Interpretability**: Top/bottom ranked functions make intuitive sense
4. **Transparency**: Explainable scoring system

---

## 📝 Next Action

**Ready for Phase 2**: Feature Engineering
- Extract 15-20 statistical features
- Use `radon`, `ast`, and custom parsers
- Save feature matrix for model training

---

**Last Updated**: October 17, 2025  
**Status**: Phase 1 Complete ✅ | Phase 2 Ready 🚀  
**Branch**: `task-5-detect-filter-utility`
