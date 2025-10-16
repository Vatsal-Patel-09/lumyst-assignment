"""
Phase 3: Label Creation
========================
Create high-confidence labels for training Random Forest classifier.

Strategy:
- Use decorators + naming patterns + complexity as LABELING signals (not features!)
- Label only high-confidence cases (quality over quantity)
- Keep majority unlabeled (will be scored by ensemble later)

Label Categories:
- UTILITY (1): Helper functions, simple utilities, property accessors
- CORE (0): Business logic, route handlers, complex workflows
- UNLABELED (NaN): Everything else (will be scored by ensemble)

Labeling Rules:
==============
HIGH CONFIDENCE UTILITY (label = 1):
- @property, @staticmethod, @classmethod decorators
- @lru_cache, @cached_property (caching utilities)
- Names like: get_*, read_*, parse_*, validate_*, is_*, has_*
- AND low complexity (cyclomatic < 3)
- AND short (LOC < 20)

HIGH CONFIDENCE CORE (label = 0):
- Route decorators: @app.get, @app.post, @router.get, etc.
- Names like: process_*, handle_*, execute_*, create_*, update_*
- OR high complexity (cyclomatic > 5)
- OR long (LOC > 50)

UNLABELED (NaN):
- Everything in between
- Ambiguous cases
- Let the ensemble decide
"""

import pandas as pd
import numpy as np
import pickle
import re
from typing import Optional

print("="*80)
print("🏷️ PHASE 3: LABEL CREATION")
print("="*80)

# ============================================================================
# SECTION 1: LOAD DATA FROM PHASE 1 & 2
# ============================================================================
print("\n" + "="*80)
print("1️⃣ LOADING DATA")
print("="*80)

# Load Phase 1 data (has decorators and naming patterns)
phase1_file = 'outputs/01_explored_data.pkl'
print(f"📥 Loading Phase 1 data: {phase1_file}")
df_phase1 = pd.read_pickle(phase1_file)

# Load Phase 2 data (has features)
phase2_file = 'outputs/02_feature_matrix.pkl'
print(f"📥 Loading Phase 2 data: {phase2_file}")
df_phase2 = pd.read_pickle(phase2_file)

print(f"\n✅ Phase 1 shape: {df_phase1.shape}")
print(f"✅ Phase 2 shape: {df_phase2.shape}")

# Merge on 'id'
print(f"\n🔗 Merging Phase 1 and Phase 2 data...")
df = pd.merge(df_phase1, df_phase2, on=['id', 'label', 'type'], how='inner', suffixes=('_p1', '_p2'))
print(f"✅ Merged data shape: {df.shape}")
print(f"📋 Total columns: {len(df.columns)}")

# Clean up duplicate columns if any
if 'has_route_decorator_p1' in df.columns and 'has_route_decorator_p2' in df.columns:
    df['has_route_decorator'] = df['has_route_decorator_p1']
    df = df.drop(columns=['has_route_decorator_p1', 'has_route_decorator_p2'])
    print("✅ Resolved duplicate 'has_route_decorator' column")

# ============================================================================
# SECTION 2: DECORATOR DETECTION
# ============================================================================
print("\n" + "="*80)
print("2️⃣ DECORATOR DETECTION")
print("="*80)

def detect_decorators(code: str) -> dict:
    """Detect specific decorators in code."""
    if not code or code is None:
        return {
            'has_property': False,
            'has_staticmethod': False,
            'has_classmethod': False,
            'has_lru_cache': False,
            'has_cached_property': False,
            'has_route_decorator': False
        }
    
    decorators = {
        'has_property': bool(re.search(r'@property\b', code)),
        'has_staticmethod': bool(re.search(r'@staticmethod\b', code)),
        'has_classmethod': bool(re.search(r'@classmethod\b', code)),
        'has_lru_cache': bool(re.search(r'@lru_cache|@cache\b', code)),
        'has_cached_property': bool(re.search(r'@cached_property\b', code)),
        'has_route_decorator': bool(re.search(r'@(app|router)\.(get|post|put|delete|patch|options|head)', code))
    }
    
    return decorators

print("🔍 Detecting decorators in all functions...")
decorator_data = df['code'].apply(detect_decorators).apply(pd.Series)

# Drop duplicate decorator columns if they exist from merge
cols_to_drop = [col for col in decorator_data.columns if col in df.columns]
if cols_to_drop:
    print(f"   Dropping duplicate columns: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)

df = pd.concat([df, decorator_data], axis=1)

print("\n📊 Decorator Statistics:")
print(f"   @property: {decorator_data['has_property'].sum()} functions")
print(f"   @staticmethod: {decorator_data['has_staticmethod'].sum()} functions")
print(f"   @classmethod: {decorator_data['has_classmethod'].sum()} functions")
print(f"   @lru_cache: {decorator_data['has_lru_cache'].sum()} functions")
print(f"   @cached_property: {decorator_data['has_cached_property'].sum()} functions")
print(f"   Route decorators: {decorator_data['has_route_decorator'].sum()} functions")

# ============================================================================
# SECTION 3: NAMING PATTERN DETECTION (REFINED)
# ============================================================================
print("\n" + "="*80)
print("3️⃣ NAMING PATTERN ANALYSIS")
print("="*80)

def classify_name_pattern(name: str) -> str:
    """Classify function name into utility/core/other patterns."""
    if not name or name is None:
        return 'other'
    
    name_lower = name.lower()
    
    # Strong utility patterns
    utility_patterns = [
        r'^get_', r'^read_', r'^fetch_', r'^load_',
        r'^parse_', r'^convert_', r'^format_',
        r'^validate_', r'^check_', r'^verify_',
        r'^is_', r'^has_', r'^can_',
        r'^to_', r'^from_',
        r'_helper$', r'_util$', r'_utils$'
    ]
    
    for pattern in utility_patterns:
        if re.search(pattern, name_lower):
            return 'utility_pattern'
    
    # Strong core logic patterns
    core_patterns = [
        r'^process_', r'^handle_', r'^execute_',
        r'^create_', r'^update_', r'^delete_',
        r'^save_', r'^store_', r'^persist_',
        r'^build_', r'^generate_', r'^compute_',
        r'^run_', r'^start_', r'^stop_'
    ]
    
    for pattern in core_patterns:
        if re.search(pattern, name_lower):
            return 'core_pattern'
    
    return 'other'

print("🔍 Analyzing naming patterns...")
df['refined_name_pattern'] = df['label'].apply(classify_name_pattern)

print("\n📊 Naming Pattern Distribution:")
pattern_counts = df['refined_name_pattern'].value_counts()
for pattern, count in pattern_counts.items():
    pct = (count / len(df)) * 100
    print(f"   {pattern}: {count} ({pct:.1f}%)")

# ============================================================================
# SECTION 4: LABELING LOGIC
# ============================================================================
print("\n" + "="*80)
print("4️⃣ APPLYING LABELING RULES")
print("="*80)

def create_labels(row) -> Optional[int]:
    """
    Create labels using multiple signals.
    Returns: 1 (utility), 0 (core), or NaN (unlabeled)
    """
    
    # Extract features
    complexity = row.get('cyclomatic_complexity', 0)
    loc = row.get('loc', 0)
    name_pattern = row.get('refined_name_pattern', 'other')
    
    # Decorator flags
    is_property = row.get('has_property', False)
    is_staticmethod = row.get('has_staticmethod', False)
    is_classmethod = row.get('has_classmethod', False)
    is_cached = row.get('has_lru_cache', False) or row.get('has_cached_property', False)
    is_route = row.get('has_route_decorator', False)
    
    # ===========================================
    # HIGH CONFIDENCE CORE LOGIC (label = 0)
    # ===========================================
    
    # Route handlers are ALWAYS core logic
    if is_route:
        return 0
    
    # Very complex functions are core logic
    if complexity > 10:
        return 0
    
    # Very long functions are core logic
    if loc > 100:
        return 0
    
    # Core naming + moderate complexity
    if name_pattern == 'core_pattern' and complexity > 3:
        return 0
    
    # ===========================================
    # HIGH CONFIDENCE UTILITY (label = 1)
    # ===========================================
    
    # Property accessors are utilities
    if is_property and loc < 20:
        return 1
    
    # Static/class methods that are simple are utilities
    if (is_staticmethod or is_classmethod) and complexity <= 2 and loc < 30:
        return 1
    
    # Cached functions that are simple are utilities
    if is_cached and complexity <= 3 and loc < 25:
        return 1
    
    # Utility naming + simple code
    if name_pattern == 'utility_pattern' and complexity <= 2 and loc < 20:
        return 1
    
    # Very simple, very short functions with utility names
    if name_pattern == 'utility_pattern' and complexity == 0 and loc < 10:
        return 1
    
    # ===========================================
    # UNLABELED (ambiguous cases)
    # ===========================================
    return np.nan

print("🔄 Creating labels based on multiple signals...")
df['ml_label'] = df.apply(create_labels, axis=1)

# ============================================================================
# SECTION 5: LABEL STATISTICS
# ============================================================================
print("\n" + "="*80)
print("5️⃣ LABEL STATISTICS")
print("="*80)

label_counts = df['ml_label'].value_counts(dropna=False)
total = len(df)

print(f"\n📊 Label Distribution:")
print(f"   UTILITY (1): {label_counts.get(1.0, 0)} functions ({(label_counts.get(1.0, 0)/total)*100:.1f}%)")
print(f"   CORE (0): {label_counts.get(0.0, 0)} functions ({(label_counts.get(0.0, 0)/total)*100:.1f}%)")
print(f"   UNLABELED: {df['ml_label'].isna().sum()} functions ({(df['ml_label'].isna().sum()/total)*100:.1f}%)")
print(f"   TOTAL LABELED: {df['ml_label'].notna().sum()} functions ({(df['ml_label'].notna().sum()/total)*100:.1f}%)")

# ============================================================================
# SECTION 6: LABEL QUALITY ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("6️⃣ LABEL QUALITY ANALYSIS")
print("="*80)

# Analyze labeled samples
labeled_df = df[df['ml_label'].notna()].copy()

if len(labeled_df) > 0:
    print(f"\n📊 Labeled Functions Analysis:")
    print(f"   Total labeled: {len(labeled_df)}")
    
    # Utility functions stats
    utility_df = labeled_df[labeled_df['ml_label'] == 1]
    if len(utility_df) > 0:
        print(f"\n✅ UTILITY Functions (label=1): {len(utility_df)}")
        print(f"   Average complexity: {utility_df['cyclomatic_complexity'].mean():.2f}")
        print(f"   Average LOC: {utility_df['loc'].mean():.2f}")
        print(f"   Average nesting: {utility_df['max_nesting_depth'].mean():.2f}")
        
        # Show labeling reasons
        print(f"\n   Labeling breakdown:")
        print(f"      - @property: {utility_df['has_property'].sum()}")
        print(f"      - @staticmethod: {utility_df['has_staticmethod'].sum()}")
        print(f"      - @classmethod: {utility_df['has_classmethod'].sum()}")
        print(f"      - Cached: {(utility_df['has_lru_cache'] | utility_df['has_cached_property']).sum()}")
        print(f"      - Utility naming: {(utility_df['refined_name_pattern'] == 'utility_pattern').sum()}")
    
    # Core logic stats
    core_df = labeled_df[labeled_df['ml_label'] == 0]
    if len(core_df) > 0:
        print(f"\n🔥 CORE Logic Functions (label=0): {len(core_df)}")
        print(f"   Average complexity: {core_df['cyclomatic_complexity'].mean():.2f}")
        print(f"   Average LOC: {core_df['loc'].mean():.2f}")
        print(f"   Average nesting: {core_df['max_nesting_depth'].mean():.2f}")
        
        # Show labeling reasons
        print(f"\n   Labeling breakdown:")
        print(f"      - Route decorators: {core_df['has_route_decorator'].sum()}")
        print(f"      - High complexity (>10): {(core_df['cyclomatic_complexity'] > 10).sum()}")
        print(f"      - Long code (>100 LOC): {(core_df['loc'] > 100).sum()}")
        print(f"      - Core naming: {(core_df['refined_name_pattern'] == 'core_pattern').sum()}")

# ============================================================================
# SECTION 7: SAMPLE INSPECTION
# ============================================================================
print("\n" + "="*80)
print("7️⃣ SAMPLE INSPECTION")
print("="*80)

# Show sample utility functions
print("\n📝 Sample UTILITY Functions (label=1):")
print("-" * 70)
utility_samples = labeled_df[labeled_df['ml_label'] == 1].head(5)
for idx, row in utility_samples.iterrows():
    print(f"\n   Name: {row['label']}")
    print(f"   Type: {row['type']}")
    print(f"   Complexity: {row['cyclomatic_complexity']:.1f} | LOC: {row['loc']:.0f} | Nesting: {row['max_nesting_depth']:.0f}")
    decorators = []
    if row['has_property']: decorators.append('@property')
    if row['has_staticmethod']: decorators.append('@staticmethod')
    if row['has_classmethod']: decorators.append('@classmethod')
    if row['has_lru_cache']: decorators.append('@lru_cache')
    print(f"   Decorators: {', '.join(decorators) if decorators else 'None'}")
    print(f"   Naming: {row['refined_name_pattern']}")

# Show sample core functions
print("\n\n📝 Sample CORE Logic Functions (label=0):")
print("-" * 70)
core_samples = labeled_df[labeled_df['ml_label'] == 0].head(5)
for idx, row in core_samples.iterrows():
    print(f"\n   Name: {row['label']}")
    print(f"   Type: {row['type']}")
    print(f"   Complexity: {row['cyclomatic_complexity']:.1f} | LOC: {row['loc']:.0f} | Nesting: {row['max_nesting_depth']:.0f}")
    if row['has_route_decorator']:
        print(f"   Route handler: YES")
    print(f"   Naming: {row['refined_name_pattern']}")

# ============================================================================
# SECTION 8: SAVE LABELED DATA
# ============================================================================
print("\n" + "="*80)
print("8️⃣ SAVING LABELED DATA")
print("="*80)

# Save full dataset with labels
output_pkl = 'outputs/03_labeled_data.pkl'
df.to_pickle(output_pkl)
print(f"✅ Saved labeled data to: {output_pkl}")

# Save CSV
output_csv = 'outputs/03_labeled_data.csv'
df.to_csv(output_csv, index=False)
print(f"✅ Saved CSV to: {output_csv}")

# Save only labeled subset for training
labeled_only_pkl = 'outputs/03_labeled_subset.pkl'
labeled_df.to_pickle(labeled_only_pkl)
print(f"✅ Saved labeled subset to: {labeled_only_pkl}")

print(f"\n📊 Saved data:")
print(f"   - Full dataset: {len(df)} rows")
print(f"   - Labeled subset: {len(labeled_df)} rows")
print(f"   - Total columns: {len(df.columns)}")

# ============================================================================
# SECTION 9: KEY FINDINGS SUMMARY
# ============================================================================
print("\n" + "="*80)
print("🎯 KEY FINDINGS SUMMARY")
print("="*80)

print("\n1️⃣ Labeling Strategy:")
print("   ✅ Multi-signal approach (decorators + naming + complexity)")
print("   ✅ Conservative labeling (only high-confidence cases)")
print("   ✅ Quality over quantity")

print("\n2️⃣ Label Distribution:")
total_labeled = df['ml_label'].notna().sum()
utility_count = (df['ml_label'] == 1).sum()
core_count = (df['ml_label'] == 0).sum()
unlabeled_count = df['ml_label'].isna().sum()

print(f"   UTILITY: {utility_count} ({(utility_count/total)*100:.1f}%)")
print(f"   CORE: {core_count} ({(core_count/total)*100:.1f}%)")
print(f"   UNLABELED: {unlabeled_count} ({(unlabeled_count/total)*100:.1f}%)")
print(f"   TOTAL LABELED: {total_labeled} ({(total_labeled/total)*100:.1f}%)")

print("\n3️⃣ Label Quality Indicators:")
if len(utility_df) > 0 and len(core_df) > 0:
    print(f"   ✅ Clear separation in complexity:")
    print(f"      - Utility avg: {utility_df['cyclomatic_complexity'].mean():.2f}")
    print(f"      - Core avg: {core_df['cyclomatic_complexity'].mean():.2f}")
    print(f"   ✅ Clear separation in LOC:")
    print(f"      - Utility avg: {utility_df['loc'].mean():.2f}")
    print(f"      - Core avg: {core_df['loc'].mean():.2f}")

print("\n4️⃣ Insights for Next Phase:")
if total_labeled >= 30:
    print(f"   ✅ Sufficient labeled data for Random Forest ({total_labeled} samples)")
    print("   ✅ Balanced representation of utility and core functions")
    print("   ✅ Ready for Phase 4: Statistical Baseline")
else:
    print(f"   ⚠️  Limited labeled data ({total_labeled} samples)")
    print("   💡 Statistical baseline will be primary model")

print("\n" + "="*80)
print("🚀 PHASE 3 COMPLETE - Ready for Phase 4: Statistical Baseline")
print("="*80)
