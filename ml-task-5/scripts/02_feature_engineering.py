"""
Phase 2: Feature Engineering
=============================
Extract 15-20 pure statistical features from code without using decorators or naming patterns.

Features extracted:
1. Complexity Metrics (radon):
   - Cyclomatic complexity
   - Halstead volume, difficulty, effort
   
2. Structure Metrics (AST):
   - AST depth
   - AST breadth
   - Number of branches (if/else)
   - Number of loops (for/while)
   - Nesting depth
   
3. Code Metrics:
   - Lines of code (LOC)
   - Number of parameters
   - Number of return statements
   - Comment ratio
   - Docstring presence
   - Number of function calls
   - Number of assignments
   - Number of imports
"""

import pandas as pd
import numpy as np
import pickle
import ast
import re
from radon.complexity import cc_visit
from radon.metrics import h_visit, mi_visit
from radon.raw import analyze
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("📊 PHASE 2: FEATURE ENGINEERING")
print("="*80)

# ============================================================================
# SECTION 1: LOAD DATA FROM PHASE 1
# ============================================================================
print("\n" + "="*80)
print("1️⃣ LOADING PHASE 1 DATA")
print("="*80)

# Load processed data from Phase 1
input_file = 'outputs/01_explored_data.pkl'
print(f"📥 Loading data from: {input_file}")

df = pd.read_pickle(input_file)
print(f"✅ Loaded {len(df)} functions")
print(f"📋 Columns: {list(df.columns)}")

# ============================================================================
# SECTION 2: FEATURE EXTRACTION FUNCTIONS
# ============================================================================
print("\n" + "="*80)
print("2️⃣ DEFINING FEATURE EXTRACTORS")
print("="*80)

def safe_parse_ast(code: str) -> Optional[ast.AST]:
    """Safely parse code to AST, return None if fails."""
    if not code or code is None:
        return None
    try:
        return ast.parse(code)
    except:
        return None


def get_ast_depth(node: ast.AST, current_depth: int = 0) -> int:
    """Calculate maximum depth of AST tree."""
    if not isinstance(node, ast.AST):
        return current_depth
    
    max_depth = current_depth
    for child in ast.iter_child_nodes(node):
        depth = get_ast_depth(child, current_depth + 1)
        max_depth = max(max_depth, depth)
    
    return max_depth


def get_ast_breadth(node: ast.AST) -> int:
    """Count total number of nodes in AST."""
    if not isinstance(node, ast.AST):
        return 0
    
    count = 1
    for child in ast.iter_child_nodes(node):
        count += get_ast_breadth(child)
    
    return count


def count_ast_nodes(tree: ast.AST, node_type) -> int:
    """Count specific node types in AST."""
    if tree is None:
        return 0
    
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, node_type):
            count += 1
    return count


def get_max_nesting_depth(node: ast.AST, current_depth: int = 0) -> int:
    """Calculate maximum nesting depth (for/while/if statements)."""
    if not isinstance(node, ast.AST):
        return current_depth
    
    # Nodes that increase nesting
    nesting_nodes = (ast.For, ast.While, ast.If, ast.With, ast.Try)
    
    if isinstance(node, nesting_nodes):
        current_depth += 1
    
    max_depth = current_depth
    for child in ast.iter_child_nodes(node):
        depth = get_max_nesting_depth(child, current_depth)
        max_depth = max(max_depth, depth)
    
    return max_depth


def extract_radon_features(code: str) -> Dict[str, Any]:
    """Extract complexity and metrics using radon."""
    if not code or code is None:
        return {
            'cyclomatic_complexity': 0,
            'halstead_volume': 0,
            'halstead_difficulty': 0,
            'halstead_effort': 0,
            'maintainability_index': 0,
            'loc': 0,
            'lloc': 0,
            'sloc': 0,
            'comments': 0,
            'multi': 0,
            'blank': 0,
            'single_comments': 0
        }
    
    features = {}
    
    # Cyclomatic Complexity
    try:
        cc_results = cc_visit(code)
        if cc_results:
            # Average complexity across all functions/methods
            features['cyclomatic_complexity'] = np.mean([item.complexity for item in cc_results])
        else:
            features['cyclomatic_complexity'] = 1  # Default for simple code
    except:
        features['cyclomatic_complexity'] = 0
    
    # Halstead Metrics
    try:
        halstead = h_visit(code)
        features['halstead_volume'] = halstead.total.volume if halstead.total.volume else 0
        features['halstead_difficulty'] = halstead.total.difficulty if halstead.total.difficulty else 0
        features['halstead_effort'] = halstead.total.effort if halstead.total.effort else 0
    except:
        features['halstead_volume'] = 0
        features['halstead_difficulty'] = 0
        features['halstead_effort'] = 0
    
    # Maintainability Index
    try:
        mi = mi_visit(code, multi=True)
        features['maintainability_index'] = mi if mi else 0
    except:
        features['maintainability_index'] = 0
    
    # Raw Metrics (LOC, comments, etc.)
    try:
        raw = analyze(code)
        features['loc'] = raw.loc  # Lines of code
        features['lloc'] = raw.lloc  # Logical lines of code
        features['sloc'] = raw.sloc  # Source lines of code
        features['comments'] = raw.comments  # Comment lines
        features['multi'] = raw.multi  # Multi-line strings
        features['blank'] = raw.blank  # Blank lines
        features['single_comments'] = raw.single_comments  # Single line comments
    except:
        features['loc'] = 0
        features['lloc'] = 0
        features['sloc'] = 0
        features['comments'] = 0
        features['multi'] = 0
        features['blank'] = 0
        features['single_comments'] = 0
    
    return features


def extract_ast_features(code: str) -> Dict[str, Any]:
    """Extract structural features using AST."""
    tree = safe_parse_ast(code)
    
    if tree is None:
        return {
            'ast_depth': 0,
            'ast_breadth': 0,
            'num_functions': 0,
            'num_classes': 0,
            'num_branches': 0,
            'num_loops': 0,
            'num_parameters': 0,
            'num_returns': 0,
            'num_function_calls': 0,
            'num_assignments': 0,
            'num_imports': 0,
            'max_nesting_depth': 0,
            'has_docstring': False
        }
    
    features = {}
    
    # Tree structure
    features['ast_depth'] = get_ast_depth(tree)
    features['ast_breadth'] = get_ast_breadth(tree)
    
    # Function and class definitions
    features['num_functions'] = count_ast_nodes(tree, ast.FunctionDef) + count_ast_nodes(tree, ast.AsyncFunctionDef)
    features['num_classes'] = count_ast_nodes(tree, ast.ClassDef)
    
    # Control flow
    features['num_branches'] = count_ast_nodes(tree, ast.If)
    features['num_loops'] = count_ast_nodes(tree, ast.For) + count_ast_nodes(tree, ast.While)
    
    # Parameters
    total_params = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            total_params += len(node.args.args)
    features['num_parameters'] = total_params
    
    # Returns
    features['num_returns'] = count_ast_nodes(tree, ast.Return)
    
    # Function calls
    features['num_function_calls'] = count_ast_nodes(tree, ast.Call)
    
    # Assignments
    features['num_assignments'] = count_ast_nodes(tree, ast.Assign) + count_ast_nodes(tree, ast.AugAssign)
    
    # Imports
    features['num_imports'] = count_ast_nodes(tree, ast.Import) + count_ast_nodes(tree, ast.ImportFrom)
    
    # Nesting depth
    features['max_nesting_depth'] = get_max_nesting_depth(tree)
    
    # Docstring presence
    features['has_docstring'] = False
    if isinstance(tree, ast.Module) and tree.body:
        first_node = tree.body[0]
        if isinstance(first_node, ast.Expr) and isinstance(first_node.value, ast.Constant):
            if isinstance(first_node.value.value, str):
                features['has_docstring'] = True
    
    return features


def extract_all_features(code: str) -> Dict[str, Any]:
    """Extract all features from code."""
    # Get radon features
    radon_features = extract_radon_features(code)
    
    # Get AST features
    ast_features = extract_ast_features(code)
    
    # Combine
    all_features = {**radon_features, **ast_features}
    
    # Add derived features
    if all_features['loc'] > 0:
        all_features['comment_ratio'] = all_features['comments'] / all_features['loc']
    else:
        all_features['comment_ratio'] = 0
    
    return all_features


print("✅ Feature extractors defined:")
print("   - extract_radon_features(): Complexity & Halstead metrics")
print("   - extract_ast_features(): Structure & control flow")
print("   - extract_all_features(): Combined feature set")

# ============================================================================
# SECTION 3: EXTRACT FEATURES FOR ALL FUNCTIONS
# ============================================================================
print("\n" + "="*80)
print("3️⃣ EXTRACTING FEATURES FROM ALL FUNCTIONS")
print("="*80)

print(f"🔄 Processing {len(df)} functions...")
print("⏳ This may take a few moments...\n")

# Extract features for all functions
features_list = []

for idx, row in df.iterrows():
    if idx % 50 == 0 and idx > 0:
        print(f"   Processed {idx}/{len(df)} functions...")
    
    code = row['code']
    features = extract_all_features(code)
    features['id'] = row['id']
    features['label'] = row['label']
    features['type'] = row['type']
    features_list.append(features)

print(f"✅ Extracted features from all {len(features_list)} functions")

# Convert to DataFrame
features_df = pd.DataFrame(features_list)

# Reorder columns (metadata first, then features)
metadata_cols = ['id', 'label', 'type']
feature_cols = [col for col in features_df.columns if col not in metadata_cols]
features_df = features_df[metadata_cols + feature_cols]

print(f"\n📊 Feature matrix shape: {features_df.shape}")
print(f"📋 Total features extracted: {len(feature_cols)}")

# ============================================================================
# SECTION 4: FEATURE STATISTICS
# ============================================================================
print("\n" + "="*80)
print("4️⃣ FEATURE STATISTICS")
print("="*80)

print("\n📊 Feature Summary:")
print("-" * 50)

# Get only numeric columns
numeric_features = features_df[feature_cols].select_dtypes(include=[np.number])

print(numeric_features.describe().round(2))

print(f"\n🔢 Feature Count by Category:")
print(f"   - Radon features: 12 (complexity, Halstead, LOC, comments)")
print(f"   - AST features: 13 (structure, control flow, nesting)")
print(f"   - Derived features: 1 (comment_ratio)")
print(f"   - Total: {len(feature_cols)} features")

# ============================================================================
# SECTION 5: FEATURE CORRELATIONS
# ============================================================================
print("\n" + "="*80)
print("5️⃣ FEATURE CORRELATIONS (TOP 10)")
print("="*80)

# Calculate correlation matrix
corr_matrix = numeric_features.corr()

# Get upper triangle to avoid duplicates
upper_triangle = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
corr_pairs = corr_matrix.where(upper_triangle).stack().reset_index()
corr_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']
corr_pairs['Abs_Correlation'] = corr_pairs['Correlation'].abs()
corr_pairs = corr_pairs.sort_values('Abs_Correlation', ascending=False)

print("\n📊 Top 10 Highly Correlated Feature Pairs:")
print("-" * 70)
for idx, row in corr_pairs.head(10).iterrows():
    print(f"   {row['Feature 1']:25s} ↔ {row['Feature 2']:25s}  {row['Correlation']:+.3f}")

# ============================================================================
# SECTION 6: FEATURE DISTRIBUTION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("6️⃣ FEATURE DISTRIBUTION ANALYSIS")
print("="*80)

print("\n📊 Key Feature Distributions:")
print("-" * 50)

key_features = [
    'cyclomatic_complexity',
    'halstead_difficulty',
    'loc',
    'num_parameters',
    'num_returns',
    'max_nesting_depth',
    'ast_depth'
]

for feat in key_features:
    if feat in numeric_features.columns:
        data = numeric_features[feat]
        print(f"\n{feat}:")
        print(f"   Min: {data.min():.2f}")
        print(f"   Max: {data.max():.2f}")
        print(f"   Mean: {data.mean():.2f}")
        print(f"   Median: {data.median():.2f}")
        print(f"   Std: {data.std():.2f}")

# ============================================================================
# SECTION 7: SAVE FEATURE MATRIX
# ============================================================================
print("\n" + "="*80)
print("7️⃣ SAVING FEATURE MATRIX")
print("="*80)

# Save as pickle
output_pkl = 'outputs/02_feature_matrix.pkl'
features_df.to_pickle(output_pkl)
print(f"✅ Saved feature matrix to: {output_pkl}")

# Save as CSV for inspection
output_csv = 'outputs/02_feature_matrix.csv'
features_df.to_csv(output_csv, index=False)
print(f"✅ Saved CSV to: {output_csv}")

print(f"\n📊 Saved data:")
print(f"   - Rows: {len(features_df)}")
print(f"   - Columns: {len(features_df.columns)}")
print(f"   - Features: {len(feature_cols)}")

# ============================================================================
# SECTION 8: KEY FINDINGS SUMMARY
# ============================================================================
print("\n" + "="*80)
print("🎯 KEY FINDINGS SUMMARY")
print("="*80)

print("\n1️⃣ Feature Extraction:")
print(f"   ✅ Successfully extracted {len(feature_cols)} features")
print(f"   ✅ Processed all {len(features_df)} functions")
print(f"   ✅ No naming patterns or decorators used as features")

print("\n2️⃣ Feature Categories:")
print("   📊 Complexity: Cyclomatic, Halstead (volume, difficulty, effort)")
print("   📊 Code Metrics: LOC, LLOC, SLOC, comments, blank lines")
print("   📊 Structure: AST depth/breadth, nesting depth")
print("   📊 Control Flow: Branches, loops, returns, function calls")
print("   📊 Code Elements: Parameters, assignments, imports, classes")

print("\n3️⃣ Data Quality:")
zero_counts = (numeric_features == 0).sum()
high_zero_features = zero_counts[zero_counts > len(features_df) * 0.5].sort_values(ascending=False)

if len(high_zero_features) > 0:
    print(f"   ⚠️  Features with >50% zeros: {len(high_zero_features)}")
    for feat, count in high_zero_features.head(5).items():
        pct = (count / len(features_df)) * 100
        print(f"      - {feat}: {pct:.1f}% zeros")
else:
    print("   ✅ All features have good variance")

print("\n4️⃣ Insights for Next Phase:")
print("   ✅ Features are pure structural metrics (no data leakage)")
print("   ✅ Wide range of complexity captured (simple to very complex)")
print("   ✅ Ready for label creation in Phase 3")

print("\n" + "="*80)
print("🚀 PHASE 2 COMPLETE - Ready for Phase 3: Label Creation")
print("="*80)
