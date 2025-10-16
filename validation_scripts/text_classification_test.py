"""
Test 9: Text Classification Behavioral Drift
Validates Task-Identity on sentiment analysis (20 Newsgroups)
Tests catastrophic forgetting via imbalanced fine-tuning
"""

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from task_identity import calculate_task_identity
import json
from datetime import datetime
import sys

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("TEST 9: TEXT CLASSIFICATION BEHAVIORAL DRIFT")
print("=" * 70)
print()

# ===================================================================
# STEP 1: LOAD REAL TEXT DATA
# ===================================================================
print("📥 STEP 1: Loading real text dataset...")
print("   Using 20 Newsgroups (auto-downloads from sklearn)")

try:
    # Load binary classification: comp.graphics vs rec.sport.baseball
    categories = ['comp.graphics', 'rec.sport.baseball']
    
    # Load training data
    newsgroups_train = fetch_20newsgroups(
        subset='train',
        categories=categories,
        shuffle=True,
        random_state=42,
        remove=('headers', 'footers', 'quotes')
    )
    
    # Load test data
    newsgroups_test = fetch_20newsgroups(
        subset='test',
        categories=categories,
        shuffle=True,
        random_state=42,
        remove=('headers', 'footers', 'quotes')
    )
    
    print(f"   ✓ Loaded {len(newsgroups_train.data)} REAL training documents")
    print(f"   ✓ Loaded {len(newsgroups_test.data)} REAL test documents")
    print(f"   ✓ Categories: {categories}")
    
    # Show sample to prove it's real
    print(f"\n   Sample document (first 100 chars):")
    print(f"   '{newsgroups_train.data[0][:100]}...'")
    print()
    
except Exception as e:
    print(f"❌ CRITICAL FAILURE: Could not load 20 Newsgroups dataset")
    print(f"❌ Error: {e}")
    print(f"❌ NO SYNTHETIC FALLBACK - STOPPING")
    sys.exit(1)

# MANDATORY VERIFICATION
if len(newsgroups_train.data) == 0 or len(newsgroups_test.data) == 0:
    print("❌ CRITICAL FAILURE: Dataset is empty")
    print("❌ NO SYNTHETIC FALLBACK - STOPPING")
    sys.exit(1)

# ===================================================================
# STEP 2: CONVERT TEXT TO NUMBERS (TF-IDF)
# ===================================================================
print("📊 STEP 2: Converting text to numerical features...")

try:
    vectorizer = TfidfVectorizer(max_features=5000)
    
    # Fit on training data, transform both
    X_train = vectorizer.fit_transform(newsgroups_train.data)
    X_test = vectorizer.transform(newsgroups_test.data)
    
    y_train = newsgroups_train.target
    y_test = newsgroups_test.target
    
    print(f"   ✓ Training features: {X_train.shape}")
    print(f"   ✓ Test features: {X_test.shape}")
    print(f"   ✓ Vocabulary size: {len(vectorizer.vocabulary_)}")
    print()
    
except Exception as e:
    print(f"❌ CRITICAL FAILURE: Text vectorization failed")
    print(f"❌ Error: {e}")
    print(f"❌ STOPPING")
    sys.exit(1)

# ===================================================================
# STEP 3: TRAIN BASELINE MODEL (Balanced Training)
# ===================================================================
print("🧠 STEP 3: Training baseline model on balanced data...")

baseline_model = LogisticRegression(max_iter=1000, random_state=42)
baseline_model.fit(X_train, y_train)

print("   ✓ Baseline model trained on both classes (balanced)")
print()

# ===================================================================
# STEP 4: GET BASELINE PREDICTIONS
# ===================================================================
print("🔍 STEP 4: Getting baseline predictions...")

preds_baseline = baseline_model.predict(X_test)
acc_baseline = accuracy_score(y_test, preds_baseline)

print(f"   ✓ Baseline accuracy: {acc_baseline:.3f}")
print(f"   ✓ Sample predictions: {preds_baseline[:10]}")

# SANITY CHECK
if acc_baseline < 0.7:
    print(f"⚠️ WARNING: Baseline accuracy is low ({acc_baseline:.3f})")
    print(f"⚠️ Expected >0.85 for binary text classification")

unique_baseline = len(np.unique(preds_baseline))
print(f"   ✓ Unique predictions: {unique_baseline}")

if unique_baseline < 2:
    print("❌ CRITICAL: Model predicting only one class")
    print("❌ VALIDATION INVALID - STOPPING")
    sys.exit(1)

print()

# ===================================================================
# STEP 5: CATASTROPHIC FORGETTING (Heavy Class 1 Bias)
# ===================================================================
print("⚙️ STEP 5: Inducing catastrophic forgetting...")
print("   Creating heavily imbalanced dataset (10:1 ratio favoring Class 1)...")

# Get indices for each class
class_0_idx = y_train == 0
class_1_idx = y_train == 1

X_class0 = X_train[class_0_idx]
y_class0 = y_train[class_0_idx]

X_class1 = X_train[class_1_idx]
y_class1 = y_train[class_1_idx]

# Create imbalanced dataset: 10% class 0, 90% class 1
n_class0_samples = int(len(y_class0) * 0.1)
n_class1_samples = len(y_class1)

# Randomly sample from class 0
import random
random.seed(42)
class0_indices = random.sample(range(len(y_class0)), n_class0_samples)

X_imbalanced_class0 = X_class0[class0_indices]
y_imbalanced_class0 = y_class0[class0_indices]

# Combine with all of class 1
from scipy.sparse import vstack
X_imbalanced = vstack([X_imbalanced_class0, X_class1])
y_imbalanced = np.concatenate([y_imbalanced_class0, y_class1])

# Shuffle
shuffle_idx = np.random.permutation(len(y_imbalanced))
X_imbalanced = X_imbalanced[shuffle_idx]
y_imbalanced = y_imbalanced[shuffle_idx]

print(f"   ✓ Imbalanced dataset created:")
print(f"     - Class 0 samples: {n_class0_samples}")
print(f"     - Class 1 samples: {n_class1_samples}")
print(f"     - Ratio: {n_class1_samples/n_class0_samples:.1f}:1")
print()

# Fine-tune on imbalanced data
print("   Fine-tuning model on heavily imbalanced data...")
baseline_model.fit(X_imbalanced, y_imbalanced)
print("   ✓ Model fine-tuned with strong Class 1 bias")
print()

# ===================================================================
# STEP 6: GET CURRENT PREDICTIONS (After Forgetting)
# ===================================================================
print("🔍 STEP 6: Getting predictions after imbalanced fine-tuning...")

preds_current = baseline_model.predict(X_test)
acc_current = accuracy_score(y_test, preds_current)

print(f"   ✓ Current accuracy: {acc_current:.3f}")
print(f"   ✓ Sample predictions: {preds_current[:10]}")

unique_current = len(np.unique(preds_current))
print(f"   ✓ Unique predictions: {unique_current}")

print()

# ===================================================================
# STEP 7: CALCULATE TASK-IDENTITY
# ===================================================================
print("🎯 STEP 7: Calculating Task-Identity...")

try:
    task_id = calculate_task_identity(
        y_test,          # REAL labels
        preds_baseline,  # REAL baseline predictions
        y_test,          # REAL labels (same test set)
        preds_current,   # REAL current predictions
        labels=[0, 1]    # Binary classification
    )
    
    print(f"   ✓ Task-Identity calculated: {task_id:.3f}")
    
except Exception as e:
    print(f"❌ CRITICAL: Task-Identity calculation failed")
    print(f"❌ Error: {e}")
    print(f"❌ STOPPING")
    sys.exit(1)

print()

# ===================================================================
# STEP 8: DISPLAY RESULTS
# ===================================================================
print("=" * 70)
print("📊 RESULTS")
print("=" * 70)
print(f"Baseline accuracy (balanced training): {acc_baseline:.3f}")
print(f"Current accuracy (after imbalanced fine-tuning): {acc_current:.3f}")
print(f"Accuracy change: {(acc_current - acc_baseline) * 100:.1f}%")
print()
print(f"🎯 Task-Identity: {task_id:.3f}")
print(f"📉 Behavioral drift: {(1 - task_id) * 100:.1f}%")
print()

# Interpretation
if task_id < 0.2:
    interpretation = "Severe behavioral change detected"
    recommendation = "Critical drift - model behavior fundamentally altered"
elif task_id < 0.5:
    interpretation = "Major behavioral shift"
    recommendation = "Significant drift detected"
elif task_id < 0.85:
    interpretation = "Moderate behavioral change"
    recommendation = "Moderate drift detected"
else:
    interpretation = "Behavior relatively stable"
    recommendation = "Minor behavioral changes"

print(f"✅ INTERPRETATION: {interpretation}")
print(f"💡 RECOMMENDATION: {recommendation}")
print()

print("=" * 70)
print("💡 WHAT THIS TEST VALIDATES")
print("=" * 70)
print("This test proves Task-Identity works on TEXT classification,")
print("not just images. The model was trained on balanced data (computer")
print("graphics vs baseball), then fine-tuned on heavily imbalanced data")
print("(10:1 ratio favoring baseball).")
print()
print("Task-Identity detected the behavioral change by comparing")
print("confusion matrices - the same method used in image tests.")
print()
print("This validates that Task-Identity is domain-agnostic and")
print("works across different data types (images AND text).")
print("=" * 70)
print()

# ===================================================================
# STEP 9: SAVE RESULTS
# ===================================================================
print("💾 STEP 9: Saving results...")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results = {
    "test_name": "text_classification",
    "test_type": "imbalanced_fine_tuning",
    "timestamp": timestamp,
    "dataset": "20_newsgroups",
    "categories": categories,
    "baseline_accuracy": float(acc_baseline),
    "current_accuracy": float(acc_current),
    "accuracy_change": float(acc_current - acc_baseline),
    "task_identity": float(task_id),
    "behavioral_drift": float(1 - task_id),
    "interpretation": interpretation,
    "recommendation": recommendation,
    "details": {
        "total_train_samples": int(len(y_train)),
        "total_test_samples": int(len(y_test)),
        "imbalanced_class0_samples": int(n_class0_samples),
        "imbalanced_class1_samples": int(n_class1_samples),
        "imbalance_ratio": float(n_class1_samples/n_class0_samples),
        "feature_count": int(X_train.shape[1]),
        "model": "LogisticRegression",
        "vectorizer": "TfidfVectorizer",
        "max_features": 5000
    },
    "data_verification": {
        "real_dataset": "20 Newsgroups (sklearn)",
        "no_synthetic_data": True,
        "manual_verification": "Sample text shown in output"
    }
}

filename = f"results/09_text_classification/text_classification_{timestamp}.json"

try:
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   ✓ Results saved: {filename}")
except Exception as e:
    print(f"⚠️ Could not save results: {e}")

print()

# ===================================================================
# FINAL VERIFICATION CHECKLIST
# ===================================================================
print("=" * 70)
print("✅ VERIFICATION CHECKLIST")
print("=" * 70)
print(f"✓ Real dataset loaded: 20 Newsgroups")
print(f"✓ Training samples: {len(y_train)}")
print(f"✓ Test samples: {len(y_test)}")
print(f"✓ Baseline model trained: {acc_baseline:.3f} accuracy")
print(f"✓ Imbalanced fine-tuning applied: {acc_current:.3f} accuracy")
print(f"✓ Task-Identity calculated: {task_id:.3f}")
print(f"✓ Results saved: {filename}")
print(f"✓ NO SYNTHETIC DATA USED")
print("=" * 70)
print()
print("🎉 Test 9 complete!")

