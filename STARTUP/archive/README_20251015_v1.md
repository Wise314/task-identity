# 🚀 STARTUP: Getting Started with Task-Identity

**Quick start guide for new users and Claude instances**

---

## 📋 What is Task-Identity?

Task-Identity is a **training-free metric** that detects behavioral drift in AI classification models by comparing confusion matrices across time periods using Pearson correlation.

**Key Result:**
- ✅ Task-Identity = 0.000 → Catastrophic behavioral failure detected
- ⚠️ Embedding similarity = 0.583 → Underestimated the severity

---

## 🎯 Core Concept

Neural networks can maintain moderate internal structural similarity even during **catastrophic behavioral failure**. Task-Identity detects this by measuring **what the model actually does** (its confusion patterns), not its internal structure.

---

## ⚡ Quick Install & Test

```bash
# 1. Clone and setup
git clone https://github.com/Wise314/task-identity.git
cd task-identity
python3 -m venv task-identity-env
source task-identity-env/bin/activate  # Windows: task-identity-env\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run validation tests
python3 catastrophic_forgetting_full_detection.py
python3 progressive_noise_validator.py
```

**Expected results:**
- Catastrophic forgetting test: Task-Identity = 0.000 (detects total failure)
- Progressive noise test: Task-Identity decreases smoothly with degradation

---

## 💻 Basic Usage

### Import and Use

```python
from task_identity import calculate_task_identity

# Compare model behavior across two time periods
task_id = calculate_task_identity(
    y_true_baseline,    # True labels from baseline
    y_pred_baseline,    # Model predictions from baseline
    y_true_current,     # True labels from current period
    y_pred_current,     # Model predictions from current period
    labels=range(10)    # All possible class labels
)

print(f"Task-Identity: {task_id:.3f}")

# Interpret the result
if task_id < 0.2:
    print("🚨 CRITICAL: Catastrophic behavioral change!")
elif task_id < 0.5:
    print("⚠️ WARNING: Major behavioral drift detected")
elif task_id < 0.8:
    print("⚠️ NOTICE: Moderate behavioral shift")
else:
    print("✅ Model behavior is stable")
```

---

## 📊 What the Tests Validate

### Test 1: Catastrophic Forgetting
**File:** `catastrophic_forgetting_full_detection.py`

**What it does:**
1. Trains neural network on MNIST digits 0-4
2. Fine-tunes on digits 5-9 (causes catastrophic forgetting)
3. Tests on original digits 0-4

**Expected results:**
- Accuracy: 99.3% → 0.0% (total collapse)
- Task-Identity: 0.000 (correctly detects catastrophic failure)
- Embedding Identity: 0.583 (underestimates severity)

**Why this matters:** Proves Task-Identity detects behavioral collapse that embedding similarity misses.

---

### Test 2: Progressive Degradation
**File:** `progressive_noise_validator.py`

**What it does:**
1. Trains model on clean MNIST
2. Tests with increasing Gaussian noise (0% → 30%)
3. Tracks Task-Identity and accuracy

**Expected results:**
| Noise | Task-Identity | Accuracy |
|-------|---------------|----------|
| 0%    | 1.000        | 93.6%    |
| 10%   | 0.999        | 92.2%    |
| 20%   | 0.948        | 79.3%    |
| 30%   | 0.780        | 61.4%    |

**Why this matters:** Shows Task-Identity tracks gradual degradation smoothly.

---

## 📁 Repository Structure

```
task-identity/
├── README.md                                  # Main documentation
├── task_identity/
│   └── __init__.py                           # Core calculate_task_identity() function
├── catastrophic_forgetting_full_detection.py # Test 1: Catastrophic forgetting
├── progressive_noise_validator.py            # Test 2: Progressive degradation
├── results/                                   # Test outputs (JSON files)
├── STARTUP/                                   # This guide
│   ├── README.md                             # You are here
│   └── archive/                              # Previous versions
└── readmearchive/                            # Archived main README versions
```

---

## 🔧 Code Architecture

### Core Function Location
**File:** `task_identity/__init__.py`

The core `calculate_task_identity()` function is centralized here. Both validation scripts import from this module.

### What Changed (October 15, 2024)
- ✅ Moved core algorithm to `task_identity/__init__.py`
- ✅ Removed code duplication from validation scripts
- ✅ Added input validation
- ✅ Made random seed configurable
- ✅ Removed unused functions

---

## 🧪 Running Custom Tests

### Example: Test Your Own Model

```python
from task_identity import calculate_task_identity
from sklearn.metrics import confusion_matrix
import numpy as np

# Your model's predictions
baseline_predictions = your_model.predict(baseline_data)
current_predictions = your_model.predict(current_data)

# Calculate Task-Identity
task_id = calculate_task_identity(
    y_true_baseline,
    baseline_predictions,
    y_true_current,
    current_predictions,
    labels=np.unique(y_true_baseline)  # All your class labels
)

print(f"Behavioral Similarity: {task_id:.3f}")
```

---

## 📖 Understanding the Results

### Task-Identity Scale

| Score | Meaning | Action |
|-------|---------|--------|
| 0.95-1.00 | Nearly identical behavior | ✅ Model is stable |
| 0.80-0.95 | Minor changes | ⚠️ Monitor closely |
| 0.50-0.80 | Moderate drift | ⚠️⚠️ Investigate cause |
| 0.20-0.50 | Major change | 🚨 Alert team |
| 0.00-0.20 | Catastrophic shift | 🚨🚨 Critical failure |

### Why It Works

Confusion matrices capture **what the model confuses with what** - its behavioral fingerprint:
- Same confusion patterns → High Task-Identity (stable behavior)
- Different confusion patterns → Low Task-Identity (behavioral drift)

---

## 🎓 Key Insight (Patent Claim)

**The Innovation:** Neural networks can maintain moderate-to-high internal structural similarity (embedding identity = 0.583) while experiencing **complete behavioral failure** (accuracy drops to 0.0%).

**Task-Identity solves this:** It measures actual behavior (Task-Identity = 0.000), not internal structure, correctly identifying catastrophic failures.

---

## ⚠️ Important Notes

### What's Core vs. Experimental

**Core Task-Identity (Patent-relevant):**
- The `calculate_task_identity()` function in `task_identity/__init__.py`
- Confusion matrix correlation method
- Validation results: 0.000 detects catastrophic forgetting

**Experimental Code (NOT core metric):**
- Multiplier calculations in validation scripts
- Autocorrelation computations
- Detection threshold tuning (v2.0 vs Config 2)

These experimental features are used in the validation scripts but are NOT part of the core Task-Identity metric.

---

## 🐛 Troubleshooting

### Installation Issues

**Problem:** `pip install` fails
**Solution:** Make sure virtual environment is activated and Python 3.8+ is installed

**Problem:** MNIST download fails
**Solution:** Scripts will auto-download MNIST on first run - requires internet connection

### Test Results Don't Match

**Problem:** Getting different Task-Identity values
**Solution:** Random seed is set to 42 by default for reproducibility. If you changed it, results will vary.

---

## 📚 Next Steps

1. **Run both validation tests** to verify everything works
2. **Read the main README.md** for detailed documentation
3. **Check results/ folder** for JSON outputs from your test runs
4. **Try on your own models** using the code examples above

---

## 💡 Quick Reference

### One-Line Summary
Task-Identity = Pearson correlation of flattened confusion matrices from two time periods

### Key Files
- **Core function:** `task_identity/__init__.py`
- **Test 1:** `catastrophic_forgetting_full_detection.py`
- **Test 2:** `progressive_noise_validator.py`
- **Results:** `results/*.json`

### Key Commands
```bash
# Activate environment
source task-identity-env/bin/activate

# Run tests
python3 catastrophic_forgetting_full_detection.py
python3 progressive_noise_validator.py

# Deactivate when done
deactivate
```

---

## 📞 Need Help?

- **Technical issues:** Open an issue on GitHub
- **Usage questions:** Check main README.md
- **Commercial inquiries:** Open an issue

---

**Last Updated:** October 15, 2024  
**Status:** Production-ready, patent pending

---

**Welcome to Task-Identity! 🚀**
