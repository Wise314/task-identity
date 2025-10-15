# 🚀 STARTUP: Getting Started with Task-Identity

**Quick start guide for new users and AI assistants**

---

## 📋 What is Task-Identity?

Task-Identity is a **training-free metric** that detects behavioral drift in AI classification models by comparing confusion matrices across time periods using Pearson correlation.

**The Problem It Solves:**
- Traditional metrics (embedding similarity, accuracy) can miss catastrophic failures
- Task-Identity measures **actual decision-making behavior**, not internal structure

**Key Validation Result:**
- ✅ **Task-Identity:** 0.000 → Correctly detected catastrophic failure
- ⚠️ **Embedding Similarity:** 0.583 → Significantly underestimated severity
- 📉 **Actual Performance:** 99.3% → 0.0% (total collapse)

---

⚡ Quick Install & Test (5 Minutes)
bash# 1. Clone and setup
git clone https://github.com/Wise314/task-identity.git
cd task-identity
python3 -m venv task-identity-env
source task-identity-env/bin/activate  # Windows: task-identity-env\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run a quick validation test
python3 catastrophic_forgetting_full_detection.py
```

**Expected output:**
```
✅ Task-Identity: 0.000 (detected catastrophic forgetting)
⚠️ Embedding Identity: 0.583 (underestimated severity)
📉 Accuracy: 99.3% → 0.0%

💻 Basic Usage (3 Lines of Code)
pythonfrom task_identity import calculate_task_identity

# Compare model behavior across two time periods
task_id = calculate_task_identity(
    y_true_baseline,    # True labels from baseline period
    y_pred_baseline,    # Model predictions from baseline period
    y_true_current,     # True labels from current period
    y_pred_current,     # Model predictions from current period
    labels=range(10)    # Complete set of class labels
)

# Interpret result
if task_id < 0.85:
    print(f"🚨 Behavioral drift detected! Task-Identity: {task_id:.3f}")
else:
    print(f"✅ Model stable. Task-Identity: {task_id:.3f}")
```

---

## 📊 8 Comprehensive Validation Tests

Task-Identity has been validated across diverse ML scenarios:

### Security & Safety
1. **[Catastrophic Forgetting](../results/01_catastrophic_forgetting/)** - Task-Identity: 0.000
2. **[Targeted Poisoning](../results/04_targeted_poisoning/)** - Task-Identity: 0.873 (per-class: 0.17)
3. **[Model Compression](../results/08_model_compression/)** - Task-Identity: 0.384

### Data Quality & Distribution
4. **[Progressive Noise](../results/02_progressive_noise/)** - Task-Identity: 0.780-1.000
5. **[Domain Shift](../results/03_domain_shift/)** - Task-Identity: 0.049
6. **[Class Imbalance](../results/06_class_imbalance/)** - Task-Identity: 0.576

### Training & Optimization
7. **[Cross-Domain Training](../results/05_cross_domain/)** - Task-Identity: 0.000
8. **[Training Dynamics](../results/07_training_dynamics/)** - Task-Identity: 0.999-1.000

**See [results/README.md](../results/README.md) for complete details on all tests.**

---

## 🎯 Key Validation Highlights

### Test 1: Catastrophic Forgetting Detection
**Why it matters:** Proves Task-Identity catches failures that structural metrics miss.

- **Scenario:** Model trained on digits 0-4, then fine-tuned on 5-9
- **Result:** Complete forgetting of original task
- **Task-Identity:** 0.000 ✅ (detected failure)
- **Embedding Similarity:** 0.583 ⚠️ (underestimated)
- **Accuracy:** 99.3% → 0.0%

### Test 6: Class Imbalance Impact
**Why it matters:** Detects hidden behavioral changes that accuracy misses.

- **Scenario:** Test on 90/10 imbalanced distribution
- **Accuracy:** 93.6% → 93.7% (appeared stable!)
- **Task-Identity:** 0.576 (detected 42% behavioral shift)
- **Insight:** Model making different mistakes despite same accuracy

### Test 8: Model Compression Validation
**Why it matters:** Prevents deploying broken compressed models.

- **Scenario:** 4x model compression (32-bit → 8-bit)
- **Size reduction:** 437KB → 109KB
- **Accuracy:** 93.6% → 39.5%
- **Task-Identity:** 0.384 (61.6% behavioral drift)
- **Outcome:** Blocked deployment - 6/10 classes destroyed

---

## 📁 Repository Structure
```
task-identity/
├── README.md                    # Main documentation
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
│
├── task_identity/               # Core package
│   └── __init__.py             # Core calculate_task_identity()
│
├── results/                     # Comprehensive validation results
│   ├── README.md               # Results overview & summary
│   ├── 01_catastrophic_forgetting/
│   ├── 02_progressive_noise/
│   ├── 03_domain_shift/
│   ├── 04_targeted_poisoning/
│   ├── 05_cross_domain/
│   ├── 06_class_imbalance/
│   ├── 07_training_dynamics/
│   ├── 08_model_compression/
│   └── archive/
│
├── validation_scripts/          # Test implementation scripts
│
├── STARTUP/                     # This guide
│   ├── README.md               # You are here
│   └── archive/
│
└── readmearchive/              # Historical README versions

🧪 Running Your Own Tests
Example: Monitor Production Model
pythonfrom task_identity import calculate_task_identity
import numpy as np

# Baseline: Last week's predictions
baseline_preds = production_model.predict(last_week_data)
baseline_labels = last_week_labels

# Current: This week's predictions
current_preds = production_model.predict(this_week_data)
current_labels = this_week_labels

# Calculate Task-Identity
task_id = calculate_task_identity(
    baseline_labels, baseline_preds,
    current_labels, current_preds,
    labels=np.unique(baseline_labels)
)

# Alert if drift detected
if task_id < 0.85:
    send_alert(f"Model drift detected! Task-Identity: {task_id:.3f}")
Example: Validate Model Compression
pythonfrom task_identity import calculate_task_identity

# Test set predictions
preds_full = full_precision_model.predict(test_data)
preds_compressed = compressed_model.predict(test_data)

# Compare behavior
task_id = calculate_task_identity(
    test_labels, preds_full,
    test_labels, preds_compressed,
    labels=range(num_classes)
)

# Deployment decision
if task_id > 0.95:
    print("✅ Compression preserved behavior - safe to deploy")
else:
    print(f"❌ Compression degraded behavior ({task_id:.3f}) - reject")

📖 Interpretation Guide
Task-Identity Thresholds
Score RangeInterpretationRecommended Action0.95 - 1.00Nearly identical behavior✅ Model stable, no action needed0.85 - 0.95Minor behavioral changes⚠️ Monitor, investigate if persistent0.70 - 0.85Moderate behavioral shift⚠️⚠️ Investigate cause, intervention likely needed0.50 - 0.70Major behavioral change🚨 Alert required, data/model issue0.00 - 0.50Catastrophic shift🚨🚨 Critical failure, immediate action
Context-Specific Thresholds
Different use cases need different thresholds:
Use CaseAlert ThresholdRationaleProduction monitoring< 0.95Catch drift earlySecurity validation< 0.85Allow some variationCompression QA< 0.95Strict preservationTransfer learning< 0.70Some forgetting OKTraining convergence≈ 1.00Stop when stable

🔬 How It Works
The Core Algorithm

Generate confusion matrix for baseline period
Generate confusion matrix for current period
Flatten both matrices to vectors
Calculate Pearson correlation coefficient
Return correlation as Task-Identity score

Why Confusion Matrices?
Confusion matrices capture complete behavioral fingerprints:

Which classes get confused with which
Pattern of mistakes the model makes
Decision boundary characteristics

Key Insight: Models with same confusion patterns behave identically, even if internal structures differ.

🎓 Core Innovation (Patent-Relevant)
The Problem
Neural networks maintain moderate internal structural similarity (e.g., embedding cosine similarity = 0.583) even during complete behavioral collapse (0% accuracy).
The Solution
Task-Identity measures actual decision-making behavior via confusion matrix correlation, correctly identifying catastrophic failures (Task-Identity = 0.000) that structural metrics miss.
Validation

✅ 8 comprehensive tests across security, data quality, and training
✅ Detects failures traditional metrics miss (e.g., class imbalance test)
✅ Works universally across classification domains
✅ No training data or model internals required


💡 Commercial Applications
1. Pre-Deployment Quality Control
Use Case: Validate model compression for edge devices
pythonif task_identity < 0.95:
    reject_deployment("Compression degraded behavior")
Value: Prevents shipping broken models to production
2. Production Monitoring
Use Case: Weekly behavioral drift detection
pythonif task_identity < 0.85:
    alert_team("Data drift detected - investigate pipeline")
Value: Catch issues before they impact users
3. Security Scanning
Use Case: Detect poisoning attacks
python# Per-class analysis
if class_5_task_id < 0.20:
    quarantine_model("Class 5 compromised - possible poisoning")
Value: Identify specific compromised classes
4. Training Optimization
Use Case: Intelligent early stopping
pythonif task_identity > 0.99:
    stop_training("Behavior converged - save compute")
Value: Reduce training costs

⚠️ What's Core vs. Experimental
Core Task-Identity (Production-Ready)
✅ calculate_task_identity() function in task_identity/__init__.py
✅ Confusion matrix correlation method
✅ All 8 validation test results
Experimental Features (Not Part of Core Metric)
⚠️ Multiplier calculations in some validation scripts
⚠️ Autocorrelation analysis
⚠️ Threshold auto-tuning experiments
Note: Only the core calculate_task_identity() function is production-ready. Experimental code in validation scripts is for research exploration.

🐛 Troubleshooting
Common Issues
Q: Installation fails
A: Ensure Python 3.8+ and virtual environment is activated
Q: MNIST download fails
A: First run requires internet - MNIST auto-downloads
Q: Different results than documentation
A: Random seed is 42 by default - changing it will vary results
Q: Import error for task_identity
A: Ensure you're in the task-identity directory and environment is activated

📚 Learning Path
New Users:

✅ Read this STARTUP guide
✅ Run catastrophic_forgetting_full_detection.py
✅ Read main README
✅ Browse results/ folder

Researchers:

✅ Review all 8 test results in results/
✅ Examine per-test READMEs for methodology
✅ Check JSON files for raw data
✅ Run validation scripts to reproduce

Developers:

✅ Study task_identity/__init__.py
✅ Try custom tests with your models
✅ Review validation script implementations
✅ Integrate into your ML pipeline


📞 Getting Help

Quick questions: Check main README.md
Technical issues: Open an issue
Commercial inquiries: Open issue with [commercial] tag
Research collaboration: Open issue with [research] tag


🎯 Quick Reference Card
Installation
bashgit clone https://github.com/Wise314/task-identity.git
cd task-identity
python3 -m venv task-identity-env
source task-identity-env/bin/activate
pip install -r requirements.txt
Usage
pythonfrom task_identity import calculate_task_identity

task_id = calculate_task_identity(
    y_true_before, y_pred_before,
    y_true_after, y_pred_after,
    labels=class_list
)
Interpretation

> 0.95: Stable ✅
0.85-0.95: Minor drift ⚠️
< 0.85: Major change 🚨

Key Files

Core: task_identity/__init__.py
Tests: results/01-08/
Docs: README.md


🏆 Summary
What: Training-free behavioral drift detection via confusion matrix correlation
Why: Traditional metrics miss catastrophic failures (validated with embedding similarity showing 0.583 while Task-Identity correctly shows 0.000)
How: Pearson correlation of confusion matrices across time periods
Validated: 8 comprehensive tests across security, data quality, and training scenarios
Status: Production-ready, validation complete

Last Updated: October 15, 2025
Version: 2.0 (Complete validation)
Status: ✅ Ready for production evaluation
Welcome to Task-Identity! 🚀

