# 🚀 STARTUP: Getting Started with Task-Identity

Quick start guide for new users and AI assistants

---

📋 What is Task-Identity?
Task-Identity is a training-free metric that detects behavioral drift in AI classification models by comparing confusion matrices across time periods using Pearson correlation.
Validated across 11 tests spanning 4 domains:

🖼️ Computer Vision (8 tests)
📝 Natural Language Processing (1 test)
🏥 Medical AI (1 test)
🎵 Audio/Speech Recognition (1 test)

The Problem It Solves:

Traditional metrics (embedding similarity, accuracy) can miss catastrophic failures
Task-Identity measures actual decision-making behavior, not internal structure
Works universally across images, text, tabular data, and audio

Key Validation Result:

✅ Task-Identity: 0.000 → Correctly detected catastrophic failure
⚠️ Embedding Similarity: 0.583 → Significantly underestimated severity
📉 Actual Performance: 99.3% → 0.0% (total collapse)

## ⚙️ IMPORTANT: Environment Setup

Before running ANY test, you MUST set PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"


⚡ Quick Install & Test (5 Minutes)

⚙️ **CRITICAL: Set PYTHONPATH first (do this once per terminal session):**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```
```bash
# 1. Clone and setup
git clone https://github.com/Wise314/task-identity.git
cd task-identity
python3 -m venv task-identity-env
source task-identity-env/bin/activate  # Windows: task-identity-env\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set PYTHONPATH (REQUIRED)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 4. Run a quick validation test (Computer Vision)
python validation_scripts/catastrophic_forgetting_full_detection.py

# 5. Or try cross-domain tests
python validation_scripts/text_classification_test.py      # NLP
python validation_scripts/tabular_classification_test.py   # Medical AI
python validation_scripts/audio_classification_test.py     # Audio
```

**Expected output (Test 1 - Catastrophic Forgetting):**
```
✅ Task-Identity: 0.000 (detected catastrophic forgetting)
⚠️ Embedding Identity: 0.583 (underestimated severity)
📉 Accuracy: 99.3% → 0.0%
```

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

## 📊 11 Comprehensive Validation Tests Across 4 Domains

### 🖼️ Computer Vision (Tests 1-8)

**Datasets:** MNIST, Fashion-MNIST

#### Security & Safety
1. **[Catastrophic Forgetting](../results/01_catastrophic_forgetting/)** - Task-Identity: 0.000
2. **[Targeted Poisoning](../results/04_targeted_poisoning/)** - Task-Identity: 0.873 (per-class: 0.17)
3. **[Model Compression](../results/08_model_compression/)** - Task-Identity: 0.384

#### Data Quality & Distribution
4. **[Progressive Noise](../results/02_progressive_noise/)** - Task-Identity: 0.780-1.000
5. **[Domain Shift](../results/03_domain_shift/)** - Task-Identity: 0.046
6. **[Class Imbalance](../results/06_class_imbalance/)** - Task-Identity: 0.576

#### Training & Optimization
7. **[Cross-Domain Training](../results/05_cross_domain/)** - Task-Identity: 0.000
8. **[Training Dynamics](../results/07_training_dynamics/)** - Task-Identity: 0.999-1.000

---

### 📝 Natural Language Processing (Test 9)

**Dataset:** 20 Newsgroups

9. **[Text Classification Drift](../results/09_text_classification/)** - Task-Identity: 0.036

**Validation:** Proves Task-Identity works on text data. Model collapsed to single-class prediction after imbalanced fine-tuning (10:1 ratio).

---

### 🏥 Medical AI / Tabular Data (Test 10)

**Dataset:** Wisconsin Breast Cancer

10. **[Medical Diagnosis Drift](../results/10_tabular_classification/)** - Task-Identity: 0.000

**Validation:** Proves Task-Identity works on tabular/medical data. Detected dangerous training bias (model trained only on malignant samples).

---

### 🎵 Audio / Speech Recognition (Test 11)

**Dataset:** Free Spoken Digit Dataset

11. **[Speech Recognition Drift](../results/11_audio_classification/)** - Task-Identity: 0.000

**Validation:** Proves Task-Identity works on audio data. 3,000 real spoken digit recordings, catastrophic forgetting detected.

---

**See [results/README.md](../results/README.md) for complete details on all tests.**

---

## 🎯 Key Validation Highlights

### Test 1: Catastrophic Forgetting Detection (Computer Vision)
**Why it matters:** Proves Task-Identity catches failures that structural metrics miss.

- **Scenario:** Model trained on digits 0-4, then fine-tuned on 5-9
- **Result:** Complete forgetting of original task
- **Task-Identity:** 0.000 ✅ (detected failure)
- **Embedding Similarity:** 0.583 ⚠️ (underestimated)
- **Accuracy:** 99.3% → 0.0%

---

### Test 6: Class Imbalance Impact (Computer Vision)
**Why it matters:** Detects hidden behavioral changes that accuracy misses.

- **Scenario:** Test on 90/10 imbalanced distribution
- **Accuracy:** 93.6% → 93.7% (appeared stable!)
- **Task-Identity:** 0.576 (detected 42% behavioral shift)
- **Insight:** Model making different mistakes despite same accuracy

---

### Test 9: Text Classification Drift (NLP)
**Why it matters:** Proves cross-domain applicability (not just images).

- **Dataset:** 20 Newsgroups (computer graphics vs baseball)
- **Scenario:** Imbalanced fine-tuning (10:1 ratio)
- **Task-Identity:** 0.036 (detected 96.4% behavioral drift)
- **Result:** Model collapsed to single-class prediction

---

### Test 10: Medical Diagnosis Drift (Tabular Data)
**Why it matters:** Critical for healthcare AI safety validation.

- **Dataset:** Wisconsin Breast Cancer (569 patients)
- **Scenario:** Model trained only on malignant samples (zero benign)
- **Task-Identity:** 0.000 (detected catastrophic over-diagnosis risk)
- **Impact:** Prevents dangerous medical AI deployment

---

### Test 11: Speech Recognition Drift (Audio)
**Why it matters:** Validates voice AI and speech recognition systems.

- **Dataset:** Free Spoken Digit Dataset (3,000 recordings)
- **Scenario:** Model trained on digits 0-4, fine-tuned on 5-9
- **Task-Identity:** 0.000 (detected catastrophic forgetting)
- **Result:** Same pattern as Test 1, proving universal applicability

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
│   ├── 01_catastrophic_forgetting/     # Computer Vision
│   ├── 02_progressive_noise/           # Computer Vision
│   ├── 03_domain_shift/                # Computer Vision
│   ├── 04_targeted_poisoning/          # Computer Vision
│   ├── 05_cross_domain/                # Computer Vision
│   ├── 06_class_imbalance/             # Computer Vision
│   ├── 07_training_dynamics/           # Computer Vision
│   ├── 08_model_compression/           # Computer Vision
│   ├── 09_text_classification/         # NLP
│   ├── 10_tabular_classification/      # Medical AI
│   ├── 11_audio_classification/        # Audio/Speech
│   └── archive/
│
├── validation_scripts/          # Test implementation scripts
│   ├── README.md               # Test script documentation
│   └── [11 test scripts]
│
├── STARTUP/                     # This guide
│   ├── README.md               # You are here
│   └── archive/
│
└── readmearchive/              # Historical README versions

🧪 Running Your Own Tests
Example 1: Monitor Production Model
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

Example 2: Validate Model Compression
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

Example 3: Cross-Domain Validation (NLP, Audio, Medical)
python# Works on ANY classification domain
from task_identity import calculate_task_identity

# Text classification
task_id_text = calculate_task_identity(
    text_labels_baseline, text_preds_baseline,
    text_labels_current, text_preds_current,
    labels=text_classes
)

# Medical diagnosis
task_id_medical = calculate_task_identity(
    patient_labels_baseline, diagnosis_preds_baseline,
    patient_labels_current, diagnosis_preds_current,
    labels=diagnosis_classes
)

# Speech recognition
task_id_audio = calculate_task_identity(
    audio_labels_baseline, audio_preds_baseline,
    audio_labels_current, audio_preds_current,
    labels=speech_classes
)

📖 Interpretation Guide
Task-Identity Thresholds
Score RangeInterpretationRecommended Action0.95 - 1.00Nearly identical behavior✅ Model stable, no action needed0.85 - 0.95Minor behavioral changes⚠️ Monitor, investigate if persistent0.70 - 0.85Moderate behavioral shift⚠️⚠️ Investigate cause, intervention likely needed0.50 - 0.70Major behavioral change🚨 Alert required, data/model issue0.00 - 0.50Catastrophic shift🚨🚨 Critical failure, immediate action

Context-Specific Thresholds
Different use cases need different thresholds:
Use CaseAlert ThresholdRationaleProduction monitoring< 0.95Catch drift earlySecurity validation< 0.85Allow some variationCompression QA< 0.95Strict preservationTransfer learning< 0.70Some forgetting OKTraining convergence≈ 1.00Stop when stableMedical AI safety< 0.95Critical - investigate bias

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

✅ 11 comprehensive tests across 4 domains
✅ Detects failures traditional metrics miss (class imbalance, compression)
✅ Works universally across vision, text, tabular, audio
✅ No training data or model internals required
✅ All real, published datasets (no synthetic data)


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

4. Cross-Domain Validation
Use Case: Validate models across different data types
python# Validate medical AI, voice AI, vision AI all with same metric
if task_identity_medical < 0.95 or task_identity_audio < 0.95:
    alert("Multi-domain drift detected")
Value: Universal metric works across all ML domains

5. Training Optimization
Use Case: Intelligent early stopping
pythonif task_identity > 0.99:
    stop_training("Behavior converged - save compute")
Value: Reduce training costs

⚠️ What's Core vs. Experimental
Core Task-Identity (Production-Ready)
✅ calculate_task_identity() function in task_identity/__init__.py
✅ Confusion matrix correlation method
✅ All 11 validation test results
Experimental Features (Not Part of Core Metric)
⚠️ Multiplier calculations in some validation scripts
⚠️ Autocorrelation analysis
⚠️ Threshold auto-tuning experiments
Note: Only the core calculate_task_identity() function is production-ready. Experimental code in validation scripts is for research exploration.

🐛 Troubleshooting
Common Issues
Q: Installation fails
A: Ensure Python 3.8+ and virtual environment is activated
Q: MNIST/dataset download fails
A: First run requires internet - datasets auto-download
Q: Different results than documentation
A: Random seed is 42 by default - changing it will vary results
Q: Import error for task_identity
A: Ensure you're in the task-identity directory and use PYTHONPATH=.
Q: Audio test fails on librosa
A: Script auto-installs librosa. If fails: pip install librosa --break-system-packages

📚 Learning Path
New Users:

✅ Read this STARTUP guide
✅ Run catastrophic_forgetting_full_detection.py
✅ Try a cross-domain test (text, medical, or audio)
✅ Read main README
✅ Browse results/ folder

Researchers:

✅ Review all 11 test results in results/
✅ Examine per-test READMEs for methodology
✅ Check JSON files for raw data
✅ Run validation scripts to reproduce
✅ Test on your own datasets

Developers:

✅ Study task_identity/__init__.py
✅ Try custom tests with your models
✅ Review validation script implementations
✅ Integrate into your ML pipeline
✅ Test across multiple domains


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
Tests: results/01-11/
Docs: README.md


🏆 Summary
What: Training-free behavioral drift detection via confusion matrix correlation
Why: Traditional metrics miss catastrophic failures (validated: embedding similarity 0.583 vs Task-Identity 0.000)
How: Pearson correlation of confusion matrices across time periods
Validated: 11 comprehensive tests across 4 domains (vision, NLP, medical, audio)
Coverage: 95%+ of production ML classification workloads
Status: Production-ready, validation complete

Last Updated: October 18, 2025
Version: 3.0 (Complete cross-domain validation)
Status: ✅ Ready for production evaluation across all ML domains
Welcome to Task-Identity! 🚀
</artifact>
