# Task-Identity: Behavioral Drift Detection for AI Systems

**Novel metric for detecting behavioral changes in classification models**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests: 8/8 Passing](https://img.shields.io/badge/tests-8%2F8%20passing-brightgreen.svg)](results/)

---

🎯 Overview
Task-Identity is a training-free metric that measures behavioral similarity between AI classification models across time periods. Unlike embedding-based drift detection, Task-Identity directly measures what the model actually does - its decision-making patterns through confusion matrix correlation.
Why Task-Identity?
Traditional drift detection methods can miss critical failures:
Example: Catastrophic Forgetting Detection

Embedding Similarity: 0.583 (appears moderate - underestimates severity)
Task-Identity: 0.000 (detects complete failure)
Actual Performance: 99.3% → 0.0% accuracy (total collapse)

✅ Task-Identity caught the catastrophic failure
⚠️ Embedding similarity significantly underestimated it

✨ Key Features

✅ Training-Free - No ML model required, pure mathematical correlation
✅ Lightweight - O(K²) complexity, runs in milliseconds
✅ Universal - Works on any classification model (CNNs, Transformers, MLPs, etc.)
✅ No Training Data Needed - Only requires predictions from two time periods
✅ Interpretable - 0.0 = different behavior, 1.0 = identical behavior
✅ Per-Class Analysis - Pinpoints which specific classes are affected
✅ Detects Hidden Failures - Finds behavioral changes that accuracy metrics miss


🚀 Quick Start
Installation
bash# Clone repository
git clone https://github.com/Wise314/task-identity.git
cd task-identity

# Create virtual environment
python3 -m venv task-identity-env
source task-identity-env/bin/activate  # On Windows: task-identity-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Basic Usage
pythonfrom task_identity import calculate_task_identity

# Compare model behavior across two time periods
task_id = calculate_task_identity(
    y_true_baseline,      # True labels from baseline period
    y_pred_baseline,      # Model predictions from baseline period
    y_true_current,       # True labels from current period
    y_pred_current,       # Model predictions from current period
    labels=range(10)      # Complete set of class labels
)

# Interpret result
if task_id < 0.85:
    print(f"🚨 Behavioral change detected! Task-Identity: {task_id:.3f}")
else:
    print(f"✅ Model behavior is stable. Task-Identity: {task_id:.3f}")

📊 Comprehensive Validation Results
Task-Identity has been validated across 8 diverse ML scenarios spanning security, data quality, and training optimization. View all results →
Security & Safety Tests
TestTask-IdentityKey FindingDetailsCatastrophic Forgetting0.000Detected complete task failureREADMETargeted Poisoning0.873 (per-class: 0.17)Pinpointed poisoned classesREADMEModel Compression0.384Blocked broken deploymentREADME
Data Quality & Distribution Tests
TestTask-IdentityKey FindingDetailsProgressive Noise0.780-1.000Tracked gradual degradationREADMEDomain Shift0.049Detected cross-domain mismatchREADMEClass Imbalance0.576Found bias accuracy missedREADME
Training & Optimization Tests
TestTask-IdentityKey FindingDetailsCross-Domain Training0.000Compared training provenanceREADMETraining Dynamics0.999-1.000Detected convergence pointREADME
Highlighted Result: Class Imbalance Detection
The test that proves Task-Identity's value:

Accuracy: 93.6% → 93.7% (appeared stable ✅)
Task-Identity: 0.576 (detected 42.4% behavioral shift 🚨)

Traditional metrics showed "everything is fine" while Task-Identity revealed the model was making fundamentally different mistakes under imbalanced conditions - exactly the kind of hidden bias that causes real-world ML failures.

🔬 How It Works
The Method

Collect predictions from model at Time Period 1
Collect predictions from model at Time Period 2
Generate confusion matrices for both periods
Calculate Pearson correlation between flattened matrices
Result: Task-Identity score ∈ [0.0, 1.0]

Why Confusion Matrices?
Confusion matrices capture what the model confuses with what - the complete behavioral fingerprint:

High Task-Identity (>0.95): Model makes same mistakes → behavioral consistency
Low Task-Identity (<0.20): Model makes different mistakes → behavioral shift detected


💡 Production Use Cases
1. Compression Validation (Pre-Deployment)
Validate model compression before deploying to edge devices:
pythontask_id = calculate_task_identity(
    y_true, preds_full_precision,
    y_true, preds_compressed
)

if task_id < 0.95:
    print("❌ Compression degraded behavior - reject deployment")
else:
    print("✅ Compression preserved behavior - safe to deploy")
Real Result: Task-Identity (0.384) blocked deployment of 4x compressed model that destroyed 6 out of 10 classes.
2. Production Monitoring
Detect behavioral drift in deployed models:
python# Weekly comparison
if task_identity < 0.85:
    alert("Model behavior has shifted - investigate data drift")
3. Security Scanning
Detect data poisoning attacks:
python# Per-class analysis reveals poisoned classes
class_5_task_id = 0.171  # Severe drift
class_8_task_id = 0.177  # Severe drift
# Other classes: 1.000 (stable)
# Conclusion: Classes 5 and 8 compromised by poisoning attack
4. A/B Testing & Model Comparison
Compare behavioral similarity between model variants:
pythontask_id = calculate_task_identity(
    y_true, model_a_predictions,
    y_true, model_b_predictions
)

if task_id > 0.95:
    print("Models behave nearly identically")
else:
    print(f"Behavioral difference: {(1-task_id)*100:.1f}%")
5. Training Optimization
Detect when training has converged to save compute:
python# Compare model at epoch N vs epoch N+10
if task_identity > 0.99:
    print("Training converged - stop to save compute")

📈 Comparison to Existing Methods
MethodWhat It MeasuresKey LimitationTask-Identity AdvantageEmbedding DriftInternal representationsShowed 0.583 during complete failureDetected 0.000 (correct)Data DriftInput distributionDoesn't measure model behaviorDirect behavioral measurementAccuracy MonitoringSingle performance metricMissed 42% behavioral shiftCaught hidden distribution biasConfusion LoggingError patternsManual analysis requiredAutomated quantitative scoring

📚 API Reference
Core Function
pythonfrom task_identity import calculate_task_identity

task_id = calculate_task_identity(
    y_true_before,   # True labels from baseline period
    y_pred_before,   # Predictions from baseline period
    y_true_after,    # True labels from current period
    y_pred_after,    # Predictions from current period
    labels           # Complete set of class labels
)
Parameters:

y_true_before (array-like): True labels from baseline period
y_pred_before (array-like): Predictions from baseline period
y_true_after (array-like): True labels from current period
y_pred_after (array-like): Predictions from current period
labels (array-like): Complete set of class labels

Returns:

float: Task-Identity score [0.0, 1.0]

Example:
pythonimport numpy as np
from task_identity import calculate_task_identity

# Simulate baseline and current predictions
y_true = np.array([0, 1, 2, 0, 1, 2])
baseline_preds = np.array([0, 1, 2, 0, 1, 2])  # Perfect
current_preds = np.array([0, 2, 1, 0, 2, 1])   # Confuses 1↔2

task_id = calculate_task_identity(
    y_true, baseline_preds,
    y_true, current_preds,
    labels=[0, 1, 2]
)
print(f"Task-Identity: {task_id:.3f}")  # Will show behavioral change
```

---

## 📖 Interpretation Guide

| Task-Identity | Interpretation | Recommended Action |
|--------------|----------------|-------------------|
| **0.95 - 1.00** | Nearly identical behavior | ✅ Model stable, no action needed |
| **0.85 - 0.95** | Minor behavioral changes | ⚠️ Monitor closely, investigate if persistent |
| **0.70 - 0.85** | Moderate behavioral shift | ⚠️⚠️ Investigate cause, may need intervention |
| **0.50 - 0.70** | Major behavioral change | 🚨 Alert required, likely data/model issue |
| **0.00 - 0.50** | Catastrophic behavioral shift | 🚨🚨 Critical failure, immediate action required |

### Context-Specific Thresholds

Different use cases require different thresholds:

| Use Case | Threshold | Rationale |
|----------|-----------|-----------|
| Production monitoring | < 0.95 | Detect drift early |
| Security validation | < 0.85 | Higher tolerance for attacks |
| Compression validation | < 0.95 | Strict preservation requirement |
| Transfer learning | < 0.70 | Some forgetting acceptable |
| Training convergence | ≈ 1.00 | Stop when behavior stabilizes |

---

## ⚠️ Limitations

- **Classification only:** Currently works for classification tasks (not regression)
- **Requires labels:** Needs true labels for both time periods (can work with sampled/batch labels)
- **Class consistency:** Assumes same set of classes across periods
- **Sample size:** Requires sufficient samples for stable confusion matrices (recommended: 100+ per class)

---

## 🗂️ Repository Structure
```
task-identity/
├── README.md                                    # This file
├── LICENSE                                      # MIT License
├── requirements.txt                             # Python dependencies
├── task_identity/                               # Core package
│   └── __init__.py                              # Core calculate_task_identity()
├── results/                                     # Comprehensive validation results
│   ├── README.md                                # Results overview & summary
│   ├── 01_catastrophic_forgetting/              # Test 1 with detailed README
│   ├── 02_progressive_noise/                    # Test 2 with detailed README
│   ├── 03_domain_shift/                         # Test 3 with detailed README
│   ├── 04_targeted_poisoning/                   # Test 4 with detailed README
│   ├── 05_cross_domain/                         # Test 5 with detailed README
│   ├── 06_class_imbalance/                      # Test 6 with detailed README
│   ├── 07_training_dynamics/                    # Test 7 with detailed README
│   ├── 08_model_compression/                    # Test 8 with detailed README
│   └── archive/                                 # Historical results
├── validation_scripts/                          # Test scripts (various)
└── readmearchive/                               # Previous README versions

🔄 Recent Updates
October 15, 2024:

✅ Completed comprehensive validation across 8 diverse ML scenarios
✅ Organized all test results into structured folders with detailed documentation
✅ Validated against security attacks, data quality issues, and training optimization
✅ Demonstrated universal applicability across different failure modes
✅ Added per-class analysis capability for targeted attack detection
✅ Proved Task-Identity detects failures that traditional metrics miss

Earlier (October 15, 2024):

✅ Refactored core algorithm into task_identity/__init__.py
✅ Removed code duplication across validation scripts
✅ Added comprehensive input validation
✅ Made random seed configurable for reproducibility


📄 License
MIT License - See LICENSE file for details.

🎓 Research & Citation
Discovery Date: October 14, 2024
Status: Validation Complete
Inventor: Shawn Barnicle
If you use Task-Identity in your research or production systems, please cite:
bibtex@software{task_identity_2024,
  title={Task-Identity: Training-Free Behavioral Drift Detection for AI Systems},
  author={Barnicle, Shawn},
  year={2024},
  url={https://github.com/Wise314/task-identity},
  note={Validated across 8 ML scenarios including security, data quality, and training optimization}
}

📞 Contact & Support

Issues & Questions: Open an issue
Commercial inquiries: Open an issue with [commercial] tag
Research collaboration: Open an issue with [research] tag


🙏 Acknowledgments
Discovered while investigating adaptive monitoring methods for AI systems. The key insight - that neural networks can maintain moderate structural similarity during catastrophic behavioral failure - emerged from systematic testing across multiple failure modes.
Validation testing conducted with assistance from Claude (Anthropic).

Last Updated: October 15, 2025
Validation Status: ✅ Complete (8/8 tests passing)
Ready for: Production evaluation, research collaboration, commercial licensing
</artifact>
