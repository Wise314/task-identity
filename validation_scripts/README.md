# Validation Scripts

**Test implementation scripts for Task-Identity validation**

---

## Overview

This folder contains the implementation scripts used to validate Task-Identity across 8 diverse ML scenarios. Each script generates test results saved in the corresponding `results/` folder.

---

## Production-Ready Tests

### ✅ Test 1: Catastrophic Forgetting
**Script:** `catastrophic_forgetting_full_detection.py`  
**Results:** `results/01_catastrophic_forgetting/`  
**Purpose:** Detect complete task failure in continual learning
```bash
python3 validation_scripts/catastrophic_forgetting_full_detection.py
```

### ✅ Test 2: Progressive Noise
**Script:** `progressive_noise_validator.py`  
**Results:** `results/02_progressive_noise/`  
**Purpose:** Track gradual performance degradation
```bash
python3 validation_scripts/progressive_noise_validator.py
```

### ✅ Test 4: Targeted Poisoning
**Script:** `targeted_poisoning_detection.py`  
**Results:** `results/04_targeted_poisoning/`  
**Purpose:** Detect data poisoning attacks on specific classes
```bash
python3 validation_scripts/targeted_poisoning_detection.py
```

### ✅ Test 5: Cross-Domain Training
**Script:** `cross_domain_behavior_test.py`  
**Results:** `results/05_cross_domain/`  
**Purpose:** Compare models trained on different domains
```bash
python3 validation_scripts/cross_domain_behavior_test.py
```

### ✅ Test 6: Class Imbalance
**Script:** `class_imbalance_detection.py`  
**Results:** `results/06_class_imbalance/`  
**Purpose:** Detect behavioral changes under imbalanced distributions
```bash
python3 validation_scripts/class_imbalance_detection.py
```

### ✅ Test 7: Training Dynamics
**Script:** `training_dynamics_test.py`  
**Results:** `results/07_training_dynamics/`  
**Purpose:** Monitor behavioral convergence during training
```bash
python3 validation_scripts/training_dynamics_test.py
```

### ✅ Test 8: Model Compression
**Script:** `model_compression_test.py`  
**Results:** `results/08_model_compression/`  
**Purpose:** Validate compressed models before deployment
```bash
python3 validation_scripts/model_compression_test.py
```

---

## Experimental/Archive Scripts

### ⚠️ Adversarial Detection (Not Recommended)
**Script:** `adversarial_detection_test.py`  
**Status:** Low success rate (7.1%) - archived for reference  
**Note:** Adversarial attacks via random perturbations proved ineffective with sklearn models

### ⚠️ Model Poisoning (Superseded)
**Script:** `model_poisoning_detection.py`  
**Status:** Replaced by `targeted_poisoning_detection.py`  
**Note:** Targeted poisoning test provides better per-class analysis

### ⚠️ Transfer Learning (Failed)
**Script:** `transfer_learning_validation.py`  
**Status:** Implementation issue - models ended up identical  
**Note:** Replaced by `cross_domain_behavior_test.py` which works correctly

---

## Running All Tests
```bash
# Navigate to project root
cd task-identity

# Activate environment
source task-identity-env/bin/activate

# Run core validation tests (use PYTHONPATH for proper imports)
PYTHONPATH=. python3 validation_scripts/catastrophic_forgetting_full_detection.py
PYTHONPATH=. python3 validation_scripts/progressive_noise_validator.py
PYTHONPATH=. python3 validation_scripts/domain_shift_test.py
PYTHONPATH=. python3 validation_scripts/targeted_poisoning_detection.py
PYTHONPATH=. python3 validation_scripts/cross_domain_behavior_test.py
PYTHONPATH=. python3 validation_scripts/class_imbalance_detection.py
PYTHONPATH=. python3 validation_scripts/training_dynamics_test.py
PYTHONPATH=. python3 validation_scripts/model_compression_test.py
```

**Expected time:** ~30 minutes for all tests

---

## Output Location

All test results are automatically saved to:
- `results/01_catastrophic_forgetting/*.json`
- `results/02_progressive_noise/*.json`
- `results/03_domain_shift/*.json` (manual Fashion-MNIST test)
- `results/04_targeted_poisoning/*.json`
- `results/05_cross_domain/*.json`
- `results/06_class_imbalance/*.json`
- `results/07_training_dynamics/*.json`
- `results/08_model_compression/*.json`

---

## Script Structure

Each validation script follows this pattern:
```python
# 1. Load dataset (MNIST or Fashion-MNIST)
# 2. Apply test-specific intervention (noise, poisoning, etc.)
# 3. Calculate Task-Identity
# 4. Generate detailed results with interpretation
# 5. Save JSON output to results/ folder
```

---

## Requirements

- Python 3.8+
- Virtual environment activated
- Dependencies installed: `pip install -r requirements.txt`
- Internet connection (first run only - downloads MNIST)

---

## Interpreting Results

All scripts output:
- **Task-Identity score** (0.0 to 1.0)
- **Accuracy metrics** (baseline vs intervention)
- **Per-class analysis** (where applicable)
- **Deployment recommendations** (where applicable)
- **JSON file** with complete results

See individual test READMEs in `results/` for detailed interpretation guides.

---

## Troubleshooting

**Q: Script fails with import error**  
A: Ensure you're in project root and environment is activated

**Q: MNIST download fails**  
A: Requires internet connection on first run

**Q: Results differ from documentation**  
A: Random seed is 42 - changing it will vary results

**Q: Script runs but no output file**  
A: Check `results/` folder - files are timestamped

---

## Adding New Tests

To add a new validation test:

1. Create script in `validation_scripts/`
2. Follow existing script structure
3. Save results to appropriate `results/XX_test_name/` folder
4. Create README in results folder
5. Update this README

---

**Last Updated:** October 15, 2025  
**Status:** 7 production tests + 3 experimental/archived
