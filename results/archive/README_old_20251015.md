# Test Results

This directory contains validation test results for Task-Identity.

## Test Files

### Catastrophic Forgetting Tests
Files: `catastrophic_forgetting_full_*.json`

**What it tests:** Neural network trained on MNIST digits 0-4, then fine-tuned on digits 5-9 (catastrophic forgetting scenario)

**Key findings:**
- **Task-Identity:** 0.000 (detected complete behavioral failure)
- **Embedding Identity:** 0.583 (underestimated severity - showed only moderate change)
- **Accuracy:** 99.3% → 0.0% (total collapse)

**Conclusion:** Task-Identity correctly identified catastrophic failure while embedding similarity significantly underestimated the severity.

**Latest result:** `catastrophic_forgetting_full_20251015_134214.json`

---

### Progressive Noise Tests
File: `progressive_noise_20251014_181010.json`

**What it tests:** MNIST with increasing Gaussian noise (0% → 30%)

**Key findings:**
- Task-Identity smoothly tracks degradation: 1.000 → 0.780
- Accuracy correlates: 93.6% → 61.4%

**Conclusion:** Task-Identity effectively tracks gradual performance degradation.

---

### Domain Shift Test
File: `FASHION_TASK_IDENTITY_20251014_170648.json`

**What it tests:** Model trained on MNIST, tested on Fashion-MNIST

**Key findings:**
- **Task-Identity:** 0.049 (very low similarity - detected domain shift)
- **Accuracy:** 93.6% → 12.8% (severe degradation)

**Conclusion:** Task-Identity detects cross-domain behavioral differences.

---

## Result File Format

Each JSON file contains:
```json
{
  "test": "test_name",
  "timestamp": "YYYYMMDD_HHMMSS",
  "task_identity": 0.0-1.0,
  "embedding_identity": 0.0-1.0,  // (catastrophic forgetting tests only)
  "baseline_accuracy": 0.0-1.0,
  "shifted_accuracy": 0.0-1.0,
  "alpha_results": { ... }  // Detection threshold analysis
}
```

## Interpreting Results

| Task-Identity | Meaning |
|--------------|---------|
| 0.95 - 1.00 | Identical behavior |
| 0.80 - 0.95 | Minor changes |
| 0.50 - 0.80 | Moderate shift |
| 0.20 - 0.50 | Major change |
| 0.00 - 0.20 | Catastrophic failure |
