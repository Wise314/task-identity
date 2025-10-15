# Test Results

This directory contains validation test results for Task-Identity.

---

## Test Files

### Catastrophic Forgetting Tests
**Files:** `catastrophic_forgetting_full_*.json`

**What it tests:** Neural network trained on MNIST digits 0-4, then fine-tuned on digits 5-9 (catastrophic forgetting scenario)

**Key findings:**
- **Task-Identity:** 0.000 (detected complete behavioral failure)
- **Embedding Identity:** 0.583 (underestimated severity - showed only moderate change)
- **Accuracy:** 99.3% → 0.0% (total collapse)

**Conclusion:** Task-Identity correctly identified catastrophic failure while embedding similarity significantly underestimated the severity.

**Recent results:** Check files with latest timestamps in this directory

---

### Progressive Noise Tests
**Files:** `progressive_noise_*.json`

**What it tests:** MNIST with increasing Gaussian noise (0% → 30%)

**Key findings:**
- Task-Identity smoothly tracks degradation: 1.000 → 0.780
- Accuracy correlates: 93.6% → 61.4%

**Conclusion:** Task-Identity effectively tracks gradual performance degradation.

---

### Domain Shift Test
**Files:** `FASHION_TASK_IDENTITY_*.json`

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
  "alpha_results": { ... }  // Experimental threshold detection (NOT core Task-Identity)
}
```

**Note:** The `alpha_results` section contains experimental threshold detection code. This is NOT part of the core Task-Identity metric. The core metric is the `task_identity` value itself.

---

## Interpreting Results

### Task-Identity Values

| Task-Identity | Meaning | Status |
|--------------|---------|--------|
| 0.95 - 1.00 | Identical behavior | ✅ Stable |
| 0.80 - 0.95 | Minor changes | ⚠️ Monitor |
| 0.50 - 0.80 | Moderate shift | ⚠️⚠️ Investigate |
| 0.20 - 0.50 | Major change | 🚨 Alert |
| 0.00 - 0.20 | Catastrophic failure | 🚨🚨 Critical |

### Key Metrics

**Core Task-Identity Metric:**
- `task_identity`: The behavioral similarity score (0.0 to 1.0)
- This is the patented metric - Pearson correlation of confusion matrices

**Comparison Metrics:**
- `embedding_identity`: Internal structural similarity (for comparison)
- Shows Task-Identity advantage when it correctly detects failure (0.000) while embeddings underestimate (0.583)

**Performance Metrics:**
- `baseline_accuracy`: Model accuracy in baseline period
- `shifted_accuracy`: Model accuracy in current period
- Used to validate that Task-Identity correlates with actual performance changes

---

## File Naming Convention

Files are named with timestamps for traceability:
- Format: `{test_name}_{YYYYMMDD}_{HHMMSS}.json`
- Example: `catastrophic_forgetting_full_20251015_174721.json`

Most recent files have the latest timestamps.

---

## Archive

Previous versions of this README are stored in `results/archive/` for reference.

---

**Last Updated:** October 15, 2024
