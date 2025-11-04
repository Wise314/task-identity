# Post-Provisional Patent Work

**Created:** November 4, 2025  
**Status:** Work completed AFTER provisional patent filing

This folder contains validation tests and results developed after the provisional patent was filed. This work extends Task-Identity to additional domains and validates advanced detection methods.

---

## Test #12: Financial Lending Behavioral Drift Detection

**Validation Date:** November 4, 2025  
**Dataset:** Lending Club 2007-2018 (2,260,701 real loans)  
**Method:** Per-Class Task-Identity (Test #4 approach)  
**Domain:** Financial Services / Credit Risk

### Challenge

Financial data has extreme class imbalance (86.9% paid vs 13.1% defaults), causing standard Task-Identity to miss catastrophic drift in minority class behavior.

### Solution

Applied per-class Task-Identity analysis (from Test #4 Targeted Poisoning) to isolate minority class behavior.

### Results

| Metric | Value | Status |
|--------|-------|--------|
| Overall Task-Identity | 0.921 | ❌ MISSED (appeared stable) |
| Paid Class Task-Identity | 0.949 | ✅ STABLE (correct) |
| Default Class Task-Identity | 0.000 | ✅ DETECTED CATASTROPHIC DRIFT |
| Detection Improvement | +92.1 points | Massive improvement |

**Behavioral Change Detected:**
- Baseline model: 81.8% default detection rate
- Drifted model: 0.5% default detection rate  
- 99.4% degradation in critical minority class

### Validation Files

- **Test Script:** `validation_scripts/financial_lending_test.py`
- **Results:** `results/12_financial_lending/lending_test_*.json`
- **Dataset:** Lending Club 2007-2018 (external - not included in repo)

### Key Finding

Per-class analysis is ESSENTIAL for class-imbalanced domains. Overall Task-Identity missed catastrophic drift due to majority class dominance, while per-class analysis successfully detected the behavioral collapse.

---

## Future Work

- Test #13: Credit scoring behavioral drift
- Test #14: Insurance claim fraud detection  
- Test #15: Stock trading algorithm monitoring

---

**Note:** This work is documented separately from the provisional patent filing to maintain clear distinction between filed and post-filing validation work.
