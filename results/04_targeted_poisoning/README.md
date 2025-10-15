# Test 4: Targeted Poisoning Detection

## Overview
Tests Task-Identity's ability to detect sophisticated data poisoning attacks on specific classes.

## Scenario
1. Train clean model on balanced MNIST
2. Create poisoned model by flipping 60% of class 5→3 and class 8→3 labels during training
3. Compare behavioral patterns between clean and poisoned models

## Key Results

### Overall Metrics
- **Task-Identity:** 0.873 (12.7% behavioral divergence)
- **Accuracy Drop:** 93.6% → 79.6% (14.9% degradation)

### Per-Class Detection
- **Class 5:** Task-Identity = 0.171 (SEVERE DRIFT) 🚨
  - 85% of "5"s misclassified as "3"s
- **Class 8:** Task-Identity = 0.177 (SEVERE DRIFT) 🚨
  - 76% of "8"s misclassified as "3"s
- **All other classes:** Task-Identity = 1.000 (STABLE) ✅

## Conclusion
🎯 **SUCCESS:** Task-Identity detected targeted poisoning attack AND pinpointed exactly which classes were compromised.

## Commercial Value
- Detect data poisoning attacks in training pipelines
- Identify backdoored models before deployment
- Security validation for third-party models
- Monitor for adversarial data injection

## Test Files
- `targeted_poisoning_*.json` - Poisoning attack test results

## Key Insight
This demonstrates Task-Identity's most powerful feature: **per-class analysis**. While overall metrics showed moderate changes (0.873), per-class analysis revealed catastrophic compromise of specific classes (0.17). This level of granularity is critical for security applications.
