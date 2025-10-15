<artifact identifier="task-identity-readme" type="text/markdown" title="README.md">
# Task-Identity: Behavioral Drift Detection for AI Systems
Training-free behavioral monitoring for classification models via confusion matrix correlation
Show Image
Show Image

Overview
Task-Identity is a novel metric for detecting behavioral changes in AI classification models. Unlike embedding-based or statistical drift detection methods, Task-Identity measures what the model actually does rather than how it represents data internally.
Key Innovation
Neural networks can maintain high embedding similarity (0.995) even during catastrophic failure. Task-Identity detects these failures by measuring behavioral similarity through confusion matrix correlation.
Example:

Embedding Similarity: 0.995 (structure preserved - looks fine!)
Task-Identity: 0.000 (behavior destroyed - complete failure!)


Features
✅ Training-free - Only requires predictions from two time periods
✅ Lightweight - Pure correlation math, runs on embedded systems
✅ Universal - Works on any classification model (CNNs, Transformers, etc.)
✅ Interpretable - 0.0 = different behavior, 1.0 = identical behavior
✅ Catches what others miss - Detects failures embedding similarity ignores

Installation
bash# Clone repository
git clone https://github.com/Wise314/task-identity.git
cd task-identity

# Create virtual environment
python3 -m venv task-identity-env
source task-identity-env/bin/activate  # On Windows: task-identity-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

Quick Start
Calculate Task-Identity
pythonimport numpy as np
from sklearn.metrics import confusion_matrix

def calculate_task_identity(y_true_before, y_pred_before, 
                            y_true_after, y_pred_after, labels):
    """
    Calculate behavioral similarity between two time periods.
    
    Returns:
        float: 0.0 (completely different) to 1.0 (identical behavior)
    """
    # Generate confusion matrices
    cm_before = confusion_matrix(y_true_before, y_pred_before, labels=labels)
    cm_after = confusion_matrix(y_true_after, y_pred_after, labels=labels)
    
    # Flatten and correlate
    flat_before = cm_before.flatten()
    flat_after = cm_after.flatten()
    
    if flat_before.std() == 0 or flat_after.std() == 0:
        return 0.0
    
    correlation = np.corrcoef(flat_before, flat_after)[0, 1]
    return max(0.0, correlation) if not np.isnan(correlation) else 0.0
Run Validation Tests
bash# Test 1: Catastrophic Forgetting Detection
python3 catastrophic_forgetting_full_detection.py

# Test 2: Progressive Degradation Tracking
python3 progressive_noise_validator.py

Validation Results
Catastrophic Forgetting Detection
MetricValueStatusTask-Identity0.000✅ Detected complete failureEmbedding Similarity0.995❌ Missed the failureAccuracy Drop99.3% → 0.0%Complete behavioral change
Conclusion: Task-Identity catches catastrophic failures that embedding-based methods completely miss.

Progressive Degradation Tracking
Noise LevelTask-IdentityAccuracyInterpretation0%1.00093.6%Baseline (identical)10%0.99992.2%Minimal drift20%0.94879.3%Moderate degradation30%0.78061.4%Severe degradation
Conclusion: Task-Identity smoothly tracks gradual degradation, correlating strongly with performance decline.

Use Cases
Production AI Monitoring
Monitor deployed models for behavioral drift without requiring:

Original training data
Model retraining
Expensive embedding extraction

Catastrophic Forgetting Detection
Identify when continual learning causes models to forget previous tasks:
pythonif task_identity < 0.1:
    alert("Critical: Model has forgotten original task")
Domain Shift Detection
Detect when production data differs from training distribution:
python# Train on MNIST, deploy on real-world data
task_identity = 0.049  # Low similarity indicates domain shift
A/B Testing
Compare behavioral similarity between model variants:
pythonif task_identity > 0.95:
    print("Models behave nearly identically")
else:
    print(f"Behavioral difference: {1 - task_identity:.2%}")
```

---

## How It Works

### The Method

1. **Collect predictions** from model at Time Period 1
2. **Collect predictions** from model at Time Period 2
3. **Generate confusion matrices** for both periods
4. **Calculate correlation** between flattened matrices
5. **Task-Identity** = correlation coefficient [0.0, 1.0]

### Why Confusion Matrices?

Confusion matrices capture **what the model confuses with what** - the essence of behavioral patterns:
```
High Task-Identity (0.95):
- Model makes same mistakes
- Behavioral consistency

Low Task-Identity (0.05):
- Model makes different mistakes
- Behavioral shift detected

Comparison to Existing Methods
MethodWhat It MeasuresLimitationTask-Identity AdvantageEmbedding DriftInternal representationsMisses behavioral changes (0.995 during failure)Measures actual behavior (0.000 catches failure)Data DriftInput distributionDoesn't measure model behaviorDirect behavioral measurementAccuracy MonitoringSingle performance metricRequires continuous labelsWorks with delayed/batch labelsTask-IdentityDecision patternsNone identifiedTraining-free + catches what others miss

Validation Test Details
Test 1: Catastrophic Forgetting

Dataset: MNIST digits
Scenario: Train on 0-4, fine-tune on 5-9
Result: Task-Identity = 0.000 (detected), Embedding = 0.995 (missed)
Output: results/catastrophic_forgetting_full_*.json

Test 2: Progressive Noise

Dataset: MNIST with Gaussian noise (0-30%)
Scenario: Gradual degradation simulation
Result: Smooth correlation (1.000 → 0.780)
Output: results/progressive_noise_*.json

Test 3: Domain Shift

Dataset: MNIST → Fashion-MNIST
Scenario: Cross-domain transfer
Result: Task-Identity = 0.049 (very low similarity)
Output: results/FASHION_TASK_IDENTITY_*.json


Technical Specifications
Computational Complexity: O(K²) where K = number of classes
Memory Requirements: Two K×K confusion matrices
Runtime: Milliseconds for typical problems
Scalability: Tested up to 10,000 samples per evaluation

API Reference
Core Function
pythoncalculate_task_identity(y_true_before, y_pred_before, 
                       y_true_after, y_pred_after, labels)
Parameters:

y_true_before (array): True labels from baseline period
y_pred_before (array): Predictions from baseline period
y_true_after (array): True labels from current period
y_pred_after (array): Predictions from current period
labels (array): Complete set of class labels

Returns:

float: Task-Identity score [0.0, 1.0]


Interpretation Guide
Task-IdentityInterpretationAction0.95 - 1.00Nearly identical behavior✅ Model stable0.80 - 0.95Minor behavioral changes⚠️ Monitor closely0.50 - 0.80Moderate behavioral shift⚠️⚠️ Investigate0.20 - 0.50Major behavioral change🚨 Alert required0.00 - 0.20Catastrophic behavioral shift🚨🚨 Critical failure

Limitations

Classification only: Currently works for classification tasks (not regression)
Requires labels: Needs true labels for both time periods
Class consistency: Assumes same set of classes across periods


Research & Citation
Discovery Date: October 14, 2024
Status: Patent pending
If you use Task-Identity in your research or production systems, please cite:
bibtex@software{task_identity_2024,
  title={Task-Identity: Training-Free Behavioral Drift Detection for AI Systems},
  author={Barnicle, Shawn},
  year={2024},
  url={https://github.com/Wise314/task-identity}
}

License
MIT License - See LICENSE file for details.

Contact
For commercial licensing inquiries: Open an issue
For technical questions: Open an issue
For research collaboration: Open an issue

Acknowledgments
Discovered while investigating adaptive monitoring methods for physical systems. The key insight - that neural networks maintain embedding similarity during catastrophic behavioral failure - emerged from systematic testing across multiple AI failure modes.
