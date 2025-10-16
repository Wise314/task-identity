"""
Test 3: Domain Shift Detection
Tests Task-Identity's ability to detect cross-domain behavioral differences.
Scenario: Model trained on MNIST (digits) tested on Fashion-MNIST (clothing)
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from task_identity import calculate_task_identity
import json
from datetime import datetime

print("📊 DOMAIN SHIFT TEST")
print("=" * 70)

# Load MNIST
print("📥 Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X_mnist = mnist.data.to_numpy() / 255.0
y_mnist = mnist.target.to_numpy().astype(int)

# Load Fashion-MNIST
print("📥 Loading Fashion-MNIST...")
fashion = fetch_openml('Fashion-MNIST', version=1, parser='auto')
X_fashion = fashion.data.to_numpy() / 255.0
y_fashion = fashion.target.to_numpy().astype(int)

# Use subset for speed
print("📊 Preparing datasets...")
X_mnist_train = X_mnist[:7000]
y_mnist_train = y_mnist[:7000]
X_mnist_test = X_mnist[60000:63000]
y_mnist_test = y_mnist[60000:63000]

X_fashion_test = X_fashion[60000:63000]
y_fashion_test = y_fashion[60000:63000]

print(f"   ✓ MNIST train: {len(X_mnist_train)} samples")
print(f"   ✓ MNIST test: {len(X_mnist_test)} samples")
print(f"   ✓ Fashion-MNIST test: {len(X_fashion_test)} samples")

# Train model on MNIST
print("\n🧠 Training model on MNIST...")
model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=20, random_state=42)
model.fit(X_mnist_train, y_mnist_train)
print("   ✓ Model trained")

# Test on MNIST (same domain)
print("\n🔍 Testing on MNIST (same domain)...")
preds_mnist = model.predict(X_mnist_test)
acc_mnist = accuracy_score(y_mnist_test, preds_mnist)
print(f"   ✓ MNIST accuracy: {acc_mnist:.3f}")

# Test on Fashion-MNIST (different domain)
print("\n🔍 Testing on Fashion-MNIST (different domain)...")
preds_fashion = model.predict(X_fashion_test)
acc_fashion = accuracy_score(y_fashion_test, preds_fashion)
print(f"   ✓ Fashion-MNIST accuracy: {acc_fashion:.3f}")

# Calculate Task-Identity
print("\n🎯 Calculating Task-Identity (cross-domain)...")
task_id = calculate_task_identity(
    y_mnist_test, preds_mnist,
    y_fashion_test, preds_fashion,
    labels=range(10)
)

# Results
print("\n" + "=" * 70)
print("📊 RESULTS")
print("=" * 70)
print(f"✓ MNIST accuracy: {acc_mnist:.3f}")
print(f"✓ Fashion-MNIST accuracy: {acc_fashion:.3f}")
print(f"🎯 Task-Identity (cross-domain): {task_id:.3f}")
print(f"📉 Behavioral divergence: {(1-task_id)*100:.1f}%")
print()

if task_id < 0.2:
    print("✅ RESULT: Severe domain shift detected")
    print("   Models trained on different domains behave fundamentally differently")
    recommendation = "Domain shift successfully detected"
else:
    print("⚠️ WARNING: Expected severe domain shift but got higher similarity")
    recommendation = "Unexpected cross-domain similarity"

print("\n" + "=" * 70)
print("💡 INTERPRETATION")
print("=" * 70)
print("Domain shift test validates that Task-Identity detects when models")
print("are operating on data from a different distribution than training,")
print("even when input format (28x28 grayscale images) is identical.")
print()

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results = {
    "test_name": "domain_shift",
    "test_type": "cross_domain_detection",
    "timestamp": timestamp,
    "mnist_accuracy": float(acc_mnist),
    "fashion_mnist_accuracy": float(acc_fashion),
    "task_identity": float(task_id),
    "behavioral_divergence": float(1 - task_id),
    "interpretation": recommendation,
    "details": {
        "training_domain": "MNIST (handwritten digits)",
        "test_domain_1": "MNIST (same domain)",
        "test_domain_2": "Fashion-MNIST (clothing items)",
        "samples_mnist": len(X_mnist_test),
        "samples_fashion": len(X_fashion_test)
    }
}

filename = f"results/03_domain_shift/domain_shift_{timestamp}.json"
with open(filename, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Results saved: {filename}")
print("=" * 70)
