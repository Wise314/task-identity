#!/usr/bin/env python3
"""
MODEL POISONING DETECTION TEST
Test Task-Identity's ability to detect data poisoning attacks
Simulates an attacker who corrupts training data with wrong labels
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from task_identity import calculate_task_identity
import json
import os
from datetime import datetime

class ModelPoisoningDetection:
    
    def __init__(self, random_seed=42):
        np.random.seed(random_seed)
        self.results = {
            'test': 'model_poisoning_detection',
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'poison_rate': None,
            'task_identity': None,
            'accuracy_clean_model': None,
            'accuracy_poisoned_model': None
        }
    
    def log(self, msg, icon='📊'):
        print(f"{icon} {msg}")
    
    def load_mnist(self):
        """Load and prepare MNIST dataset"""
        self.log("Loading MNIST...", '📥')
        
        mnist = fetch_openml('mnist_784', parser='auto')
        images = np.array(mnist.data[:10000]) / 255.0
        labels = np.array(mnist.target[:10000], dtype=int)
        
        # Train/test split
        train_size = 7000
        train_images = images[:train_size]
        train_labels = labels[:train_size]
        test_images = images[train_size:]
        test_labels = labels[train_size:]
        
        self.log(f"Train: {len(train_images)} samples", '✓')
        self.log(f"Test: {len(test_images)} samples", '✓')
        
        return train_images, train_labels, test_images, test_labels
    
    def poison_training_data(self, train_labels, poison_rate=0.2):
        """
        Poison training data by flipping labels
        
        Attack strategy: Randomly flip poison_rate% of labels to wrong classes
        This simulates an attacker injecting bad data into training set
        """
        self.log(f"Poisoning {poison_rate*100:.0f}% of training data...", '☠️')
        
        poisoned_labels = train_labels.copy()
        n_samples = len(train_labels)
        n_poison = int(n_samples * poison_rate)
        
        # Select random samples to poison
        poison_indices = np.random.choice(n_samples, n_poison, replace=False)
        
        # Flip each poisoned label to a random wrong class
        for idx in poison_indices:
            true_label = train_labels[idx]
            # Pick a random wrong label
            wrong_labels = [l for l in range(10) if l != true_label]
            poisoned_labels[idx] = np.random.choice(wrong_labels)
        
        self.log(f"Poisoned {n_poison} samples (flipped to wrong classes)", '☠️')
        
        return poisoned_labels
    
    def train_model(self, train_images, train_labels, model_name="model"):
        """Train neural network classifier"""
        self.log(f"Training {model_name}...", '🧠')
        
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=20,
            random_state=42,
            verbose=False
        )
        
        clf.fit(train_images, train_labels)
        
        return clf
    
    def evaluate_poisoning_detection(self, clf_clean, clf_poisoned, test_images, test_labels):
        """
        Evaluate Task-Identity on clean vs poisoned model predictions
        """
        self.log("\nEvaluating poisoning detection...", '🔍')
        
        # Get predictions from both models
        preds_clean = clf_clean.predict(test_images)
        preds_poisoned = clf_poisoned.predict(test_images)
        
        # Calculate accuracies
        acc_clean = (preds_clean == test_labels).mean()
        acc_poisoned = (preds_poisoned == test_labels).mean()
        
        self.log(f"\n{'='*70}")
        self.log("ACCURACY COMPARISON", '📊')
        self.log(f"{'='*70}")
        self.log(f"Clean model accuracy: {acc_clean:.3f}", '✓')
        self.log(f"Poisoned model accuracy: {acc_poisoned:.3f}", '☠️')
        self.log(f"Accuracy degradation: {(acc_clean - acc_poisoned):.3f} ({((acc_clean - acc_poisoned)/acc_clean*100):.1f}%)", '📉')
        
        # Calculate Task-Identity: Clean predictions vs Poisoned predictions
        task_identity = calculate_task_identity(
            test_labels, preds_clean,
            test_labels, preds_poisoned,
            labels=range(10)
        )
        
        self.log(f"\n{'='*70}")
        self.log("TASK-IDENTITY ANALYSIS", '💥')
        self.log(f"{'='*70}")
        self.log(f"Task-Identity (clean vs poisoned): {task_identity:.3f}", '🎯')
        self.log(f"Behavioral divergence: {(1 - task_identity):.3f} ({(1 - task_identity)*100:.1f}%)", '⚠️')
        
        # Analyze confusion patterns
        cm_clean = confusion_matrix(test_labels, preds_clean, labels=range(10))
        cm_poisoned = confusion_matrix(test_labels, preds_poisoned, labels=range(10))
        
        self.log(f"\n{'='*70}")
        self.log("CONFUSION PATTERN ANALYSIS", '🔬')
        self.log(f"{'='*70}")
        
        # Find classes with biggest behavioral change
        class_changes = []
        for cls in range(10):
            clean_pattern = cm_clean[cls, :] / (cm_clean[cls, :].sum() + 1e-10)
            poisoned_pattern = cm_poisoned[cls, :] / (cm_poisoned[cls, :].sum() + 1e-10)
            diff = np.abs(clean_pattern - poisoned_pattern).sum()
            class_changes.append((cls, diff))
        
        class_changes.sort(key=lambda x: x[1], reverse=True)
        
        self.log("Classes most affected by poisoning:", '🎯')
        for cls, change in class_changes[:5]:
            self.log(f"  Digit {cls}: {change:.3f} behavioral change", '  ')
        
        # Store results
        self.results['accuracy_clean_model'] = float(acc_clean)
        self.results['accuracy_poisoned_model'] = float(acc_poisoned)
        self.results['accuracy_degradation'] = float(acc_clean - acc_poisoned)
        self.results['task_identity'] = float(task_identity)
        self.results['behavioral_divergence'] = float(1 - task_identity)
        
        return task_identity, acc_clean, acc_poisoned
    
    def run(self, poison_rate=0.2):
        print("\n" + "="*70)
        print("☠️  MODEL POISONING DETECTION TEST")
        print("Testing Task-Identity's ability to detect poisoned training data")
        print("="*70 + "\n")
        
        self.results['poison_rate'] = float(poison_rate)
        
        # Load data
        train_images, train_labels, test_images, test_labels = self.load_mnist()
        
        # Train CLEAN model
        clf_clean = self.train_model(train_images, train_labels, "clean model")
        acc_clean_train = clf_clean.score(train_images, train_labels)
        acc_clean_test = clf_clean.score(test_images, test_labels)
        self.log(f"Clean model - Train: {acc_clean_train:.3f}, Test: {acc_clean_test:.3f}", '✓')
        
        # Create POISONED training data
        poisoned_labels = self.poison_training_data(train_labels, poison_rate)
        
        # Train POISONED model
        clf_poisoned = self.train_model(train_images, poisoned_labels, "poisoned model")
        acc_poisoned_train = clf_poisoned.score(train_images, train_labels)  # Test on CLEAN labels
        acc_poisoned_test = clf_poisoned.score(test_images, test_labels)
        self.log(f"Poisoned model - Train: {acc_poisoned_train:.3f}, Test: {acc_poisoned_test:.3f}", '☠️')
        
        # Evaluate detection
        task_id, acc_clean, acc_poisoned = self.evaluate_poisoning_detection(
            clf_clean, clf_poisoned, test_images, test_labels
        )
        
        # Final summary
        print("\n" + "="*70)
        print("📊 FINAL RESULTS")
        print("="*70)
        
        behavior_change = (1 - task_id) * 100
        acc_degradation = ((acc_clean - acc_poisoned) / acc_clean) * 100
        
        self.log(f"Poison rate: {poison_rate*100:.0f}% of training data", '☠️')
        self.log(f"Accuracy impact: {acc_clean:.3f} → {acc_poisoned:.3f} ({acc_degradation:.1f}% drop)", '📉')
        self.log(f"Task-Identity: {task_id:.3f}", '💥')
        self.log(f"Behavioral divergence: {behavior_change:.1f}%", '🎯')
        
        print()
        if task_id < 0.7:
            print("🎯 SUCCESS: Task-Identity detected MAJOR poisoning attack!")
            print(f"   {behavior_change:.0f}% behavioral divergence indicates compromised model")
            print("   Security application: Detect backdoored/poisoned models")
        elif task_id < 0.85:
            print("✓ Task-Identity detected MODERATE poisoning")
            print(f"   {behavior_change:.0f}% behavioral change suggests data corruption")
        elif task_id < 0.95:
            print("⚠️  Task-Identity detected MINOR poisoning")
            print(f"   {behavior_change:.0f}% behavioral shift detected")
        else:
            print("❌ Poisoning had minimal behavioral impact")
            print("   Model is robust to this poison rate")
        
        # Save results
        os.makedirs('results', exist_ok=True)
        filename = f"results/model_poisoning_{self.results['timestamp']}.json"
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.log(f"\nResults saved: {filename}", '✓')
        print("="*70 + "\n")
        
        return self.results


if __name__ == "__main__":
    detector = ModelPoisoningDetection()
    results = detector.run(poison_rate=0.2)  # Poison 20% of training data
