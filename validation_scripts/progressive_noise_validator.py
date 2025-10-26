#!/usr/bin/env python3
"""
PROGRESSIVE NOISE VALIDATOR
Test Task-Identity on GRADUAL AI degradation via increasing Gaussian noise
This should create intermediate accuracy values where thresholds can $

Noise levels: 0% → 5% → 10% → 15% → 20% → 25%
Expected: Accuracy gradually declines instead of binary collapse
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import json
import os
from datetime import datetime
import math
from task_identity import calculate_task_identity

class ProgressiveNoiseValidator:
    
    def __init__(self, random_seed=42):
        np.random.seed(random_seed)
        self.results = {
            'test': 'progressive_noise_degradation',
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'noise_levels': [],
            'accuracies': [],
            'task_identities': [],
        }
    
    def log(self, msg, icon='📊'):
        print(f"{icon} {msg}")
    
    def add_gaussian_noise(self, images, noise_level):
        """Add Gaussian noise to images"""
        noise = np.random.normal(0, noise_level, images.shape)
        noisy = images + noise
        return np.clip(noisy, 0, 1)  # Keep in valid range
    
    def load_mnist(self):
        self.log("Loading MNIST...")
        
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
    
    def train_baseline(self, train_images, train_labels):
        self.log("Training baseline model on clean MNIST...", '🧠')
        
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=20,
            random_state=42,
            verbose=False
        )
        
        clf.fit(train_images, train_labels)
        
        return clf
    
    def test_progressive_noise(self, clf, test_images, test_labels):
        self.log("\nTesting progressive noise levels...", '🔬')
        
        noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        
        # Baseline (0% noise)
        preds_baseline = clf.predict(test_images)
        cm_baseline = confusion_matrix(test_labels, preds_baseline, labels=range(10))
        acc_baseline = (preds_baseline == test_labels).mean()
        
        self.log(f"\nBaseline (0% noise): {acc_baseline:.3f}", '✓')
        
        # Test each noise level
        noise_data = []
        
        print("\n" + "="*70)
        print("PROGRESSIVE NOISE DEGRADATION")
        print("="*70)
        print(f"{'Noise':<10} {'Accuracy':<12} {'Task-ID':<12} {'Status':<20}")
        print("-"*70)
        
        for noise_level in noise_levels:
            # Add noise
            noisy_images = self.add_gaussian_noise(test_images, noise_level)
            
            # Test
            preds_noisy = clf.predict(noisy_images)
            cm_noisy = confusion_matrix(test_labels, preds_noisy, labels=range(10))
            acc_noisy = (preds_noisy == test_labels).mean()
            
            # Calculate task-identity
            # Get baseline predictions for comparison
            preds_baseline = clf.predict(test_images)
            task_identity = calculate_task_identity(test_labels, preds_baseline, test_labels, preds_noisy, labels=range(10))
            
            # Status
            if acc_noisy > 0.9:
                status = "✓ Healthy"
            elif acc_noisy > 0.7:
                status = "⚠️ Mild degradation"
            elif acc_noisy > 0.5:
                status = "⚠️⚠️ Moderate degradation"
            else:
                status = "❌ Severe degradation"
            
            print(f"{noise_level:<10.2f} {acc_noisy:<12.3f} {task_identity:<12.3f} {status:<20}")
            
            noise_data.append({
                'noise_level': float(noise_level),
                'accuracy': float(acc_noisy),
                'task_identity': float(task_identity),
                'confusion_matrix': cm_noisy.tolist()
            })
        
        print("="*70)
        
        # Average task-identity across noise levels
        avg_task_identity = np.mean([d['task_identity'] for d in noise_data[1:]])  # Exclude 0% noise
        
        self.log(f"\nAverage Task-Identity: {avg_task_identity:.3f}", '💥')
        
        self.results['noise_levels'] = [d['noise_level'] for d in noise_data]
        self.results['accuracies'] = [d['accuracy'] for d in noise_data]
        self.results['task_identities'] = [d['task_identity'] for d in noise_data]
        self.results['avg_task_identity'] = float(avg_task_identity)
        self.results['baseline_accuracy'] = float(acc_baseline)
        
        return noise_data, acc_baseline, avg_task_identity
    
    def run(self):
        print("\n" + "="*70)
        print("🎯 PROGRESSIVE NOISE VALIDATOR - THE FINAL TEST")
        print("Testing Task-Identity on GRADUAL AI degradation")
        print("="*70 + "\n")
        
        # Load data
        train_images, train_labels, test_images, test_labels = self.load_mnist()
        
        # Train baseline
        clf = self.train_baseline(train_images, train_labels)
        
        # Test progressive noise
        noise_data, baseline_acc, avg_task_id = self.test_progressive_noise(
            clf, test_images, test_labels
        )
        
        # Final summary
        print("\n" + "="*70)
        print("📊 FINAL RESULTS")
        print("="*70)
        
        self.log(f"Average Task-Identity: {avg_task_id:.3f}", '💥')
        
        # Save results
        os.makedirs('results', exist_ok=True)
        filename = f"results/02_progressive_noise/progressive_noise_{self.results['timestamp']}.json"
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.log(f"\nResults saved: {filename}", '✓')
        
        print("="*70 + "\n")
        
        return self.results


if __name__ == "__main__":
    validator = ProgressiveNoiseValidator()
    results = validator.run()
