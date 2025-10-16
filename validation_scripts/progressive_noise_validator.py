#!/usr/bin/env python3
"""
PROGRESSIVE NOISE VALIDATOR
Test Config 2 on GRADUAL AI degradation via increasing Gaussian noise
This should create intermediate accuracy values where thresholds can $

Noise levels: 0% → 5% → 10% → 15% → 20% → 25%
Expected: Accuracy gradually declines instead of binary collapse
"""

# ============================================================================
# NOTE: This script contains EXPERIMENTAL threshold detection code
# ============================================================================
# The core Task-Identity metric (confusion matrix correlation) is production-
# ready and validated. This script also includes experimental threshold 
# tuning methods (v2.0, Config 2) that test various multipliers and 
# autocorrelation heuristics. These experimental sections are clearly marked.
#
# For production use, focus on the core Task-Identity values, not the 
# experimental threshold detection methods.
# ============================================================================


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
            'alpha_results': {}
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
        
        # Calculate autocorrelation from accuracy progression
        accuracies = [d['accuracy'] for d in noise_data]
        
        if len(accuracies) > 2:
            autocorr = np.corrcoef(accuracies[:-1], accuracies[1:])[0, 1]
            if np.isnan(autocorr):
                autocorr = 0.9  # Default for smooth progression
        else:
            autocorr = 0.9
        
        # Average task-identity across noise levels
        avg_task_identity = np.mean([d['task_identity'] for d in noise_data[1:]])  # Exclude 0% noise
        
        # Calculate multipliers
        multiplier = math.sqrt(max(0.0, avg_task_identity)) * max(0.0, abs(autocorr))
        inverted_multiplier = 2 - multiplier
        
        self.log(f"\nAverage Task-Identity: {avg_task_identity:.3f}", '💥')
        self.log(f"Autocorrelation: {autocorr:.3f}", '🧮')
        self.log(f"Multiplier (√I × ρ): {multiplier:.3f}", '🧮')
        self.log(f"Inverted multiplier: {inverted_multiplier:.3f}", '🚀')
        
        self.results['noise_levels'] = [d['noise_level'] for d in noise_data]
        self.results['accuracies'] = [d['accuracy'] for d in noise_data]
        self.results['task_identities'] = [d['task_identity'] for d in noise_data]
        self.results['avg_task_identity'] = float(avg_task_identity)
        self.results['autocorrelation'] = float(autocorr)
        self.results['multiplier'] = float(multiplier)
        self.results['inverted_multiplier'] = float(inverted_multiplier)
        self.results['baseline_accuracy'] = float(acc_baseline)
        
        return noise_data, acc_baseline, avg_task_identity, autocorr, inverted_multiplier
    
    def run_detection_tests(self, noise_data, baseline_acc, inverted_multiplier):
        # ========================================================================
        # EXPERIMENTAL SECTION - NOT PART OF CORE TASK-IDENTITY
        # ========================================================================
        # The code below tests alternative threshold detection methods (v2.0, 
        # Config 2) which include multipliers, autocorrelation, and inverted 
        # multipliers. These are exploratory features from physical system 
        # monitoring and are NOT part of the core Task-Identity metric.
        #
        # Core Task-Identity = Pearson correlation of confusion matrices (simple)
        # This experimental code = Additional threshold tuning heuristics (complex)
        # ========================================================================

        self.log("\n" + "="*70)
        self.log("DETECTION TESTS - v2.0 vs Config 2", '🔬')
        self.log("="*70)
        
        # Ground truth: noise > 15% is degraded (moderate degradation threshold)
        accuracies = [d['accuracy'] for d in noise_data]
        ground_truth = np.array([1 if d['noise_level'] > 0.15 else 0 for d in noise_data])
        
        self.log(f"Ground truth: {int(ground_truth.sum())}/{len(ground_truth)} noise levels marked as degraded (>15%)\n")
        
        # Average degradation rate
        avg_acc = np.mean(accuracies)
        degradation_rate = ((baseline_acc - avg_acc) / baseline_acc) * 100
        
        alphas = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
        
        print(f"{'Alpha':<8} {'v2.0 F1':<12} {'Config 2 F1':<14} {'Δ%':<12} {'Status':<12}")
        print("-"*72)
        
        improvements = []
        
        for alpha in alphas:
            # v2.0: Standard adaptive threshold
            v2_threshold = 1 + alpha * (degradation_rate / 100)
            v2_acc_threshold = baseline_acc / v2_threshold
            
            # Config 2: With inverted multiplier
            c2_threshold = 1 + alpha * (degradation_rate / 100) * inverted_multiplier
            c2_acc_threshold = baseline_acc / c2_threshold
            
            # Detect based on accuracy at each noise level
            v2_detections = np.array([1 if d['accuracy'] < v2_acc_threshold else 0 for d in noise_data])
            c2_detections = np.array([1 if d['accuracy'] < c2_acc_threshold else 0 for d in noise_data])
            
            # Calculate F1
            v2_prec, v2_rec, v2_f1, _ = precision_recall_fscore_support(
                ground_truth, v2_detections, average='binary', zero_division=0
            )
            
            c2_prec, c2_rec, c2_f1, _ = precision_recall_fscore_support(
                ground_truth, c2_detections, average='binary', zero_division=0
            )
            
            improvement = ((c2_f1 - v2_f1) / v2_f1 * 100) if v2_f1 > 0 else 0
            improvements.append(improvement)
            
            status = "🚀 RESCUE!" if improvement > 10 else ("✓ Better" if improvement > 0 else ("≈ Same" if abs(improvement) < 1 else "⚠️ Worse"))
            
            print(f"{alpha:<8.2f} {v2_f1:<12.4f} {c2_f1:<14.4f} {improvement:>+10.1f}% {status:<12}")
            
            self.results['alpha_results'][str(alpha)] = {
                'v2': {
                    'f1': float(v2_f1),
                    'precision': float(v2_prec),
                    'recall': float(v2_rec),
                    'threshold': float(v2_threshold),
                    'acc_threshold': float(v2_acc_threshold),
                    'detections': int(v2_detections.sum())
                },
                'config2': {
                    'f1': float(c2_f1),
                    'precision': float(c2_prec),
                    'recall': float(c2_rec),
                    'threshold': float(c2_threshold),
                    'acc_threshold': float(c2_acc_threshold),
                    'detections': int(c2_detections.sum())
                },
                'improvement_percent': float(improvement)
            }
        
        avg_improvement = np.mean(improvements)
        return avg_improvement
    
    def run(self):
        print("\n" + "="*70)
        print("🎯 PROGRESSIVE NOISE VALIDATOR - THE FINAL TEST")
        print("Testing Config 2 on GRADUAL AI degradation")
        print("="*70 + "\n")
        
        # Load data
        train_images, train_labels, test_images, test_labels = self.load_mnist()
        
        # Train baseline
        clf = self.train_baseline(train_images, train_labels)
        
        # Test progressive noise
        noise_data, baseline_acc, avg_task_id, autocorr, inv_mult = self.test_progressive_noise(
            clf, test_images, test_labels
        )
        
        # Run detection tests
        avg_improvement = self.run_detection_tests(noise_data, baseline_acc, inv_mult)
        
        # Final summary
        print("\n" + "="*70)
        print("📊 FINAL RESULTS")
        print("="*70)
        
        self.log(f"Average Task-Identity: {avg_task_id:.3f}", '💥')
        self.log(f"Inverted Multiplier: {inv_mult:.3f}", '🚀')
        self.log(f"Average F1 Improvement: {avg_improvement:+.2f}%", 
                 '🔥' if avg_improvement > 10 else ('✓' if avg_improvement > 0 else '⚠️'))
        
        if avg_improvement > 10:
            print("\n🎉 CONFIG 2 SHOWS RESCUE FOR GRADUAL AI DEGRADATION!")
            print("Progressive noise creates intermediate states where Config 2 helps!")
        elif avg_improvement > 0:
            print("\n✓ Config 2 shows modest improvement for gradual degradation")
        else:
            print("\n⚠️ Config 2 does not improve detection even for gradual degradation")
            print("This confirms Config 2 is specific to physical systems")
        
        # Save results
        os.makedirs('results', exist_ok=True)
        filename = f"results/progressive_noise_{self.results['timestamp']}.json"
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.log(f"\nResults saved: {filename}", '✓')
        
        print("="*70 + "\n")
        
        return self.results


if __name__ == "__main__":
    validator = ProgressiveNoiseValidator()
    results = validator.run()
