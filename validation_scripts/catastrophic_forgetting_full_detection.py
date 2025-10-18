#!/usr/bin/env python3
"""
CATASTROPHIC FORGETTING - FULL DETECTION TEST
Complete validator with task-identity and F1 comparison
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

class CatastrophicForgettingFullDetection:
    
    def __init__(self, random_seed=42):
        np.random.seed(random_seed)
        self.results = {
            'test': 'catastrophic_forgetting_full_detection',
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'task_identity': None,
            'autocorrelation': None,
            'alpha_results': {}
        }
    
    def log(self, msg, icon='📊'):
        print(f"{icon} {msg}")
    

    def calculate_embedding_identity_models(self, clf_before, clf_after, images):
        """Compare embedding spaces of two different models on same images"""
        W1_before = clf_before.coefs_[0]
        b1_before = clf_before.intercepts_[0]
        
        W1_after = clf_after.coefs_[0]
        b1_after = clf_after.intercepts_[0]
        
        # Get hidden activations from BOTH models
        hidden_before = np.maximum(0, images @ W1_before + b1_before)
        hidden_after = np.maximum(0, images @ W1_after + b1_after)
        
        # Compare mean activation patterns
        mean_before = hidden_before.mean(axis=0)
        mean_after = hidden_after.mean(axis=0)
        
        if mean_before.std() == 0 or mean_after.std() == 0:
            return 1.0
        
        correlation = np.corrcoef(mean_before, mean_after)[0, 1]
        return max(0.0, correlation) if not np.isnan(correlation) else 1.0

    def load_and_split_mnist(self):
        self.log("Loading MNIST...")
        
        mnist = fetch_openml('mnist_784', parser='auto')
        images = np.array(mnist.data[:60000]) / 255.0
        labels = np.array(mnist.target[:60000], dtype=int)
        
        # Phase 1: digits 0-4
        phase1_mask = labels <= 4
        phase1_images = images[phase1_mask]
        phase1_labels = labels[phase1_mask]
        
        # Phase 2: digits 5-9
        phase2_mask = labels >= 5
        phase2_images = images[phase2_mask]
        phase2_labels = labels[phase2_mask]  # Keep as 5-9 to create label space mismatch
        
        self.log(f"Phase 1 (0-4): {len(phase1_images)} samples", '✓')
        self.log(f"Phase 2 (5-9): {len(phase2_images)} samples", '✓')
        
        return phase1_images, phase1_labels, phase2_images, phase2_labels
    
    def train_phase1(self, phase1_images, phase1_labels):
        self.log("Training Phase 1 (digits 0-4)...", '🧠')
        
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=20,
            random_state=42,
            verbose=False,
            warm_start=False
        )
        
        # Train on Phase 1
        train_size = int(len(phase1_images) * 0.8)
        clf.fit(phase1_images[:train_size], phase1_labels[:train_size])
        
        # Test set for Phase 1
        test_images = phase1_images[train_size:]
        test_labels = phase1_labels[train_size:]
        
        # Get confusion matrix BEFORE fine-tuning
        preds_before = clf.predict(test_images)
        cm_before = confusion_matrix(test_labels, preds_before, labels=range(5))
        acc_before = (preds_before == test_labels).mean()
        
        self.log(f"Phase 1 accuracy (BEFORE fine-tune): {acc_before:.3f}", '✓')
        
        # Save the initial weights for catastrophic forgetting simulation
        initial_weights = [layer.copy() for layer in clf.coefs_]
        initial_biases = [layer.copy() for layer in clf.intercepts_]
        
        return clf, test_images, test_labels, cm_before, acc_before, initial_weights, initial_biases
    
    def catastrophic_fine_tune(self, clf, phase2_images, phase2_labels, initial_weights, initial_biases):
        self.log("Simulating catastrophic forgetting via retraining...", '🔥')
        
        # Create new classifier with same architecture
        clf_retrained = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=20,
            random_state=42,
            verbose=False,
            warm_start=True
        )
        
        # Initialize with Phase 1 weights
        clf_retrained.fit(phase2_images[:100], phase2_labels[:100])  # Quick init
        clf_retrained.coefs_ = [w.copy() for w in initial_weights]
        clf_retrained.intercepts_ = [b.copy() for b in initial_biases]
        
        # Now train heavily on Phase 2, causing catastrophic forgetting
        clf_retrained.fit(phase2_images[:15000], phase2_labels[:15000])
        
        phase2_acc = clf_retrained.score(phase2_images[15000:20000], phase2_labels[15000:20000])
        self.log(f"Phase 2 accuracy (after forgetting): {phase2_acc:.3f}", '✓')
        
        return clf_retrained
    
    def test_forgetting(self, clf_before, clf_after, test_images, test_labels, cm_before, acc_before):
        self.log("Testing catastrophic forgetting...", '💥')
        
        # Test on Phase 1 AFTER catastrophic forgetting
        preds_after = clf_after.predict(test_images)
        cm_after = confusion_matrix(test_labels, preds_after, labels=range(5))
        acc_after = (preds_after == test_labels).mean()
        
        self.log(f"Phase 1 accuracy (AFTER forgetting): {acc_after:.3f}", '⚠️')
        self.log(f"Forgetting: {((acc_before - acc_after) / acc_before * 100):.1f}%", '🔥')
        
        # Calculate task-identity
        # Get predictions from before model for comparison
        preds_before = clf_before.predict(test_images)
        task_identity = calculate_task_identity(test_labels, preds_before, test_labels, preds_after, labels=range(5))
        self.log(f"Task-Identity: {task_identity:.3f}", '💥')
        # Calculate embedding identity (structure similarity)
        embedding_identity = self.calculate_embedding_identity_models(clf_before, clf_after, test_images)
        self.log(f"Embedding Identity: {embedding_identity:.3f}", '🧠')

        
        # Per-class accuracies for autocorrelation and detection
        class_accs = []
        self.log("\nPer-Class Accuracies (Phase 1 after forgetting):")
        for digit in range(5):
            mask = test_labels == digit
            if mask.sum() > 0:
                acc = (preds_after[mask] == test_labels[mask]).mean()
                class_accs.append(acc)
                self.log(f"  Digit {digit}: {acc:.3f}", '🧮')
        
        # Autocorrelation
        if len(class_accs) > 2:
            autocorr = np.corrcoef(class_accs[:-1], class_accs[1:])[0, 1]
            if np.isnan(autocorr):
                autocorr = 0.0
        else:
            autocorr = 0.0
        
        # Calculate multipliers
        multiplier = math.sqrt(max(0.0, task_identity)) * max(0.0, abs(autocorr))
        inverted_multiplier = 2 - multiplier
        
        self.log(f"Autocorrelation: {autocorr:.3f}", '🧮')
        self.log(f"Multiplier (√I × ρ): {multiplier:.3f}", '🧮')
        self.log(f"Inverted multiplier: {inverted_multiplier:.3f}", '🚀')
        
        self.results['task_identity'] = float(task_identity)
        self.results['embedding_identity'] = float(embedding_identity)
        self.results['autocorrelation'] = float(autocorr)
        self.results['multiplier'] = float(multiplier)
        self.results['inverted_multiplier'] = float(inverted_multiplier)
        self.results['baseline_accuracy'] = float(acc_before)
        self.results['shifted_accuracy'] = float(acc_after)
        self.results['class_accuracies'] = [float(a) for a in class_accs]
        
        return task_identity, autocorr, inverted_multiplier, class_accs, acc_before
    
    def run_detection_tests(self, baseline_acc, class_accs, inverted_multiplier):
        self.log("\n" + "="*70)
        self.log("DETECTION TESTS - v2.0 vs Config 2", '🔬')
        self.log("="*70)
        
        # Ground truth: ALL 5 classes are degraded (accuracy dropped significantly)
        ground_truth = np.ones(len(class_accs))
        
        avg_acc = np.mean(class_accs)
        degradation_rate = ((baseline_acc - avg_acc) / baseline_acc) * 100
        
        self.log(f"Degradation rate: {degradation_rate:.1f}%")
        self.log(f"Ground truth: {int(ground_truth.sum())}/{len(ground_truth)} classes degraded\n")
        
        alphas = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0]
        
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
            
            # Detect based on per-class accuracy
            v2_detections = np.array([1 if acc < v2_acc_threshold else 0 for acc in class_accs])
            c2_detections = np.array([1 if acc < c2_acc_threshold else 0 for acc in class_accs])
            
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
                    'threshold': float(v2_threshold),
                    'acc_threshold': float(v2_acc_threshold),
                },
                'config2': {
                    'f1': float(c2_f1),
                    'threshold': float(c2_threshold),
                    'acc_threshold': float(c2_acc_threshold),
                },
                'improvement_percent': float(improvement)
            }
        
        avg_improvement = np.mean(improvements)
        return avg_improvement
    
    def run(self):
        print("\n" + "="*70)
        print("💥 CATASTROPHIC FORGETTING - FULL DETECTION TEST")
        print("Task-identity should be ~0.0, Config 2 should show rescue")
        print("="*70 + "\n")
        
        # Load data
        phase1_images, phase1_labels, phase2_images, phase2_labels = self.load_and_split_mnist()
        
        # Train Phase 1
        clf, test_images, test_labels, cm_before, acc_before, init_w, init_b = self.train_phase1(
            phase1_images, phase1_labels
        )
        
        # Catastrophic forgetting simulation
        clf_forgotten = self.catastrophic_fine_tune(clf, phase2_images, phase2_labels, init_w, init_b)
        
        # Test forgetting and calculate task-identity
        task_id, autocorr, inv_mult, class_accs, baseline_acc = self.test_forgetting(
            clf, clf_forgotten, test_images, test_labels, cm_before, acc_before
        )
        
        # Run detection tests
        avg_improvement = self.run_detection_tests(baseline_acc, class_accs, inv_mult)
        
        # Final summary
        print("\n" + "="*70)
        print("📊 FINAL RESULTS")
        print("="*70)
        
        self.log(f"Task-Identity: {task_id:.3f}", '💥')
        self.log(f"Inverted Multiplier: {inv_mult:.3f}", '🚀')
        self.log(f"Average F1 Improvement: {avg_improvement:+.2f}%", 
                 '🔥' if avg_improvement > 10 else ('✓' if avg_improvement > 0 else '⚠️'))
        
        if avg_improvement > 10:
            print("\n💥 CONFIG 2 SHOWS DRAMATIC RESCUE FOR CATASTROPHIC FORGETTING!")
        elif avg_improvement > 0:
            print("\n✓ Config 2 shows modest improvement")
        else:
            print("\n⚠️ Config 2 does not improve detection for catastrophic forgetting")
        
        # Save results
        os.makedirs('results', exist_ok=True)
        filename = f"results/01_catastrophic_forgetting/catastrophic_forgetting_full_{self.results['timestamp']}.json"
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.log(f"Results saved: {filename}", '✓')
        
        print("="*70 + "\n")
        
        return self.results


if __name__ == "__main__":
    validator = CatastrophicForgettingFullDetection()
    results = validator.run()
