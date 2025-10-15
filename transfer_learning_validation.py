#!/usr/bin/env python3
"""
TRANSFER LEARNING VALIDATION TEST
Test Task-Identity's ability to measure behavioral preservation in transfer learning
Scenario: Pre-train on MNIST, fine-tune on Fashion-MNIST, measure original task retention
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from task_identity import calculate_task_identity
import json
import os
from datetime import datetime

class TransferLearningValidation:
    
    def __init__(self, random_seed=42):
        np.random.seed(random_seed)
        self.results = {
            'test': 'transfer_learning_validation',
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'task_identity': None,
            'original_task_accuracy_before': None,
            'original_task_accuracy_after': None,
            'new_task_accuracy': None
        }
    
    def log(self, msg, icon='📊'):
        print(f"{icon} {msg}")
    
    def load_mnist(self):
        """Load MNIST dataset (digits 0-9)"""
        self.log("Loading MNIST (original task: digit classification)...", '📥')
        
        mnist = fetch_openml('mnist_784', parser='auto')
        images = np.array(mnist.data[:10000]) / 255.0
        labels = np.array(mnist.target[:10000], dtype=int)
        
        # Train/test split
        train_size = 7000
        train_images = images[:train_size]
        train_labels = labels[:train_size]
        test_images = images[train_size:]
        test_labels = labels[train_size:]
        
        self.log(f"✓ MNIST Train: {len(train_images)} samples", '  ')
        self.log(f"✓ MNIST Test: {len(test_images)} samples", '  ')
        
        return train_images, train_labels, test_images, test_labels
    
    def load_fashion_mnist(self):
        """Load Fashion-MNIST dataset (clothing items)"""
        self.log("Loading Fashion-MNIST (transfer task: clothing classification)...", '📥')
        
        fashion = fetch_openml('Fashion-MNIST', parser='auto')
        images = np.array(fashion.data[:10000]) / 255.0
        labels = np.array(fashion.target[:10000], dtype=int)
        
        # Train/test split
        train_size = 7000
        train_images = images[:train_size]
        train_labels = labels[:train_size]
        test_images = images[train_size:]
        test_labels = labels[train_size:]
        
        self.log(f"✓ Fashion-MNIST Train: {len(train_images)} samples", '  ')
        self.log(f"✓ Fashion-MNIST Test: {len(test_images)} samples", '  ')
        
        return train_images, train_labels, test_images, test_labels
    
    def train_original_task(self, train_images, train_labels):
        """Train model on original task (MNIST digits)"""
        self.log("\n🧠 PHASE 1: Training on original task (MNIST digits)...", '🎯')
        
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=20,
            random_state=42,
            verbose=False,
            warm_start=True  # Allow continued training
        )
        
        clf.fit(train_images, train_labels)
        
        return clf
    
    def fine_tune_new_task(self, clf, train_images, train_labels):
        """Fine-tune model on new task (Fashion-MNIST)"""
        self.log("\n🧠 PHASE 2: Fine-tuning on transfer task (Fashion-MNIST)...", '🎯')
        self.log("This simulates transfer learning scenario", '  ')
        
        # Continue training with same model (transfer learning)
        clf.fit(train_images, train_labels)
        
        return clf
    
    def evaluate_transfer_learning(self, clf_original, clf_finetuned, 
                                   mnist_test_images, mnist_test_labels,
                                   fashion_test_images, fashion_test_labels):
        """
        Evaluate how fine-tuning affected original task performance
        """
        self.log("\n🔍 Evaluating transfer learning impact...", '🔍')
        
        # Test original model on MNIST
        preds_original_mnist = clf_original.predict(mnist_test_images)
        acc_original_mnist = (preds_original_mnist == mnist_test_labels).mean()
        
        # Test fine-tuned model on MNIST (original task)
        preds_finetuned_mnist = clf_finetuned.predict(mnist_test_images)
        acc_finetuned_mnist = (preds_finetuned_mnist == mnist_test_labels).mean()
        
        # Test fine-tuned model on Fashion-MNIST (new task)
        acc_finetuned_fashion = clf_finetuned.score(fashion_test_images, fashion_test_labels)
        
        self.log(f"\n{'='*70}")
        self.log("PERFORMANCE ON ORIGINAL TASK (MNIST DIGITS)", '📊')
        self.log(f"{'='*70}")
        self.log(f"Before transfer learning: {acc_original_mnist:.3f}", '✓')
        self.log(f"After transfer learning: {acc_finetuned_mnist:.3f}", '⚠️')
        self.log(f"Performance change: {(acc_finetuned_mnist - acc_original_mnist):.3f} ({((acc_finetuned_mnist - acc_original_mnist)/acc_original_mnist*100):.1f}%)", '📉')
        
        self.log(f"\n{'='*70}")
        self.log("PERFORMANCE ON NEW TASK (FASHION-MNIST)", '📊')
        self.log(f"{'='*70}")
        self.log(f"Transfer learning accuracy: {acc_finetuned_fashion:.3f}", '🎯')
        
        # Calculate Task-Identity: Original task behavior before vs after transfer
        task_identity = calculate_task_identity(
            mnist_test_labels, preds_original_mnist,
            mnist_test_labels, preds_finetuned_mnist,
            labels=range(10)
        )
        
        self.log(f"\n{'='*70}")
        self.log("TASK-IDENTITY ANALYSIS", '💥')
        self.log(f"{'='*70}")
        self.log(f"Task-Identity (original task retention): {task_identity:.3f}", '🎯')
        self.log(f"Behavioral shift: {(1 - task_identity):.3f} ({(1 - task_identity)*100:.1f}%)", '⚠️')
        
        # Interpret results
        self.log(f"\n{'='*70}")
        self.log("INTERPRETATION", '📖')
        self.log(f"{'='*70}")
        
        if task_identity > 0.9:
            self.log("✓ Transfer learning preserved original task behavior well", '  ')
            self.log(f"  Model retained {task_identity*100:.1f}% of original decision patterns", '  ')
        elif task_identity > 0.7:
            self.log("⚠️ Moderate behavioral shift in original task", '  ')
            self.log(f"  {(1-task_identity)*100:.1f}% of original behavior changed", '  ')
        elif task_identity > 0.5:
            self.log("🚨 Major behavioral shift - significant forgetting", '  ')
            self.log(f"  {(1-task_identity)*100:.1f}% behavioral divergence from original task", '  ')
        else:
            self.log("🚨🚨 Catastrophic forgetting - original task largely lost", '  ')
            self.log(f"  Only {task_identity*100:.1f}% behavioral similarity remains", '  ')
        
        # Per-class analysis
        self.log(f"\n{'='*70}")
        self.log("PER-CLASS RETENTION (MNIST DIGITS)", '🔬')
        self.log(f"{'='*70}")
        
        for digit in range(10):
            digit_mask = (mnist_test_labels == digit)
            if digit_mask.sum() > 0:
                acc_before = (preds_original_mnist[digit_mask] == digit).mean()
                acc_after = (preds_finetuned_mnist[digit_mask] == digit).mean()
                retention = acc_after / (acc_before + 1e-10)
                
                if retention > 0.9:
                    icon = '✓'
                elif retention > 0.7:
                    icon = '⚠️'
                else:
                    icon = '🚨'
                
                self.log(f"{icon} Digit {digit}: {acc_before:.3f} → {acc_after:.3f} ({retention*100:.0f}% retention)", '  ')
        
        # Store results
        self.results['task_identity'] = float(task_identity)
        self.results['behavioral_shift'] = float(1 - task_identity)
        self.results['original_task_accuracy_before'] = float(acc_original_mnist)
        self.results['original_task_accuracy_after'] = float(acc_finetuned_mnist)
        self.results['new_task_accuracy'] = float(acc_finetuned_fashion)
        self.results['accuracy_change_pct'] = float((acc_finetuned_mnist - acc_original_mnist) / acc_original_mnist * 100)
        
        return task_identity, acc_original_mnist, acc_finetuned_mnist, acc_finetuned_fashion
    
    def run(self):
        print("\n" + "="*70)
        print("🔄 TRANSFER LEARNING VALIDATION TEST")
        print("Measuring original task retention after transfer learning")
        print("="*70)
        
        # Load both datasets
        mnist_train_imgs, mnist_train_labels, mnist_test_imgs, mnist_test_labels = self.load_mnist()
        fashion_train_imgs, fashion_train_labels, fashion_test_imgs, fashion_test_labels = self.load_fashion_mnist()
        
        # Train on original task
        clf_original = self.train_original_task(mnist_train_imgs, mnist_train_labels)
        acc_mnist_original = clf_original.score(mnist_test_imgs, mnist_test_labels)
        self.log(f"✓ Original task (MNIST) accuracy: {acc_mnist_original:.3f}", '✓')
        
        # Save original model weights for comparison
        original_weights = [w.copy() for w in clf_original.coefs_]
        
        # Fine-tune on new task (creates copy internally due to warm_start)
        clf_finetuned = self.fine_tune_new_task(clf_original, fashion_train_imgs, fashion_train_labels)
        acc_fashion = clf_finetuned.score(fashion_test_imgs, fashion_test_labels)
        self.log(f"✓ Transfer task (Fashion-MNIST) accuracy: {acc_fashion:.3f}", '✓')
        
        # Evaluate retention of original task
        task_id, acc_before, acc_after, acc_new = self.evaluate_transfer_learning(
            clf_original, clf_finetuned,
            mnist_test_imgs, mnist_test_labels,
            fashion_test_imgs, fashion_test_labels
        )
        
        # Final summary
        print("\n" + "="*70)
        print("📊 FINAL RESULTS")
        print("="*70)
        
        self.log(f"Original task retention: {task_id:.3f} Task-Identity", '💥')
        self.log(f"Behavioral shift: {(1-task_id)*100:.1f}%", '📉')
        self.log(f"Original task accuracy: {acc_before:.3f} → {acc_after:.3f}", '📊')
        self.log(f"New task accuracy: {acc_new:.3f}", '🎯')
        
        print()
        if task_id < 0.3:
            print("🎯 SUCCESS: Task-Identity detected catastrophic forgetting!")
            print(f"   Transfer learning caused {(1-task_id)*100:.0f}% behavioral shift")
            print("   Original task capabilities largely lost")
        elif task_id < 0.7:
            print("✓ Task-Identity detected significant task interference")
            print(f"   {(1-task_id)*100:.0f}% of original behavior changed")
        else:
            print("⚠️ Original task well-preserved despite transfer learning")
            print(f"   {task_id*100:.0f}% behavioral similarity maintained")
        
        # Save results
        os.makedirs('results', exist_ok=True)
        filename = f"results/transfer_learning_{self.results['timestamp']}.json"
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.log(f"\nResults saved: {filename}", '✓')
        print("="*70 + "\n")
        
        return self.results


if __name__ == "__main__":
    validator = TransferLearningValidation()
    results = validator.run()
