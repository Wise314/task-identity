#!/usr/bin/env python3
"""
CLASS IMBALANCE DETECTION TEST
Test Task-Identity's ability to detect behavioral changes under extreme class imbalance
Scenario: Train on balanced data, test on heavily imbalanced distribution
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from task_identity import calculate_task_identity
import json
import os
from datetime import datetime

class ClassImbalanceDetection:
    
    def __init__(self, random_seed=42):
        np.random.seed(random_seed)
        self.results = {
            'test': 'class_imbalance_detection',
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
    
    def log(self, msg, icon='📊'):
        print(f"{icon} {msg}")
    
    def load_mnist(self):
        """Load MNIST dataset"""
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
        
        self.log(f"✓ Train: {len(train_images)} samples", '  ')
        self.log(f"✓ Test: {len(test_images)} samples", '  ')
        
        return train_images, train_labels, test_images, test_labels
    
    def create_imbalanced_test_set(self, test_images, test_labels, 
                                   majority_class=0, majority_ratio=0.9):
        """
        Create heavily imbalanced test set
        
        Args:
            majority_class: Which class should dominate
            majority_ratio: What fraction should be majority class
        """
        self.log(f"\n⚖️ Creating EXTREME imbalanced test set...", '⚖️')
        self.log(f"Strategy: {majority_ratio*100:.0f}% class {majority_class}, rest distributed among other classes", '  ')
        
        n_samples = len(test_labels)
        n_majority = int(n_samples * majority_ratio)
        n_minority_total = n_samples - n_majority
        n_per_minority = n_minority_total // 9  # 9 other classes
        
        imbalanced_images = []
        imbalanced_labels = []
        
        # Add majority class samples
        majority_indices = np.where(test_labels == majority_class)[0]
        selected_majority = np.random.choice(majority_indices, 
                                            min(n_majority, len(majority_indices)), 
                                            replace=True)
        imbalanced_images.extend(test_images[selected_majority])
        imbalanced_labels.extend([majority_class] * len(selected_majority))
        
        # Add minority class samples
        for cls in range(10):
            if cls == majority_class:
                continue
            
            cls_indices = np.where(test_labels == cls)[0]
            if len(cls_indices) > 0:
                selected = np.random.choice(cls_indices, 
                                           min(n_per_minority, len(cls_indices)), 
                                           replace=True)
                imbalanced_images.extend(test_images[selected])
                imbalanced_labels.extend([cls] * len(selected))
        
        imbalanced_images = np.array(imbalanced_images)
        imbalanced_labels = np.array(imbalanced_labels)
        
        # Show distribution
        self.log(f"\n📊 Imbalanced test set distribution:", '  ')
        for cls in range(10):
            count = (imbalanced_labels == cls).sum()
            pct = count / len(imbalanced_labels) * 100
            if cls == majority_class:
                self.log(f"  Class {cls}: {count:4d} samples ({pct:5.1f}%) ← MAJORITY", '  ')
            else:
                self.log(f"  Class {cls}: {count:4d} samples ({pct:5.1f}%)", '  ')
        
        return imbalanced_images, imbalanced_labels
    
    def train_model(self, train_images, train_labels):
        """Train model on balanced data"""
        self.log("\n🧠 Training model on BALANCED data...", '🧠')
        
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=20,
            random_state=42,
            verbose=False
        )
        
        clf.fit(train_images, train_labels)
        
        return clf
    
    def evaluate_imbalance_impact(self, clf, test_balanced, labels_balanced, 
                                  test_imbalanced, labels_imbalanced,
                                  majority_class):
        """
        Evaluate how class imbalance affects Task-Identity
        """
        self.log("\n🔍 Evaluating imbalance impact...", '🔍')
        
        # Get predictions on balanced test set
        preds_balanced = clf.predict(test_balanced)
        acc_balanced = (preds_balanced == labels_balanced).mean()
        
        # Get predictions on imbalanced test set
        preds_imbalanced = clf.predict(test_imbalanced)
        acc_imbalanced = (preds_imbalanced == labels_imbalanced).mean()
        
        self.log(f"\n{'='*70}")
        self.log("ACCURACY COMPARISON", '📊')
        self.log(f"{'='*70}")
        self.log(f"Balanced test set: {acc_balanced:.3f}", '✓')
        self.log(f"Imbalanced test set: {acc_imbalanced:.3f}", '⚖️')
        self.log(f"Accuracy change: {(acc_imbalanced - acc_balanced):.3f} ({((acc_imbalanced - acc_balanced)/acc_balanced*100):.1f}%)", '📊')
        
        # Overall Task-Identity
        task_identity_overall = calculate_task_identity(
            labels_balanced, preds_balanced,
            labels_imbalanced, preds_imbalanced,
            labels=range(10)
        )
        
        self.log(f"\n{'='*70}")
        self.log("OVERALL TASK-IDENTITY", '💥')
        self.log(f"{'='*70}")
        self.log(f"Task-Identity (balanced vs imbalanced): {task_identity_overall:.3f}", '🎯')
        self.log(f"Behavioral shift: {(1-task_identity_overall)*100:.1f}%", '📉')
        
        # Per-class analysis
        self.log(f"\n{'='*70}")
        self.log("PER-CLASS ANALYSIS", '🔬')
        self.log(f"{'='*70}")
        
        class_stats = {}
        
        for cls in range(10):
            # Balanced set
            balanced_mask = (labels_balanced == cls)
            if balanced_mask.sum() > 0:
                acc_bal = float((preds_balanced[balanced_mask] == cls).mean())
            else:
                acc_bal = 0.0
            
            # Imbalanced set
            imbalanced_mask = (labels_imbalanced == cls)
            if imbalanced_mask.sum() > 0:
                acc_imb = float((preds_imbalanced[imbalanced_mask] == cls).mean())
                n_imb = int(imbalanced_mask.sum())
            else:
                acc_imb = 0.0
                n_imb = 0
            
            # Calculate per-class Task-Identity if we have samples
            if balanced_mask.sum() > 0 and imbalanced_mask.sum() > 0:
                try:
                    cls_task_id = calculate_task_identity(
                        labels_balanced[balanced_mask], preds_balanced[balanced_mask],
                        labels_imbalanced[imbalanced_mask], preds_imbalanced[imbalanced_mask],
                        labels=range(10)
                    )
                    cls_task_id = float(cls_task_id)
                except:
                    cls_task_id = None
            else:
                cls_task_id = None
            
            class_stats[cls] = {
                'balanced_acc': acc_bal,
                'imbalanced_acc': acc_imb,
                'imbalanced_count': n_imb,
                'task_identity': cls_task_id
            }
            
            # Display
            if cls == majority_class:
                icon = '👑'
                label = 'MAJORITY'
            else:
                icon = '📉'
                label = 'MINORITY'
            
            if cls_task_id is not None:
                self.log(f"{icon} Class {cls} ({label}): Acc {acc_bal:.3f}→{acc_imb:.3f} | Task-ID: {cls_task_id:.3f} | n={n_imb}", '  ')
            else:
                self.log(f"{icon} Class {cls} ({label}): Acc {acc_bal:.3f}→{acc_imb:.3f} | n={n_imb}", '  ')
        
        # Analyze majority vs minority class performance
        self.log(f"\n{'='*70}")
        self.log("MAJORITY VS MINORITY CLASS IMPACT", '🎯')
        self.log(f"{'='*70}")
        
        majority_acc = class_stats[majority_class]['imbalanced_acc']
        minority_accs = [class_stats[c]['imbalanced_acc'] for c in range(10) if c != majority_class]
        avg_minority_acc = float(np.mean([a for a in minority_accs if a > 0]))
        
        self.log(f"Majority class ({majority_class}) accuracy: {majority_acc:.3f}", '👑')
        self.log(f"Average minority class accuracy: {avg_minority_acc:.3f}", '📉')
        self.log(f"Performance gap: {(majority_acc - avg_minority_acc):.3f}", '⚖️')
        
        # Store results (ensure all numpy types converted to Python types)
        self.results['task_identity_overall'] = float(task_identity_overall)
        self.results['behavioral_shift'] = float(1 - task_identity_overall)
        self.results['balanced_accuracy'] = float(acc_balanced)
        self.results['imbalanced_accuracy'] = float(acc_imbalanced)
        self.results['majority_class_acc'] = float(majority_acc)
        self.results['minority_avg_acc'] = float(avg_minority_acc)
        self.results['class_stats'] = class_stats
        
        return task_identity_overall, class_stats
    
    def run(self, majority_class=0, majority_ratio=0.9):
        print("\n" + "="*70)
        print("⚖️  CLASS IMBALANCE DETECTION TEST")
        print("Testing Task-Identity under EXTREME class imbalance")
        print("="*70)
        
        self.results['majority_class'] = int(majority_class)
        self.results['majority_ratio'] = float(majority_ratio)
        
        # Load balanced data
        train_images, train_labels, test_balanced, labels_balanced = self.load_mnist()
        
        # Train model on balanced data
        clf = self.train_model(train_images, train_labels)
        acc_train = clf.score(train_images, train_labels)
        acc_test_balanced = clf.score(test_balanced, labels_balanced)
        self.log(f"✓ Model trained on balanced data", '✓')
        self.log(f"  Train accuracy: {acc_train:.3f}", '  ')
        self.log(f"  Test accuracy (balanced): {acc_test_balanced:.3f}", '  ')
        
        # Create imbalanced test set
        test_imbalanced, labels_imbalanced = self.create_imbalanced_test_set(
            test_balanced, labels_balanced, majority_class, majority_ratio
        )
        
        # Evaluate impact
        task_id, class_stats = self.evaluate_imbalance_impact(
            clf, test_balanced, labels_balanced,
            test_imbalanced, labels_imbalanced,
            majority_class
        )
        
        # Final summary
        print("\n" + "="*70)
        print("📊 FINAL RESULTS")
        print("="*70)
        
        self.log(f"Majority class: {majority_class} ({majority_ratio*100:.0f}% of data)", '👑')
        self.log(f"Overall Task-Identity: {task_id:.3f}", '💥')
        self.log(f"Behavioral shift: {(1-task_id)*100:.1f}%", '📉')
        
        print()
        if task_id < 0.7:
            print("🎯 SUCCESS: Task-Identity detected MAJOR imbalance impact!")
            print(f"   Class imbalance caused {(1-task_id)*100:.0f}% behavioral shift")
            print("   Model behavior significantly different on imbalanced data")
        elif task_id < 0.85:
            print("✓ Task-Identity detected MODERATE imbalance impact")
            print(f"   {(1-task_id)*100:.0f}% behavioral change detected")
        else:
            print("ℹ️  Model relatively robust to class imbalance")
            print(f"   {task_id*100:.0f}% behavioral similarity maintained")
        
        # Save results
        os.makedirs('results', exist_ok=True)
        filename = f"results/06_class_imbalance/class_imbalance_{self.results['timestamp']}.json"
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.log(f"\nResults saved: {filename}", '✓')
        print("="*70 + "\n")
        
        return self.results


if __name__ == "__main__":
    detector = ClassImbalanceDetection()
    # Test with EXTREME imbalance: class 0 as majority (90% of data!)
    results = detector.run(majority_class=0, majority_ratio=0.9)
