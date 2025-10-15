#!/usr/bin/env python3
"""
TARGETED POISONING DETECTION TEST
Test Task-Identity's ability to detect targeted class poisoning attacks
Simulates sophisticated attacker who targets specific classes
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from task_identity import calculate_task_identity
import json
import os
from datetime import datetime

class TargetedPoisoningDetection:
    
    def __init__(self, random_seed=42):
        np.random.seed(random_seed)
        self.results = {
            'test': 'targeted_poisoning_detection',
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'target_classes': None,
            'poison_rate': None,
            'task_identity': None
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
    
    def targeted_poison(self, train_labels, target_mappings, poison_rate=0.5):
        """
        Targeted poisoning: Flip specific classes to specific wrong classes
        
        Args:
            target_mappings: dict like {5: 3, 8: 3} means "poison 5s to be 3s, 8s to be 3s"
            poison_rate: fraction of target class samples to poison (0.0 to 1.0)
        """
        self.log(f"\n🎯 TARGETED POISONING ATTACK", '☠️')
        self.log(f"Strategy: Poison {poison_rate*100:.0f}% of specific classes", '☠️')
        
        poisoned_labels = train_labels.copy()
        poison_stats = {}
        
        for source_class, target_class in target_mappings.items():
            # Find all samples of source class
            source_indices = np.where(train_labels == source_class)[0]
            n_source = len(source_indices)
            
            # Select poison_rate% to poison
            n_to_poison = int(n_source * poison_rate)
            poison_indices = np.random.choice(source_indices, n_to_poison, replace=False)
            
            # Flip labels
            poisoned_labels[poison_indices] = target_class
            
            poison_stats[source_class] = {
                'total': n_source,
                'poisoned': n_to_poison,
                'target': target_class
            }
            
            self.log(f"  Class {source_class} → {target_class}: Poisoned {n_to_poison}/{n_source} samples ({n_to_poison/n_source*100:.0f}%)", '🎯')
        
        self.results['poison_stats'] = poison_stats
        
        return poisoned_labels
    
    def train_model(self, train_images, train_labels, model_name="model"):
        """Train neural network classifier"""
        self.log(f"\n🧠 Training {model_name}...", '🧠')
        
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=20,
            random_state=42,
            verbose=False
        )
        
        clf.fit(train_images, train_labels)
        
        return clf
    
    def analyze_per_class_identity(self, test_labels, preds_clean, preds_poisoned):
        """
        Calculate Task-Identity for each class individually
        """
        self.log(f"\n{'='*70}")
        self.log("PER-CLASS TASK-IDENTITY ANALYSIS", '🔬')
        self.log(f"{'='*70}")
        
        class_identities = {}
        
        for cls in range(10):
            # Get samples of this class
            cls_mask = (test_labels == cls)
            
            if cls_mask.sum() == 0:
                continue
            
            # Get predictions for this class
            cls_true = test_labels[cls_mask]
            cls_preds_clean = preds_clean[cls_mask]
            cls_preds_poisoned = preds_poisoned[cls_mask]
            
            # Calculate Task-Identity for this class only
            try:
                cls_task_id = calculate_task_identity(
                    cls_true, cls_preds_clean,
                    cls_true, cls_preds_poisoned,
                    labels=range(10)
                )
                class_identities[cls] = cls_task_id
                
                # Color code based on drift
                if cls_task_id < 0.7:
                    icon = '🚨'
                    status = "SEVERE DRIFT"
                elif cls_task_id < 0.85:
                    icon = '⚠️'
                    status = "MAJOR DRIFT"
                elif cls_task_id < 0.95:
                    icon = '⚠️'
                    status = "MODERATE DRIFT"
                else:
                    icon = '✓'
                    status = "STABLE"
                
                self.log(f"{icon} Class {cls}: Task-Identity = {cls_task_id:.3f} ({status})", '  ')
                
            except Exception as e:
                self.log(f"  Class {cls}: Unable to calculate (insufficient data)", '  ')
        
        return class_identities
    
    def evaluate_detection(self, clf_clean, clf_poisoned, test_images, test_labels, target_mappings):
        """
        Evaluate Task-Identity on clean vs poisoned model predictions
        """
        self.log("\n🔍 Evaluating poisoning detection...", '🔍')
        
        # Get predictions from both models
        preds_clean = clf_clean.predict(test_images)
        preds_poisoned = clf_poisoned.predict(test_images)
        
        # Calculate accuracies
        acc_clean = (preds_clean == test_labels).mean()
        acc_poisoned = (preds_poisoned == test_labels).mean()
        
        self.log(f"\n{'='*70}")
        self.log("OVERALL ACCURACY COMPARISON", '📊')
        self.log(f"{'='*70}")
        self.log(f"Clean model accuracy: {acc_clean:.3f}", '✓')
        self.log(f"Poisoned model accuracy: {acc_poisoned:.3f}", '☠️')
        self.log(f"Accuracy degradation: {(acc_clean - acc_poisoned):.3f} ({((acc_clean - acc_poisoned)/acc_clean*100):.1f}%)", '📉')
        
        # Overall Task-Identity
        task_identity_overall = calculate_task_identity(
            test_labels, preds_clean,
            test_labels, preds_poisoned,
            labels=range(10)
        )
        
        self.log(f"\n{'='*70}")
        self.log("OVERALL TASK-IDENTITY", '💥')
        self.log(f"{'='*70}")
        self.log(f"Task-Identity (clean vs poisoned): {task_identity_overall:.3f}", '🎯')
        self.log(f"Behavioral divergence: {(1 - task_identity_overall):.3f} ({(1 - task_identity_overall)*100:.1f}%)", '⚠️')
        
        # Per-class analysis
        class_identities = self.analyze_per_class_identity(test_labels, preds_clean, preds_poisoned)
        
        # Analyze attack success
        self.log(f"\n{'='*70}")
        self.log("ATTACK IMPACT ANALYSIS", '🎯')
        self.log(f"{'='*70}")
        
        for source_class, target_class in target_mappings.items():
            # Check how many source_class samples now predict as target_class
            source_mask = (test_labels == source_class)
            if source_mask.sum() > 0:
                clean_correct = (preds_clean[source_mask] == source_class).sum()
                poisoned_correct = (preds_poisoned[source_mask] == source_class).sum()
                poisoned_as_target = (preds_poisoned[source_mask] == target_class).sum()
                
                self.log(f"Class {source_class} samples:", '  ')
                self.log(f"  Clean model: {clean_correct}/{source_mask.sum()} correct ({clean_correct/source_mask.sum()*100:.0f}%)", '    ')
                self.log(f"  Poisoned model: {poisoned_correct}/{source_mask.sum()} correct ({poisoned_correct/source_mask.sum()*100:.0f}%)", '    ')
                self.log(f"  Poisoned model: {poisoned_as_target}/{source_mask.sum()} misclassified as {target_class} ({poisoned_as_target/source_mask.sum()*100:.0f}%)", '    ')
        
        # Store results
        self.results['accuracy_clean_model'] = float(acc_clean)
        self.results['accuracy_poisoned_model'] = float(acc_poisoned)
        self.results['task_identity_overall'] = float(task_identity_overall)
        self.results['behavioral_divergence'] = float(1 - task_identity_overall)
        self.results['class_identities'] = {int(k): float(v) for k, v in class_identities.items()}
        
        return task_identity_overall, class_identities, acc_clean, acc_poisoned
    
    def run(self, target_mappings={5: 3, 8: 3}, poison_rate=0.6):
        """
        Run targeted poisoning attack
        
        Args:
            target_mappings: dict of source_class: target_class for poisoning
            poison_rate: fraction of each source class to poison
        """
        print("\n" + "="*70)
        print("🎯 TARGETED POISONING DETECTION TEST")
        print("Sophisticated attacker targets specific vulnerable classes")
        print("="*70)
        
        self.results['target_mappings'] = target_mappings
        self.results['poison_rate'] = float(poison_rate)
        
        # Load data
        train_images, train_labels, test_images, test_labels = self.load_mnist()
        
        # Train CLEAN model
        clf_clean = self.train_model(train_images, train_labels, "CLEAN MODEL")
        acc_clean_test = clf_clean.score(test_images, test_labels)
        self.log(f"✓ Clean model test accuracy: {acc_clean_test:.3f}", '✓')
        
        # Create TARGETED POISONED training data
        poisoned_labels = self.targeted_poison(train_labels, target_mappings, poison_rate)
        
        # Train POISONED model
        clf_poisoned = self.train_model(train_images, poisoned_labels, "POISONED MODEL")
        acc_poisoned_test = clf_poisoned.score(test_images, test_labels)
        self.log(f"☠️ Poisoned model test accuracy: {acc_poisoned_test:.3f}", '☠️')
        
        # Evaluate detection
        task_id, class_ids, acc_clean, acc_poisoned = self.evaluate_detection(
            clf_clean, clf_poisoned, test_images, test_labels, target_mappings
        )
        
        # Final summary
        print("\n" + "="*70)
        print("📊 FINAL RESULTS")
        print("="*70)
        
        behavior_change = (1 - task_id) * 100
        
        self.log(f"Attack: Poisoned classes {list(target_mappings.keys())} → {list(set(target_mappings.values()))}", '🎯')
        self.log(f"Poison rate: {poison_rate*100:.0f}% of target classes", '☠️')
        self.log(f"Overall Task-Identity: {task_id:.3f}", '💥')
        self.log(f"Overall behavioral divergence: {behavior_change:.1f}%", '📉')
        
        # Find most affected class
        if class_ids:
            min_class = min(class_ids.items(), key=lambda x: x[1])
            self.log(f"Most affected class: {min_class[0]} (Task-Identity: {min_class[1]:.3f})", '🚨')
        
        print()
        if task_id < 0.7:
            print("🎯 SUCCESS: Task-Identity detected MAJOR targeted poisoning!")
            print(f"   {behavior_change:.0f}% overall behavioral divergence")
            print("   Specific classes show severe compromise")
        elif task_id < 0.85:
            print("✓ Task-Identity detected MODERATE targeted poisoning")
            print(f"   {behavior_change:.0f}% behavioral shift in targeted classes")
        elif task_id < 0.95:
            print("⚠️ Task-Identity detected MINOR poisoning impact")
        else:
            print("ℹ️  Minimal overall impact - but check per-class analysis")
            print("   Targeted attacks may only affect specific classes")
        
        # Save results
        os.makedirs('results', exist_ok=True)
        filename = f"results/targeted_poisoning_{self.results['timestamp']}.json"
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.log(f"\nResults saved: {filename}", '✓')
        print("="*70 + "\n")
        
        return self.results


if __name__ == "__main__":
    detector = TargetedPoisoningDetection()
    # Attack strategy: Poison 60% of "5"s and "8"s to be labeled as "3"
    # This should show clear drift in those specific classes
    results = detector.run(
        target_mappings={5: 3, 8: 3},  # Poison 5→3 and 8→3
        poison_rate=0.6  # 60% of each target class
    )
