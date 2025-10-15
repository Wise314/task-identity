#!/usr/bin/env python3
"""
ADVERSARIAL ATTACK DETECTION TEST - V3: SMART AGGRESSIVE ATTACK
Test Task-Identity's ability to detect adversarial attacks
Uses intelligent iterative attack with adaptive strategies
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from task_identity import calculate_task_identity
import json
import os
from datetime import datetime

class AdversarialDetectionTest:
    
    def __init__(self, random_seed=42):
        np.random.seed(random_seed)
        self.results = {
            'test': 'adversarial_attack_detection_v3',
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'task_identity_clean': None,
            'task_identity_adversarial': None,
            'accuracy_clean': None,
            'accuracy_adversarial': None
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
    
    def train_model(self, train_images, train_labels):
        """Train neural network classifier"""
        self.log("Training baseline model...", '🧠')
        
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=20,
            random_state=42,
            verbose=False
        )
        
        clf.fit(train_images, train_labels)
        
        return clf
    
    def generate_adversarial_examples(self, clf, images, labels, epsilon=0.5):
        """
        Generate adversarial examples using SMART aggressive attack
        
        Strategy:
        1. Attack ALL correctly classified samples (no holding back!)
        2. Use adaptive step sizes (coarse → fine search)
        3. Target the most confusing wrong class
        4. Use model sensitivity to guide perturbations
        """
        self.log(f"Generating adversarial examples (epsilon={epsilon})...", '⚔️')
        self.log("Using SMART AGGRESSIVE strategy - this will take ~10 minutes", '🎯')
        
        adversarial_images = []
        attack_success = 0
        total_attacked = 0
        
        for i, (image, true_label) in enumerate(zip(images, labels)):
            if i % 250 == 0:
                success_rate = attack_success / (total_attacked + 1e-10) * 100
                self.log(f"  Progress: {i}/{len(images)} | Success: {attack_success}/{total_attacked} ({success_rate:.1f}%)", '⏳')
            
            # Get model's prediction
            pred = clf.predict([image])[0]
            
            # Attack ALL correctly classified samples
            if pred == true_label:
                total_attacked += 1
                perturbed, success = self._smart_aggressive_attack(clf, image, true_label, epsilon)
                if success:
                    attack_success += 1
                adversarial_images.append(perturbed)
            else:
                # Already wrong, keep it
                adversarial_images.append(image)
        
        success_rate = attack_success / (total_attacked + 1e-10) * 100
        self.log(f"Adversarial generation complete!", '✓')
        self.log(f"Attack success rate: {attack_success}/{total_attacked} = {success_rate:.1f}%", '🎯')
        
        self.results['attack_success_count'] = int(attack_success)
        self.results['total_attacked'] = int(total_attacked)
        self.results['attack_success_rate'] = float(success_rate)
        
        return np.array(adversarial_images)
    
    def _smart_aggressive_attack(self, clf, image, true_label, epsilon):
        """
        Smart adaptive attack with multiple strategies
        """
        # Get initial probabilities
        probs = clf.predict_proba([image])[0]
        
        # Find target: the wrong class with HIGHEST probability (easiest to fool toward)
        target_class = None
        max_wrong_prob = -1
        for cls in range(10):
            if cls != true_label and probs[cls] > max_wrong_prob:
                max_wrong_prob = probs[cls]
                target_class = cls
        
        if target_class is None:
            return image, False
        
        perturbed = image.copy()
        
        # PHASE 1: Coarse search (big steps, find promising direction)
        for _ in range(20):
            noise = np.random.randn(*image.shape) * 0.5
            candidate = image + epsilon * noise / (np.linalg.norm(noise) + 1e-10)
            candidate = np.clip(candidate, 0, 1)
            
            new_probs = clf.predict_proba([candidate])[0]
            if new_probs[target_class] > probs[target_class]:
                perturbed = candidate
                probs = new_probs
                
                # Check if we fooled it
                new_pred = clf.predict([perturbed])[0]
                if new_pred != true_label:
                    return perturbed, True
        
        # PHASE 2: Refined search (smaller steps, optimize)
        for iteration in range(30):
            best_candidate = None
            best_target_prob = probs[target_class]
            
            # Try multiple perturbations
            for _ in range(10):
                # Smaller, more targeted perturbations
                step_size = epsilon * (0.5 - iteration / 60.0)  # Decreasing step
                noise = np.random.randn(*image.shape)
                
                candidate = perturbed + step_size * noise / (np.linalg.norm(noise) + 1e-10)
                
                # Ensure within epsilon budget of original
                total_pert = candidate - image
                if np.max(np.abs(total_pert)) > epsilon:
                    total_pert = total_pert * epsilon / (np.max(np.abs(total_pert)) + 1e-10)
                    candidate = image + total_pert
                
                candidate = np.clip(candidate, 0, 1)
                
                new_probs = clf.predict_proba([candidate])[0]
                if new_probs[target_class] > best_target_prob:
                    best_target_prob = new_probs[target_class]
                    best_candidate = candidate
            
            # Apply best found
            if best_candidate is not None:
                perturbed = best_candidate
                probs = clf.predict_proba([perturbed])[0]
                
                # Check success
                new_pred = clf.predict([perturbed])[0]
                if new_pred != true_label:
                    return perturbed, True
        
        # PHASE 3: Desperate last-ditch attempts with maximum perturbation
        for _ in range(10):
            noise = np.random.randn(*image.shape)
            candidate = image + epsilon * noise / (np.linalg.norm(noise) + 1e-10)
            candidate = np.clip(candidate, 0, 1)
            
            new_pred = clf.predict([candidate])[0]
            if new_pred != true_label:
                return candidate, True
        
        return perturbed, False
    
    def evaluate_detection(self, clf, test_images, test_labels, adversarial_images):
        """
        Evaluate Task-Identity on clean vs adversarial examples
        """
        self.log("\nEvaluating Task-Identity for attack detection...", '🔍')
        
        # Get predictions on clean test set
        preds_clean = clf.predict(test_images)
        acc_clean = (preds_clean == test_labels).mean()
        
        # Get predictions on adversarial examples
        preds_adversarial = clf.predict(adversarial_images)
        acc_adversarial = (preds_adversarial == test_labels).mean()
        
        self.log(f"\n{'='*70}")
        self.log("ACCURACY COMPARISON", '📊')
        self.log(f"{'='*70}")
        self.log(f"Clean data accuracy: {acc_clean:.3f}", '✓')
        self.log(f"Adversarial data accuracy: {acc_adversarial:.3f}", '⚠️')
        self.log(f"Accuracy drop: {(acc_clean - acc_adversarial):.3f} ({((acc_clean - acc_adversarial)/acc_clean*100):.1f}%)", '🔻')
        
        # Calculate Task-Identity: Clean baseline vs Clean test
        task_id_clean = calculate_task_identity(
            test_labels, preds_clean,
            test_labels, preds_clean,
            labels=range(10)
        )
        
        # Calculate Task-Identity: Clean baseline vs Adversarial
        task_id_adversarial = calculate_task_identity(
            test_labels, preds_clean,
            test_labels, preds_adversarial,
            labels=range(10)
        )
        
        self.log(f"\n{'='*70}")
        self.log("TASK-IDENTITY ANALYSIS", '💥')
        self.log(f"{'='*70}")
        self.log(f"Task-Identity (clean baseline): {task_id_clean:.3f}", '✓')
        self.log(f"Task-Identity (vs adversarial): {task_id_adversarial:.3f}", '⚠️')
        self.log(f"Behavioral change detected: {(1 - task_id_adversarial):.3f} ({(1 - task_id_adversarial)*100:.1f}%)", '🎯')
        
        # Store results
        self.results['accuracy_clean'] = float(acc_clean)
        self.results['accuracy_adversarial'] = float(acc_adversarial)
        self.results['accuracy_drop'] = float(acc_clean - acc_adversarial)
        self.results['task_identity_clean'] = float(task_id_clean)
        self.results['task_identity_adversarial'] = float(task_id_adversarial)
        self.results['behavioral_change'] = float(1 - task_id_adversarial)
        
        return task_id_adversarial, acc_clean, acc_adversarial
    
    def run(self):
        print("\n" + "="*70)
        print("⚔️  ADVERSARIAL ATTACK DETECTION TEST - V3: SMART AGGRESSIVE")
        print("Testing Task-Identity with intelligent iterative attacks")
        print("="*70 + "\n")
        
        # Load data
        train_images, train_labels, test_images, test_labels = self.load_mnist()
        
        # Train model
        clf = self.train_model(train_images, train_labels)
        
        # Test on clean data first
        clean_acc = clf.score(test_images, test_labels)
        self.log(f"Baseline accuracy on clean data: {clean_acc:.3f}", '✓')
        
        # Generate adversarial examples
        adversarial_images = self.generate_adversarial_examples(
            clf, test_images, test_labels, epsilon=0.5
        )
        
        # Evaluate detection capability
        task_id, acc_clean, acc_adv = self.evaluate_detection(
            clf, test_images, test_labels, adversarial_images
        )
        
        # Final summary
        print("\n" + "="*70)
        print("📊 FINAL RESULTS")
        print("="*70)
        
        acc_drop_pct = ((acc_clean-acc_adv)/acc_clean*100)
        behavior_change_pct = (1-task_id)*100
        
        self.log(f"Accuracy drop: {acc_clean:.3f} → {acc_adv:.3f} ({acc_drop_pct:.1f}% decrease)", '📉')
        self.log(f"Task-Identity (adversarial): {task_id:.3f}", '💥')
        self.log(f"Behavioral change: {behavior_change_pct:.1f}%", '🎯')
        
        if task_id < 0.7:
            print("\n🎯 SUCCESS: Task-Identity detected significant behavioral change!")
            print(f"   Adversarial attacks cause {behavior_change_pct:.0f}% behavioral drift")
            print(f"   Security application: Detects {acc_drop_pct:.0f}% accuracy degradation")
        elif task_id < 0.9:
            print("\n✓ Task-Identity detected moderate behavioral change")
            print(f"   Adversarial attacks cause {behavior_change_pct:.0f}% behavioral shift")
        else:
            print("\n⚠️  Minimal behavioral change detected")
            print(f"   Attack success rate may need improvement")
        
        # Save results
        os.makedirs('results', exist_ok=True)
        filename = f"results/adversarial_detection_{self.results['timestamp']}.json"
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.log(f"\nResults saved: {filename}", '✓')
        print("="*70 + "\n")
        
        return self.results


if __name__ == "__main__":
    validator = AdversarialDetectionTest()
    results = validator.run()
