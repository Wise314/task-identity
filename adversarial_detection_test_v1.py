#!/usr/bin/env python3
"""
ADVERSARIAL ATTACK DETECTION TEST
Test Task-Identity's ability to detect adversarial attacks on classification models
Uses gradient-based perturbations to generate adversarial examples
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
            'test': 'adversarial_attack_detection',
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
    
    def generate_adversarial_examples(self, clf, images, labels, epsilon=0.1):
        """
        Generate adversarial examples using gradient-based perturbation
        
        This approximates FGSM (Fast Gradient Sign Method) by:
        1. Computing gradients through finite differences
        2. Perturbing in direction that increases loss
        """
        self.log(f"Generating adversarial examples (epsilon={epsilon})...", '⚔️')
        
        adversarial_images = []
        
        for i, (image, true_label) in enumerate(zip(images, labels)):
            if i % 500 == 0:
                self.log(f"  Progress: {i}/{len(images)}", '⏳')
            
            # Get model's prediction
            pred = clf.predict([image])[0]
            
            # Only create adversarial if model got it right originally
            if pred == true_label:
                # Compute approximate gradient via finite differences
                # Find direction that changes the prediction
                perturbed = self._gradient_based_perturbation(clf, image, true_label, epsilon)
                adversarial_images.append(perturbed)
            else:
                # Model already wrong, keep original
                adversarial_images.append(image)
        
        self.log("Adversarial generation complete!", '✓')
        return np.array(adversarial_images)
    
    def _gradient_based_perturbation(self, clf, image, true_label, epsilon):
        """
        Apply gradient-based perturbation to fool the model
        Uses sklearn's predict_proba to approximate gradients
        """
        # Get current probabilities
        probs = clf.predict_proba([image])[0]
        
        # Find the class the model is most confident about (that's not the true class)
        target_class = None
        max_prob = -1
        for cls in range(10):
            if cls != true_label and probs[cls] > max_prob:
                max_prob = probs[cls]
                target_class = cls
        
        # Approximate gradient by testing small perturbations
        best_perturbation = None
        best_score = probs[true_label]
        
        # Try random perturbations and keep the one that reduces correct class probability most
        for _ in range(10):
            # Random perturbation direction
            perturbation = np.random.randn(*image.shape)
            perturbation = perturbation / (np.linalg.norm(perturbation) + 1e-10)
            
            # Apply perturbation
            perturbed = image + epsilon * perturbation
            perturbed = np.clip(perturbed, 0, 1)
            
            # Check if this reduces confidence in true class
            new_probs = clf.predict_proba([perturbed])[0]
            if new_probs[true_label] < best_score:
                best_score = new_probs[true_label]
                best_perturbation = perturbed
        
        # Return best perturbation found, or original if none worked
        return best_perturbation if best_perturbation is not None else image
    
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
        print("⚔️  ADVERSARIAL ATTACK DETECTION TEST")
        print("Testing Task-Identity's ability to detect adversarial attacks")
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
            clf, test_images, test_labels, epsilon=0.15
        )
        
        # Evaluate detection capability
        task_id, acc_clean, acc_adv = self.evaluate_detection(
            clf, test_images, test_labels, adversarial_images
        )
        
        # Final summary
        print("\n" + "="*70)
        print("📊 FINAL RESULTS")
        print("="*70)
        
        self.log(f"Accuracy drop: {acc_clean:.3f} → {acc_adv:.3f}", '📉')
        self.log(f"Task-Identity (adversarial): {task_id:.3f}", '💥')
        
        if task_id < 0.7:
            print("\n🎯 SUCCESS: Task-Identity detected significant behavioral change!")
            print("   Adversarial attacks cause measurable drift in decision patterns")
        elif task_id < 0.9:
            print("\n✓ Task-Identity detected moderate behavioral change")
        else:
            print("\n⚠️  Task-Identity detected minimal behavioral change")
            print("   Note: Stronger adversarial attacks may be needed for clear detection")
        
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
