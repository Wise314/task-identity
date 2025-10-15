#!/usr/bin/env python3
"""
BREAKTHROUGH: Task-Identity Metric for AI
Measures behavioral similarity, not embedding similarity!
"""

import yaml
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import json
import sys
import os
from datetime import datetime
import math

class TaskIdentityValidator:
    
    def __init__(self, config_path):
        np.random.seed(42)
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.results = {
            'domain': 'ai_catastrophic_forgetting',
            'test_type': 'TASK_IDENTITY',
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'embedding_identity': None,
            'task_identity': None,
            'alpha_results': {}
        }
    
    def log(self, msg, level='INFO'):
        icons = {'INFO': '📊', 'SUCCESS': '✓', 'WARNING': '⚠️', 'CALC': '🧮', 'FIRE': '🔥', 'BOOM': '💥'}
        print(f"{icons.get(level, '•')} {msg}")
    
    def calculate_task_identity(self, clf, images, labels):
        """
        CRITICAL: Measure how SIMILARLY the model BEHAVES
        Not embeddings - actual predictions!
        """
        # Get predictions
        predictions = clf.predict(images)
        
        # Create confusion matrix (model's behavior fingerprint)
        cm = confusion_matrix(labels, predictions, labels=range(10))
        
        return cm, predictions
    
    def compare_task_identity(self, cm_before, cm_after):
        """
        Compare two confusion matrices
        High similarity = model behaves similarly
        Low similarity = model behavior changed drastically
        """
        # Flatten and correlate
        flat_before = cm_before.flatten()
        flat_after = cm_after.flatten()
        
        # Correlation coefficient
        if flat_before.std() == 0 or flat_after.std() == 0:
            return 0.0
        
        correlation = np.corrcoef(flat_before, flat_after)[0, 1]
        
        if np.isnan(correlation):
            return 0.0
        
        return max(0.0, correlation)  # Clip to [0, 1]
    
    def load_and_split_data(self):
        """Load MNIST and split by digit classes"""
        self.log("Loading MNIST...", 'INFO')
        
        mnist = fetch_openml('mnist_784', parser='auto')
        images = np.array(mnist.data[:60000]) / 255.0
        labels = np.array(mnist.target[:60000], dtype=int)
        
        # Phase 1 classes: 0-4
        phase1_mask = labels <= 4
        phase1_images = images[phase1_mask]
        phase1_labels = labels[phase1_mask]
        
        # Phase 2 classes: 5-9
        phase2_mask = labels >= 5
        phase2_images = images[phase2_mask]
        phase2_labels = labels[phase2_mask]
        
        self.log(f"Phase 1 (0-4): {len(phase1_images)} samples", 'SUCCESS')
        self.log(f"Phase 2 (5-9): {len(phase2_images)} samples", 'SUCCESS')
        
        return {
            'phase1_images': phase1_images,
            'phase1_labels': phase1_labels,
            'phase2_images': phase2_images,
            'phase2_labels': phase2_labels
        }
    
    def train_phase1(self, data):
        """Train on digits 0-4"""
        self.log("Phase 1: Training on digits 0-4...", 'INFO')
        
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=1,
            random_state=42,
            verbose=False
        )
        
        all_classes = np.array([0,1,2,3,4,5,6,7,8,9])
        
        train_size = int(len(data['phase1_images']) * 0.8)
        
        # Train
        for epoch in range(20):
            clf.partial_fit(
                data['phase1_images'][:train_size],
                data['phase1_labels'][:train_size],
                classes=all_classes
            )
        
        # Test BEFORE fine-tuning
        test_images = data['phase1_images'][train_size:]
        test_labels = data['phase1_labels'][train_size:]
        
        cm_before, preds_before = self.calculate_task_identity(clf, test_images, test_labels)
        
        acc_before = (preds_before == test_labels).mean()
        
        self.log(f"Phase 1 accuracy (before): {acc_before:.3f}", 'SUCCESS')
        
        return clf, train_size, cm_before, test_images, test_labels
    
    def fine_tune_phase2(self, clf, data):
        """Fine-tune HEAVILY on digits 5-9"""
        self.log("Phase 2: Fine-tuning HEAVILY on digits 5-9...", 'FIRE')
        
        # Fine-tune for many epochs
        for epoch in range(40):
            clf.partial_fit(
                data['phase2_images'][:10000],
                data['phase2_labels'][:10000]
            )
        
        return clf
    
    def validate(self):
        print("\n" + "="*70)
        print("💥 TASK-IDENTITY VALIDATOR - THE BREAKTHROUGH")
        print("Measuring BEHAVIORAL similarity, not embedding similarity!")
        print("="*70 + "\n")
        
        data = self.load_and_split_data()
        
        # Phase 1: Train and get BEFORE confusion matrix
        clf, train_size, cm_before, test_images, test_labels = self.train_phase1(data)
        
        # Phase 2: Fine-tune
        clf_finetuned = self.fine_tune_phase2(clf, data)
        
        # Test AFTER fine-tuning on SAME phase 1 data
        cm_after, preds_after = self.calculate_task_identity(clf_finetuned, test_images, test_labels)
        
        acc_after = (preds_after == test_labels).mean()
        
        # Calculate TASK identity
        task_identity = self.compare_task_identity(cm_before, cm_after)
        
        print("\n" + "="*70)
        print("💥 THE MOMENT OF TRUTH")
        print("="*70)
        
        self.log(f"Phase 1 accuracy BEFORE: {(cm_before.diagonal().sum() / cm_before.sum()):.3f}", 'INFO')
        self.log(f"Phase 1 accuracy AFTER:  {acc_after:.3f}", 'WARNING')
        
        self.log(f"TASK-IDENTITY: {task_identity:.3f}", 'BOOM')
        
        self.results['task_identity'] = float(task_identity)
        
        if task_identity < 0.4:
            self.log("🎯 LOW TASK-IDENTITY! Config 2 should show DRAMATIC rescue!", 'FIRE')
        elif task_identity < 0.7:
            self.log("⚠️  MODERATE task-identity - Config 2 may help", 'WARNING')
        else:
            self.log("ℹ️  HIGH task-identity - battery-like", 'INFO')
        
        print("\n📊 Confusion Matrix BEFORE:")
        print(cm_before)
        print("\n📊 Confusion Matrix AFTER:")
        print(cm_after)
        
        # Save
        os.makedirs('results', exist_ok=True)
        filename = f"results/TASK_IDENTITY_{self.results['timestamp']}.json"
        
        self.results['cm_before'] = cm_before.tolist()
        self.results['cm_after'] = cm_after.tolist()
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.log(f"Results saved: {filename}", 'SUCCESS')
        
        print("\n" + "="*70)
        if task_identity < 0.4:
            print("💥 BREAKTHROUGH! LOW TASK-IDENTITY ACHIEVED!")
            print("Next: Build full validator with Config 2 using task-identity!")
        print("="*70)
        
        return self.results


def main():
    if len(sys.argv) != 2:
        print("Usage: python task_identity_validator.py <config.yaml>")
        sys.exit(1)
    
    validator = TaskIdentityValidator(sys.argv[1])
    validator.validate()


if __name__ == "__main__":
    main()
