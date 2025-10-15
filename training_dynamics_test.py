#!/usr/bin/env python3
"""
TRAINING DYNAMICS TEST
Test Task-Identity's ability to measure behavioral convergence during training
Scenario: Compare models at different training stages (early vs converged vs overtrained)
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from task_identity import calculate_task_identity
import json
import os
from datetime import datetime

class TrainingDynamicsTest:
    
    def __init__(self, random_seed=42):
        np.random.seed(random_seed)
        self.results = {
            'test': 'training_dynamics',
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'models': {}
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
    
    def train_model_epochs(self, train_images, train_labels, max_iter, stage_name):
        """Train model for specific number of iterations"""
        self.log(f"\n🧠 Training model: {stage_name} ({max_iter} iterations)...", '🧠')
        
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=max_iter,
            random_state=42,
            verbose=False
        )
        
        clf.fit(train_images, train_labels)
        
        return clf
    
    def compare_training_stages(self, models, test_images, test_labels):
        """
        Compare behavioral similarity across training stages
        """
        self.log("\n🔍 Comparing models at different training stages...", '🔍')
        
        # Get predictions from all models
        predictions = {}
        accuracies = {}
        
        for stage_name, clf in models.items():
            preds = clf.predict(test_images)
            acc = (preds == test_labels).mean()
            predictions[stage_name] = preds
            accuracies[stage_name] = acc
            
            self.log(f"✓ {stage_name}: {acc:.3f} accuracy", '  ')
        
        # Calculate Task-Identity between all pairs
        self.log(f"\n{'='*70}")
        self.log("PAIRWISE TASK-IDENTITY ANALYSIS", '💥')
        self.log(f"{'='*70}")
        
        comparisons = {}
        stage_names = list(models.keys())
        
        for i, stage1 in enumerate(stage_names):
            for stage2 in stage_names[i+1:]:
                task_id = calculate_task_identity(
                    test_labels, predictions[stage1],
                    test_labels, predictions[stage2],
                    labels=range(10)
                )
                
                comparison_key = f"{stage1}_vs_{stage2}"
                comparisons[comparison_key] = float(task_id)
                
                # Interpret
                if task_id > 0.95:
                    icon = '✓'
                    status = "Nearly identical"
                elif task_id > 0.85:
                    icon = '⚠️'
                    status = "Minor differences"
                elif task_id > 0.7:
                    icon = '⚠️⚠️'
                    status = "Moderate divergence"
                else:
                    icon = '🚨'
                    status = "Major differences"
                
                self.log(f"{icon} {stage1} ↔ {stage2}: Task-Identity = {task_id:.3f} ({status})", '  ')
        
        # Analyze training progression
        self.log(f"\n{'='*70}")
        self.log("TRAINING PROGRESSION ANALYSIS", '📈')
        self.log(f"{'='*70}")
        
        # Early → Normal
        early_to_normal = comparisons.get('undertrained_vs_normal', None)
        if early_to_normal:
            self.log(f"Undertrained → Normal training:", '  ')
            self.log(f"  Behavioral similarity: {early_to_normal:.3f}", '    ')
            self.log(f"  Behavioral change: {(1-early_to_normal)*100:.1f}%", '    ')
        
        # Normal → Extended
        normal_to_extended = comparisons.get('normal_vs_extended', None)
        if normal_to_extended:
            self.log(f"\nNormal → Extended training:", '  ')
            self.log(f"  Behavioral similarity: {normal_to_extended:.3f}", '    ')
            self.log(f"  Behavioral change: {(1-normal_to_extended)*100:.1f}%", '    ')
            
            if normal_to_extended > 0.98:
                self.log(f"  ✓ Converged: Additional training didn't change behavior", '    ')
            elif normal_to_extended > 0.95:
                self.log(f"  ⚠️ Minor refinement: Slight behavioral improvement", '    ')
            else:
                self.log(f"  🚨 Significant change: Model still evolving", '    ')
        
        # Early → Extended (overall progress)
        early_to_extended = comparisons.get('undertrained_vs_extended', None)
        if early_to_extended:
            self.log(f"\nUndertrained → Extended training:", '  ')
            self.log(f"  Total behavioral evolution: {(1-early_to_extended)*100:.1f}%", '    ')
        
        # Accuracy progression
        self.log(f"\n{'='*70}")
        self.log("ACCURACY PROGRESSION", '📊')
        self.log(f"{'='*70}")
        
        for stage_name in stage_names:
            self.log(f"{stage_name}: {accuracies[stage_name]:.3f}", '  ')
        
        # Store results
        self.results['accuracies'] = {k: float(v) for k, v in accuracies.items()}
        self.results['task_identity_comparisons'] = comparisons
        
        return comparisons, accuracies
    
    def run(self):
        print("\n" + "="*70)
        print("📈 TRAINING DYNAMICS TEST")
        print("Measuring behavioral convergence across training stages")
        print("="*70)
        
        # Load data
        train_images, train_labels, test_images, test_labels = self.load_mnist()
        
        # Train models at different stages
        models = {}
        
        # Stage 1: Undertrained (5 iterations)
        models['undertrained'] = self.train_model_epochs(
            train_images, train_labels, 5, "Undertrained (5 iter)"
        )
        
        # Stage 2: Normal training (20 iterations)
        models['normal'] = self.train_model_epochs(
            train_images, train_labels, 20, "Normal (20 iter)"
        )
        
        # Stage 3: Extended training (50 iterations)
        models['extended'] = self.train_model_epochs(
            train_images, train_labels, 50, "Extended (50 iter)"
        )
        
        # Compare all models
        comparisons, accuracies = self.compare_training_stages(
            models, test_images, test_labels
        )
        
        # Final summary
        print("\n" + "="*70)
        print("📊 FINAL RESULTS")
        print("="*70)
        
        self.log(f"Accuracy improvement: {accuracies['undertrained']:.3f} → {accuracies['normal']:.3f} → {accuracies['extended']:.3f}", '📈')
        
        early_to_normal = comparisons.get('undertrained_vs_normal', 0)
        normal_to_extended = comparisons.get('normal_vs_extended', 1)
        
        self.log(f"\nBehavioral changes:", '💥')
        self.log(f"  Early → Normal: {(1-early_to_normal)*100:.1f}% change", '  ')
        self.log(f"  Normal → Extended: {(1-normal_to_extended)*100:.1f}% change", '  ')
        
        print()
        if normal_to_extended > 0.98:
            print("✓ Model converged: Extended training provides no behavioral benefit")
            print("  Recommendation: Stop at 20 iterations to save compute")
        elif normal_to_extended > 0.95:
            print("⚠️ Model still refining: Extended training provides minor improvements")
            print(f"  {(1-normal_to_extended)*100:.1f}% additional behavioral refinement")
        else:
            print("🚨 Model still evolving: Extended training significantly changes behavior")
            print(f"  {(1-normal_to_extended)*100:.1f}% behavioral change suggests more training needed")
        
        print()
        if early_to_normal < 0.7:
            print("🎯 SUCCESS: Task-Identity detected major training progression!")
            print(f"   Early model dramatically different from converged ({(1-early_to_normal)*100:.0f}% change)")
        else:
            print("✓ Task-Identity tracked training evolution")
        
        # Save results
        os.makedirs('results', exist_ok=True)
        filename = f"results/training_dynamics_{self.results['timestamp']}.json"
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.log(f"\nResults saved: {filename}", '✓')
        print("="*70 + "\n")
        
        return self.results


if __name__ == "__main__":
    tester = TrainingDynamicsTest()
    results = tester.run()
