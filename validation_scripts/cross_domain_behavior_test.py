#!/usr/bin/env python3
"""
CROSS-DOMAIN BEHAVIOR TEST
Train on MNIST vs Fashion-MNIST, test both on MNIST
Shows Task-Identity detects when models trained on different domains behave differently
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from task_identity import calculate_task_identity
import json
import os
from datetime import datetime

class CrossDomainBehaviorTest:
    
    def __init__(self, random_seed=42):
        np.random.seed(random_seed)
        self.results = {
            'test': 'cross_domain_behavior',
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
    
    def log(self, msg, icon='📊'):
        print(f"{icon} {msg}")
    
    def load_datasets(self):
        """Load both MNIST and Fashion-MNIST"""
        self.log("Loading datasets...", '📥')
        
        # MNIST
        mnist = fetch_openml('mnist_784', parser='auto')
        mnist_imgs = np.array(mnist.data[:10000]) / 255.0
        mnist_labels = np.array(mnist.target[:10000], dtype=int)
        
        # Fashion-MNIST  
        fashion = fetch_openml('Fashion-MNIST', parser='auto')
        fashion_imgs = np.array(fashion.data[:10000]) / 255.0
        fashion_labels = np.array(fashion.target[:10000], dtype=int)
        
        # Split MNIST
        mnist_train = mnist_imgs[:7000]
        mnist_train_labels = mnist_labels[:7000]
        mnist_test = mnist_imgs[7000:]
        mnist_test_labels = mnist_labels[7000:]
        
        # Split Fashion
        fashion_train = fashion_imgs[:7000]
        fashion_train_labels = fashion_labels[:7000]
        
        self.log(f"✓ MNIST: {len(mnist_train)} train, {len(mnist_test)} test", '  ')
        self.log(f"✓ Fashion-MNIST: {len(fashion_train)} train", '  ')
        
        return mnist_train, mnist_train_labels, mnist_test, mnist_test_labels, fashion_train, fashion_train_labels
    
    def train_model(self, images, labels, name):
        """Train a model"""
        self.log(f"\n🧠 Training model on {name}...", '🧠')
        
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=20,
            random_state=42,
            verbose=False
        )
        
        clf.fit(images, labels)
        return clf
    
    def run(self):
        print("\n" + "="*70)
        print("🌍 CROSS-DOMAIN BEHAVIOR TEST")
        print("Compare models trained on different domains (MNIST vs Fashion)")
        print("="*70)
        
        # Load data
        mnist_train, mnist_train_labels, mnist_test, mnist_test_labels, fashion_train, fashion_train_labels = self.load_datasets()
        
        # Train model on MNIST
        clf_mnist = self.train_model(mnist_train, mnist_train_labels, "MNIST (digits)")
        acc_mnist = clf_mnist.score(mnist_test, mnist_test_labels)
        self.log(f"✓ MNIST model accuracy on MNIST test: {acc_mnist:.3f}", '✓')
        
        # Train model on Fashion-MNIST
        clf_fashion = self.train_model(fashion_train, fashion_train_labels, "Fashion-MNIST (clothing)")
        acc_fashion_on_mnist = clf_fashion.score(mnist_test, mnist_test_labels)
        self.log(f"⚠️ Fashion model accuracy on MNIST test: {acc_fashion_on_mnist:.3f}", '⚠️')
        
        # Get predictions from both models ON THE SAME TEST SET (MNIST)
        preds_mnist_model = clf_mnist.predict(mnist_test)
        preds_fashion_model = clf_fashion.predict(mnist_test)
        
        self.log(f"\n{'='*70}")
        self.log("BEHAVIORAL COMPARISON", '📊')
        self.log(f"{'='*70}")
        self.log(f"MNIST-trained model on MNIST: {acc_mnist:.3f}", '  ')
        self.log(f"Fashion-trained model on MNIST: {acc_fashion_on_mnist:.3f}", '  ')
        
        # Calculate Task-Identity between the two models' behaviors
        task_identity = calculate_task_identity(
            mnist_test_labels, preds_mnist_model,
            mnist_test_labels, preds_fashion_model,
            labels=range(10)
        )
        
        self.log(f"\n{'='*70}")
        self.log("TASK-IDENTITY ANALYSIS", '💥')
        self.log(f"{'='*70}")
        self.log(f"Task-Identity (MNIST-model vs Fashion-model): {task_identity:.3f}", '🎯')
        self.log(f"Behavioral divergence: {(1-task_identity)*100:.1f}%", '📉')
        
        print("\n" + "="*70)
        print("📊 FINAL RESULTS")
        print("="*70)
        
        self.log(f"Task-Identity: {task_identity:.3f}", '💥')
        self.log(f"Domain shift detected: {(1-task_identity)*100:.1f}%", '🎯')
        
        print()
        if task_identity < 0.3:
            print("🎯 SUCCESS: Task-Identity detected MAJOR domain difference!")
            print(f"   Models trained on different domains behave {(1-task_identity)*100:.0f}% differently")
        elif task_identity < 0.7:
            print("✓ Task-Identity detected significant behavioral difference")
            print(f"   {(1-task_identity)*100:.0f}% behavioral divergence between domains")
        else:
            print("⚠️ Models show similar behavior despite different training domains")
        
        self.results['task_identity'] = float(task_identity)
        self.results['mnist_model_acc'] = float(acc_mnist)
        self.results['fashion_model_acc_on_mnist'] = float(acc_fashion_on_mnist)
        
        os.makedirs('results', exist_ok=True)
        filename = f"results/cross_domain_{self.results['timestamp']}.json"
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.log(f"\nResults saved: {filename}", '✓')
        print("="*70 + "\n")
        
        return self.results


if __name__ == "__main__":
    tester = CrossDomainBehaviorTest()
    results = tester.run()
