#!/usr/bin/env python3
"""
MODEL COMPRESSION TEST
Test Task-Identity's ability to detect behavioral changes from model compression
Scenario: Simulate weight quantization and measure behavioral preservation
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from task_identity import calculate_task_identity
import json
import os
from datetime import datetime

class ModelCompressionTest:
    
    def __init__(self, random_seed=42):
        np.random.seed(random_seed)
        self.results = {
            'test': 'model_compression',
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
    
    def train_model(self, train_images, train_labels):
        """Train full precision model"""
        self.log("\n🧠 Training full precision model...", '🧠')
        
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=20,
            random_state=42,
            verbose=False
        )
        
        clf.fit(train_images, train_labels)
        
        return clf
    
    def quantize_model(self, clf, bits=8, compression_level='moderate'):
        """
        Simulate model quantization by reducing weight precision
        
        Real quantization: Convert float32 → int8
        Our simulation: Round weights to fewer decimal places + add quantization noise
        
        Args:
            bits: Simulated bit depth (8 = int8, 4 = int4, 2 = int2)
            compression_level: 'light', 'moderate', 'aggressive'
        """
        compression_params = {
            'light': {'noise_scale': 0.01, 'decimals': 4},
            'moderate': {'noise_scale': 0.05, 'decimals': 2},
            'aggressive': {'noise_scale': 0.15, 'decimals': 1}
        }
        
        params = compression_params[compression_level]
        
        self.log(f"\n📦 Compressing model ({compression_level} compression)...", '📦')
        self.log(f"  Simulating {bits}-bit quantization", '  ')
        self.log(f"  Quantization noise scale: {params['noise_scale']}", '  ')
        
        # Create compressed version
        clf_compressed = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=1,  # Don't train, just initialize
            random_state=42,
            verbose=False,
            warm_start=True
        )
        
        # Initialize with dummy data to create weight structure
        dummy_X = np.random.randn(10, 784)
        dummy_y = np.random.randint(0, 10, 10)
        clf_compressed.fit(dummy_X, dummy_y)
        
        # Copy and quantize weights
        clf_compressed.coefs_ = []
        clf_compressed.intercepts_ = []
        
        total_params = 0
        
        for i, (weights, biases) in enumerate(zip(clf.coefs_, clf.intercepts_)):
            # Simulate quantization
            # 1. Round to fewer decimals
            quantized_weights = np.round(weights, decimals=params['decimals'])
            quantized_biases = np.round(biases, decimals=params['decimals'])
            
            # 2. Add quantization noise
            quantized_weights += np.random.randn(*weights.shape) * params['noise_scale']
            quantized_biases += np.random.randn(*biases.shape) * params['noise_scale']
            
            clf_compressed.coefs_.append(quantized_weights)
            clf_compressed.intercepts_.append(quantized_biases)
            
            total_params += weights.size + biases.size
        
        # Calculate theoretical compression ratio
        original_size = total_params * 32  # 32-bit floats
        compressed_size = total_params * bits  # n-bit quantized
        compression_ratio = original_size / compressed_size
        
        self.log(f"  Original size: ~{original_size/8000:.1f}KB (32-bit)", '  ')
        self.log(f"  Compressed size: ~{compressed_size/8000:.1f}KB ({bits}-bit)", '  ')
        self.log(f"  Compression ratio: {compression_ratio:.1f}x smaller", '  ')
        
        self.results['compression_ratio'] = float(compression_ratio)
        self.results['original_size_kb'] = float(original_size/8000)
        self.results['compressed_size_kb'] = float(compressed_size/8000)
        
        return clf_compressed
    
    def evaluate_compression(self, clf_original, clf_compressed, test_images, test_labels, compression_level):
        """
        Evaluate if compression preserved behavior
        """
        self.log("\n🔍 Evaluating compression impact...", '🔍')
        
        # Get predictions from both models
        preds_original = clf_original.predict(test_images)
        preds_compressed = clf_compressed.predict(test_images)
        
        # Calculate accuracies
        acc_original = (preds_original == test_labels).mean()
        acc_compressed = (preds_compressed == test_labels).mean()
        
        self.log(f"\n{'='*70}")
        self.log("ACCURACY COMPARISON", '📊')
        self.log(f"{'='*70}")
        self.log(f"Original model: {acc_original:.3f}", '✓')
        self.log(f"Compressed model: {acc_compressed:.3f}", '📦')
        self.log(f"Accuracy degradation: {(acc_original - acc_compressed):.3f} ({((acc_original - acc_compressed)/acc_original*100):.1f}%)", '📉')
        
        # Calculate Task-Identity
        task_identity = calculate_task_identity(
            test_labels, preds_original,
            test_labels, preds_compressed,
            labels=range(10)
        )
        
        self.log(f"\n{'='*70}")
        self.log("TASK-IDENTITY ANALYSIS", '💥')
        self.log(f"{'='*70}")
        self.log(f"Task-Identity (original vs compressed): {task_identity:.3f}", '🎯')
        self.log(f"Behavioral preservation: {task_identity*100:.1f}%", '✓')
        self.log(f"Behavioral drift: {(1-task_identity)*100:.1f}%", '📉')
        
        # Per-class impact analysis
        self.log(f"\n{'='*70}")
        self.log("PER-CLASS COMPRESSION IMPACT", '🔬')
        self.log(f"{'='*70}")
        
        class_impacts = {}
        
        for cls in range(10):
            cls_mask = (test_labels == cls)
            if cls_mask.sum() > 0:
                acc_orig_cls = (preds_original[cls_mask] == cls).mean()
                acc_comp_cls = (preds_compressed[cls_mask] == cls).mean()
                degradation = acc_orig_cls - acc_comp_cls
                
                class_impacts[cls] = {
                    'original_acc': float(acc_orig_cls),
                    'compressed_acc': float(acc_comp_cls),
                    'degradation': float(degradation)
                }
                
                if degradation > 0.05:
                    icon = '🚨'
                    status = 'High impact'
                elif degradation > 0.02:
                    icon = '⚠️'
                    status = 'Moderate impact'
                else:
                    icon = '✓'
                    status = 'Low impact'
                
                self.log(f"{icon} Class {cls}: {acc_orig_cls:.3f} → {acc_comp_cls:.3f} (Δ {degradation:.3f}) - {status}", '  ')
        
        # Deployment recommendation
        self.log(f"\n{'='*70}")
        self.log("DEPLOYMENT RECOMMENDATION", '🎯')
        self.log(f"{'='*70}")
        
        if task_identity > 0.95 and (acc_original - acc_compressed) < 0.02:
            recommendation = "✅ APPROVED: Compression preserves behavior well"
            deploy = True
        elif task_identity > 0.85 and (acc_original - acc_compressed) < 0.05:
            recommendation = "⚠️ ACCEPTABLE: Minor behavioral changes, validate edge cases"
            deploy = True
        else:
            recommendation = "🚨 NOT RECOMMENDED: Significant behavioral drift detected"
            deploy = False
        
        self.log(recommendation, '  ')
        self.log(f"  Compression ratio: {self.results['compression_ratio']:.1f}x", '  ')
        self.log(f"  Behavioral preservation: {task_identity*100:.1f}%", '  ')
        self.log(f"  Accuracy preserved: {(1 - (acc_original - acc_compressed)/acc_original)*100:.1f}%", '  ')
        
        # Store results
        self.results['compression_level'] = compression_level
        self.results['task_identity'] = float(task_identity)
        self.results['behavioral_drift'] = float(1 - task_identity)
        self.results['accuracy_original'] = float(acc_original)
        self.results['accuracy_compressed'] = float(acc_compressed)
        self.results['accuracy_degradation_pct'] = float((acc_original - acc_compressed)/acc_original*100)
        self.results['deployment_approved'] = bool(deploy)
        self.results['class_impacts'] = class_impacts
        
        return task_identity, acc_original, acc_compressed, deploy
    
    def run(self, compression_level='moderate', bits=8):
        print("\n" + "="*70)
        print("📦 MODEL COMPRESSION TEST")
        print("Testing Task-Identity for model compression validation")
        print("="*70)
        
        # Load data
        train_images, train_labels, test_images, test_labels = self.load_mnist()
        
        # Train original model
        clf_original = self.train_model(train_images, train_labels)
        acc_train = clf_original.score(train_images, train_labels)
        acc_test = clf_original.score(test_images, test_labels)
        self.log(f"✓ Full precision model trained", '✓')
        self.log(f"  Train accuracy: {acc_train:.3f}", '  ')
        self.log(f"  Test accuracy: {acc_test:.3f}", '  ')
        
        # Compress model
        clf_compressed = self.quantize_model(clf_original, bits=bits, compression_level=compression_level)
        
        # Evaluate compression
        task_id, acc_orig, acc_comp, deploy = self.evaluate_compression(
            clf_original, clf_compressed, test_images, test_labels, compression_level
        )
        
        # Final summary
        print("\n" + "="*70)
        print("📊 FINAL RESULTS")
        print("="*70)
        
        self.log(f"Compression: {compression_level} ({bits}-bit quantization)", '📦')
        self.log(f"Size reduction: {self.results['compression_ratio']:.1f}x smaller", '💾')
        self.log(f"Task-Identity: {task_id:.3f}", '💥')
        self.log(f"Behavioral drift: {(1-task_id)*100:.1f}%", '📉')
        self.log(f"Accuracy: {acc_orig:.3f} → {acc_comp:.3f}", '📊')
        
        print()
        if deploy:
            if task_identity > 0.95:
                print("🎯 SUCCESS: Compression preserved behavior excellently!")
                print(f"   {task_id*100:.1f}% behavioral similarity maintained")
                print(f"   {self.results['compression_ratio']:.1f}x size reduction with minimal impact")
                print("   ✅ Safe for deployment")
            else:
                print("✓ Compression acceptable with minor trade-offs")
                print(f"   {(1-task_id)*100:.1f}% behavioral change detected")
                print("   ⚠️ Validate critical use cases before deployment")
        else:
            print("🚨 Compression caused significant behavioral drift")
            print(f"   {(1-task_id)*100:.1f}% behavioral change")
            print("   🚫 NOT recommended for deployment without review")
        
        # Save results
        os.makedirs('results', exist_ok=True)
        filename = f"results/08_model_compression/model_compression_{self.results['timestamp']}.json"
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.log(f"\nResults saved: {filename}", '✓')
        print("="*70 + "\n")
        
        return self.results


if __name__ == "__main__":
    tester = ModelCompressionTest()
    # Test moderate compression (8-bit quantization simulation)
    results = tester.run(compression_level='moderate', bits=8)
