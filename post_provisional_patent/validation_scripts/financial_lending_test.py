import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import json
from datetime import datetime

def calculate_task_identity(y_true_before, y_pred_before, 
                            y_true_after, y_pred_after, labels):
    cm_before = confusion_matrix(y_true_before, y_pred_before, labels=labels)
    cm_after = confusion_matrix(y_true_after, y_pred_after, labels=labels)
    flat_before = cm_before.flatten()
    flat_after = cm_after.flatten()
    if flat_before.std() == 0 or flat_after.std() == 0:
        return 0.0
    correlation = np.corrcoef(flat_before, flat_after)[0, 1]
    return max(0.0, correlation) if not np.isnan(correlation) else 0.0

def calculate_per_class_task_identity(y_true, y_pred_before, y_pred_after, target_class):
    mask = (y_true == target_class)
    if mask.sum() < 10:
        return None
    
    y_true_class = y_true[mask]
    y_pred_before_class = y_pred_before[mask]
    y_pred_after_class = y_pred_after[mask]
    
    labels_binary = [target_class, 1 - target_class]
    cm_before = confusion_matrix(y_true_class, y_pred_before_class, labels=labels_binary)
    cm_after = confusion_matrix(y_true_class, y_pred_after_class, labels=labels_binary)
    
    flat_before = cm_before.flatten()
    flat_after = cm_after.flatten()
    
    if flat_before.std() == 0 or flat_after.std() == 0:
        return 0.0
    
    correlation = np.corrcoef(flat_before, flat_after)[0, 1]
    return max(0.0, correlation) if not np.isnan(correlation) else 0.0

print("="*70)
print("TASK-IDENTITY: LENDING - PER-CLASS (Test #4 Approach)")
print("="*70)

print("\n📊 Loading...")
df = pd.read_csv('../../archive/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv', low_memory=False)
print(f"✅ {len(df):,} loans")

paid = ['Fully Paid', 'Current']
default = ['Charged Off', 'Default', 'Late (31-120 days)', 'Late (16-30 days)']
df_filtered = df[df['loan_status'].isin(paid + default)].copy()
df_filtered['target'] = (df_filtered['loan_status'].isin(default)).astype(int)

key_features = ['loan_amnt', 'funded_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 'open_acc', 'total_acc', 'revol_bal', 'revol_util', 'total_pymnt']
df_clean = df_filtered[key_features + ['target']].dropna()

X = df_clean[key_features].values
y = df_clean['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Test: {len(X_test):,} (Paid={sum(y_test==0):,}, Default={sum(y_test==1):,})")

print("\n🤖 Training models...")
model1 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1, verbose=0)
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)

model2 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight={0: 1, 1: 0.3}, n_jobs=-1, verbose=0)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)

cm1 = confusion_matrix(y_test, y_pred1, labels=[0, 1])
cm2 = confusion_matrix(y_test, y_pred2, labels=[0, 1])

print("\nBaseline:")
print(cm1)
print(f"Default detection: {cm1[1,1]/(cm1[1,0]+cm1[1,1])*100:.1f}%")

print("\nDrifted:")
print(cm2)
print(f"Default detection: {cm2[1,1]/(cm2[1,0]+cm2[1,1])*100:.1f}%")

print("\n" + "="*70)
print("OVERALL vs PER-CLASS")
print("="*70)

task_overall = calculate_task_identity(y_test, y_pred1, y_test, y_pred2, labels=[0, 1])
task_paid = calculate_per_class_task_identity(y_test, y_pred1, y_pred2, 0)
task_default = calculate_per_class_task_identity(y_test, y_pred1, y_pred2, 1)

print(f"\nOverall: {task_overall:.3f} ({'STABLE' if task_overall > 0.85 else 'DRIFTED'})")
print(f"Paid class: {task_paid:.3f} ({'STABLE' if task_paid > 0.85 else 'DRIFTED'})")
print(f"Default class: {task_default:.3f} ({'STABLE' if task_default > 0.85 else 'CATASTROPHIC!'})")

if task_overall > 0.85 and task_default < 0.85:
    print(f"\n✅ PER-CLASS CAUGHT DRIFT THAT OVERALL MISSED!")
    print(f"   Gap: {(task_overall - task_default)*100:.1f} points")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results = {
    "task_identity_overall": float(task_overall),
    "task_identity_paid": float(task_paid),
    "task_identity_default": float(task_default),
    "default_detection_baseline": float(cm1[1,1]/(cm1[1,0]+cm1[1,1])*100),
    "default_detection_drifted": float(cm2[1,1]/(cm2[1,0]+cm2[1,1])*100),
    "timestamp": timestamp
}

with open(f'../results/12_financial_lending/lending_test_{timestamp}.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Saved: ./results/12_financial_lending/lending_test_{timestamp}.json")
