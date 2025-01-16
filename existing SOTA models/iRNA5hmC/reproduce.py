import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import numpy as np
import random
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, f1_score, matthews_corrcoef, precision_recall_curve, auc, accuracy_score, balanced_accuracy_score, precision_score
from sklearn.svm import SVC
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
import csv



# Define a function to calculate metrics
def calculate_metrics(y_true, y_pred_probs, threshold=0.5):
    y_pred = (y_pred_probs >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Sensitivity (Recall)
    SN = tp / (tp + fn) if (tp + fn) > 0 else 0
    # Specificity
    SP = tn / (tn + fp) if (tn + fp) > 0 else 0
    # Accuracy
    ACC = accuracy_score(y_true, y_pred)
    # Balanced Accuracy
    BACC = balanced_accuracy_score(y_true, y_pred)
    # Precision
    PREC = precision_score(y_true, y_pred, zero_division=0)
    # F1 Score
    F1 = f1_score(y_true, y_pred)
    # Matthews Correlation Coefficient
    MCC = matthews_corrcoef(y_true, y_pred)
    # ROC AUC
    AUC = roc_auc_score(y_true, y_pred_probs)
    # Area Under Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    AUPR = auc(recall, precision)
    
    return {
        "SN": SN, "SP": SP, "ACC": ACC, "BACC": BACC, "PREC": PREC,
        "F1": F1, "MCC": MCC, "AUC": AUC, "AUPR": AUPR
    }

def set_reproducibility(seed_value=1):
    # Set random seed for Python's built-in random module
    random.seed(seed_value)

    # Set random seed for NumPy
    np.random.seed(seed_value)

    torch.manual_seed(seed_value)

    os.environ['PYTHONHASHSEED'] = str(seed_value)

set_reproducibility(seed_value=1)
# Example one-hot encoded input tensor
X = np.load('features.npy')

selected_features = pd.read_csv("top_25_indices.txt", header=None)
selected_features = selected_features.values.flatten()

X = X[:, selected_features]

y = np.array([1] * 662 + [0] * 662)

X_train_indices = pd.read_csv("train_indices.txt", header=None)
X_test_indices = pd.read_csv("test_indices.txt", header=None)

X_train_indices = X_train_indices.values.flatten()
X_test_indices = X_test_indices.values.flatten()

X_train = X[X_train_indices]
X_test = X[X_test_indices]
y_train = y[X_train_indices]
y_test = y[X_test_indices]
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(f"Class distribution in y: {np.unique(y_train, return_counts=True)}")
print(f"Class distribution in y: {np.unique(y_test, return_counts=True)}")

svm_model = SVC(kernel='rbf', C=1.0, random_state=1, probability=True)  # You can adjust kernel type and C parameter
svm_model.fit(X_train, y_train)

y_proba = svm_model.predict_proba(X_test)[:, 1]

test_metrics = calculate_metrics(y_test, y_proba)

print("Test Metrics:")
for metric, value in test_metrics.items():
    print(f"{metric}: {value:.3f}")

fpr, tpr, _ = roc_curve(np.array(y_test), np.array(y_proba))

f = open("./Plot_csvs/fpr.csv", "w", newline="")
writer = csv.writer(f)
writer.writerows([[fp] for fp in fpr])
f.close()

f = open("./Plot_csvs/tpr.csv", "w", newline="")
writer = csv.writer(f)
writer.writerows([[tp] for tp in tpr])
f.close()

precision, recall, _ = precision_recall_curve(np.array(y_test), np.array(y_proba))

f_name = f'./Plot_csvs/precision.csv'
with open(f_name, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows([[p] for p in precision])

f_name = f'./Plot_csvs/recall.csv'
with open(f_name, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows([[r] for r in recall])