# 5-fold cross validation with independent datasets
import numpy as np
from numpy import loadtxt
from tensorflow import keras
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow.keras import layers
from keras.models import Sequential
from keras.optimizers import Adam as opt
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, f1_score, matthews_corrcoef, precision_recall_curve, auc, accuracy_score, balanced_accuracy_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import random
import os
import tensorflow as tf
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
import csv

def set_reproducibility(seed_value=1):
    # Set random seed for Python's built-in random module
    random.seed(seed_value)

    # Set random seed for NumPy
    np.random.seed(seed_value)

    # Set random seed for TensorFlow
    tf.random.set_seed(seed_value)

    # Set random seed for Keras using TensorFlow
    tf.keras.utils.set_random_seed(seed_value)

    os.environ['PYTHONHASHSEED'] = str(seed_value)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


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

# Set seeds for reproducibility
set_reproducibility(seed_value=1)

dataset = np.loadtxt(r"features.csv", delimiter=",")

X = dataset
y = np.array([1]*662+[0]*662)
print(X.shape, y.shape)

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

model = Sequential()
model.add(Dropout(0.4))
model.add(Dense(96, kernel_initializer='he_uniform', activation='relu', input_dim=319))
#model.add(Dropout(0.4))
model.add(Dense(64, kernel_initializer='he_uniform', activation='relu'))
# model.add3(Dropout(0.01))
model.add(Dense(32, kernel_initializer='he_uniform', activation='relu'))
# model.add(Dropout(0.01))
model.add(Dense(16, kernel_initializer='he_uniform', activation='relu'))
# model.add(Dropout(0.01))
model.add(Dense(8, kernel_initializer='he_uniform', activation='relu'))
model.add(Dense(units=1, kernel_initializer='he_uniform', activation='sigmoid'))

opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='binary_crossentropy', optimizer='Adam',metrics=['accuracy'])

# Fit the model
history = model.fit(X_train, y_train, epochs=32, batch_size=8, verbose=0)

# Evaluate model
y_proba = model.predict(X_test).flatten()  # Predicted probabilities

# Calculate and print all metrics
metrics = calculate_metrics(y_test, y_proba, threshold=0.5)
for metric_name, value in metrics.items():
    print(f"{metric_name}: {value:.3f}")


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