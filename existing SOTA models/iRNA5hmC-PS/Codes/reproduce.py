import random
import string
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tabulate import tabulate

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from matplotlib import pyplot
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from imblearn.metrics import specificity_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os
import random
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
import csv


def set_reproducibility(seed_value=1):
    # Set random seed for Python's built-in random module
    random.seed(seed_value)

    # Set random seed for NumPy
    np.random.seed(seed_value)

    os.environ['PYTHONHASHSEED'] = str(seed_value)

def sub_feature_importance(X, y, name):
    random_forest_classifier = RandomForestClassifier()
    random_forest_classifier.fit(X, y)
    importances = random_forest_classifier.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    important_features = []
    for i in range(X.shape[1]):
        ## A
        if name == 'PDF':
            if indices[i] <= 3299:
                important_features.append(indices[i])
        ## B
        if name == 'PDG_1':
            if indices[i] >= 3300 and indices[i] <= 10403:
                important_features.append(indices[i])
        ## C
        if name == 'PDG_2':
            if indices[i] >= 10404 and indices[i] <= 17507:
                important_features.append(indices[i])
        ## D
        if name == 'PDG_3':
            if indices[i] >= 17508 and indices[i] <= 19875:
                important_features.append(indices[i])
                
    return important_features[:100]

def feature_importance(X, y):
    
    important_features = []
        
    PDF    = sub_feature_importance(X, y, 'PDF')
    PDG_1  = sub_feature_importance(X, y, 'PDG_1')
    PDG_2  = sub_feature_importance(X, y, 'PDG_2')
    PDG_3  = sub_feature_importance(X, y, 'PDG_3')
     
    for i in PDF:
        important_features.append(i)
        
    for i in PDG_1:
        important_features.append(i)
        
    for i in PDG_2:
        important_features.append(i)
        
    for i in PDG_3:
        important_features.append(i)
        
    important_features.append(19876)
   
    return important_features

def scores(classifier, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    predictions_proba = classifier.predict_proba(X_test)
    predictions_proba = predictions_proba[:, 1]
    
    fp, tp, threshold = roc_curve(y_test, predictions_proba)
    precision_position, recall_position, _ = precision_recall_curve(y_test, predictions_proba)
    
    accuracy    = round(accuracy_score(y_test, predictions) * 100, 4)
    auc         = round(roc_auc_score(y_test, predictions_proba) * 100, 4)
    aupr        = round(average_precision_score(y_test, predictions_proba) * 100, 4)
    precision   = round(precision_score(y_test, predictions, average='binary') * 100, 4)
    recall      = round(recall_score(y_test, predictions, average='binary') * 100, 4)
    specificity = round(specificity_score(y_test, predictions) * 100, 4)
    f1          = round(f1_score(y_test, predictions, average='binary') * 100, 4)
    mcc         = round(matthews_corrcoef(y_test, predictions) * 100, 4)
    
    return accuracy, auc, aupr, precision, recall, specificity, f1, mcc, fp, tp, y_test, predictions_proba


def independent_test_scores(classifier, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    predictions_proba = classifier.predict_proba(X_test)
    predictions_proba = predictions_proba[:, 1]
    
    fp, tp, threshold = roc_curve(y_test, predictions_proba)
    precision_position, recall_position, _ = precision_recall_curve(y_test, predictions_proba)
    
    accuracy    = round(accuracy_score(y_test, predictions), 3)
    auc         = round(roc_auc_score(y_test, predictions_proba), 3)
    aupr        = round(average_precision_score(y_test, predictions_proba), 3)
    precision   = round(precision_score(y_test, predictions, average='binary'), 3)
    recall      = round(recall_score(y_test, predictions, average='binary'), 3)
    specificity = round(specificity_score(y_test, predictions), 3)
    f1          = round(f1_score(y_test, predictions, average='binary'), 3)
    mcc         = round(matthews_corrcoef(y_test, predictions), 3)
        
    return accuracy, auc, aupr, precision, recall, specificity, f1, mcc, fp, tp, y_test, predictions_proba


set_reproducibility(seed_value=1)
df = pd.read_csv("../Data/All_Features_Dataset.csv")
X = df.iloc[:,1:]
y = df.iloc[:,0]

encoder = preprocessing.LabelEncoder()
y = encoder.fit_transform(y)
print(y)

print(f"Class distribution in y: {np.unique(y, return_counts=True)}")

print(X.shape)

X_train_indices = pd.read_csv("train_indices.txt", header=None)
X_test_indices = pd.read_csv("test_indices.txt", header=None)

X_train_indices = X_train_indices.values.flatten()
X_test_indices = X_test_indices.values.flatten()

X_train = X.iloc[X_train_indices]
X_test = X.iloc[X_test_indices]
y_train = y[X_train_indices]
y_test = y[X_test_indices]

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(f"Class distribution in y: {np.unique(y_train, return_counts=True)}")
print(f"Class distribution in y: {np.unique(y_test, return_counts=True)}")

features = feature_importance(X_train, y_train)

X_train_independent = X_train.iloc[:, features]
X_test_independent  = X_test.iloc[:, features]

print(X_train_independent.shape, X_test_independent.shape)

classifier = LogisticRegression()

print("\nIndependent Results:\n")

temp_accuracy    = []
temp_auc         = []
temp_aupr        = []
temp_precision   = []
temp_recall      = []
temp_specificity = []
temp_f1          = []
temp_mcc         = []


accuracy, auc, aupr, precision, recall, specificity, f1, mcc, fp, tp, y_test, y_proba = independent_test_scores(classifier, X_train_independent , X_test_independent, y_train, y_test)
temp_accuracy.append(accuracy)
temp_auc.append(auc)
temp_aupr.append(aupr)
temp_precision.append(precision)
temp_recall.append(recall)
temp_specificity.append(specificity)
temp_f1.append(f1)
temp_mcc.append(mcc)

accuracy    = np.mean(temp_accuracy)
auc         = np.mean(temp_auc)
aupr        = np.mean(temp_aupr)
precision   = np.mean(temp_precision)
recall      = np.mean(temp_recall)
specificity = np.mean(temp_specificity)
f1          = np.mean(temp_f1)
mcc         = np.mean(temp_mcc)

std_accuracy    = np.std(temp_accuracy, dtype = np.float32)
std_auc         = np.std(temp_auc, dtype = np.float32)
std_aupr        = np.std(temp_aupr, dtype = np.float32)
std_precision   = np.std(temp_precision, dtype = np.float32)
std_recall      = np.std(temp_recall, dtype = np.float32)
std_specificity = np.std(temp_specificity, dtype = np.float32)
std_f1          = np.std(temp_f1, dtype = np.float32)
std_mcc         = np.std(temp_mcc, dtype = np.float32)


print("Recall: %.3f" % (recall))
print("Specificity: %.3f" % (specificity))
print("Accuracy: %.3f" % (accuracy))
print("Balanced Accuracy: %.3f" % ((recall + specificity) / 2))
print("Precision: %.3f" % (precision))
print("F1: %.3f" % (f1))
print("MCC: %.3f" % (mcc))
print("AUC: %.3f" % (auc))
print("AUPR: %.3f" % (aupr))

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