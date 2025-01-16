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


class ProteinClassifier(nn.Module):
    def __init__(self):
        super(ProteinClassifier, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.groupnorm = nn.GroupNorm(num_groups=2, num_channels=16)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.4)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=16 * 20, out_features=32)  # Adjusted for maxpooling
        self.relu = nn.ReLU()
        self.output = nn.Linear(32, 1)  # Output layer for binary classification

    def forward(self, x):
        x = x.float()  # Convert input to float
        x = x.permute(0, 2, 1)  # Adjusted for PyTorch input shape
        # Input shape: (batch_size, 4, sequence_length)
        x = self.conv1d(x)  # (batch_size, 16, sequence_length)
        x = self.groupnorm(x)  # (batch_size, 16, sequence_length)
        x = F.relu(x)  # Non-linear activation
        x = self.maxpool(x)  # (batch_size, 16, sequence_length // 2)
        x = self.dropout(x)  # Apply dropout
        x = self.flatten(x)  # (batch_size, 16 * sequence_length // 2)
        x = self.fc(x)  # Fully connected layer
        x = self.relu(x)  # ReLU activation
        x = self.output(x)  # Linear output for binary classification
        x = torch.sigmoid(x)  # Sigmoid activation
        return x


set_reproducibility(seed_value=1)

batch_size = 32
sequence_length = 41
one_hot_features = 4  # Adjusted for one-hot encoded dimensions for RNA

# Example one-hot encoded input tensor
X = np.load('features.npy')

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

# split into 5 folds. three folds for training, one fold for validation, and one fold for testing
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
# turn into PyTorch tensors
X_train, X_val, X_test = map(torch.tensor, (X_train, X_val, X_test))
y_train, y_val, y_test = map(torch.tensor, (y_train, y_val, y_test))
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

train_loader = torch.utils.data.DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(list(zip(X_val, y_val)), batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(list(zip(X_test, y_test)), batch_size=batch_size, shuffle=False)

model = ProteinClassifier()
criterion = nn.BCELoss()  # Binary cross-entropy loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-2)  # L2 regularization with weight_decay
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=10, verbose=True)

num_epochs = 81
best_val_loss = float('inf')
patience = 15
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    # Assume train_loader and val_loader are defined elsewhere
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)

        targets = targets.float()  # Adjusted for BCELoss
        outputs = outputs.reshape(-1)  # Adjusted for BCELoss

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Validation step
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for val_inputs, val_targets in val_loader:
            val_outputs = model(val_inputs)
            val_targets = val_targets.float()  # Adjusted for BCELoss
            val_outputs = val_outputs.reshape(-1)  # Adjusted for BCELoss
            val_loss += criterion(val_outputs, val_targets).item()
    val_loss /= len(val_loader)
    
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1

    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

    if patience_counter >= patience:
        print("Early stopping triggered.")
        break

# Test step
model.eval()
test_loss = 0
predictions = []

with torch.no_grad():
    for test_inputs, test_targets in test_loader:
        test_outputs = model(test_inputs)
        test_targets = test_targets.float()  # Adjusted for BCELoss
        test_outputs = test_outputs.reshape(-1)  # Adjusted for BCELoss
        test_loss += criterion(test_outputs, test_targets).item()
        predictions.extend(test_outputs.tolist())


test_loss /= len(test_loader)
print(f"Test Loss: {test_loss:.4f}")
y_true = y_test.numpy()
y_proba = np.array(predictions)
metrics = calculate_metrics(y_true, y_proba)

print("Test Metrics:")
for metric, value in metrics.items():
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