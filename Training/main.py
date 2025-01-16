import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import os
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, f1_score, matthews_corrcoef, precision_recall_curve, auc, accuracy_score, balanced_accuracy_score, precision_score


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

# Transformer Encoder block
class TransformerBlock(nn.Module):
    def __init__(self, d_model=1280, nhead=8, num_layers=6, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, num_layers=num_layers
        )

    def forward(self, x):
        # Expecting input shape: (batch_size, seq_len, d_model)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        out = self.transformer_encoder(x)
        out = out.permute(1, 0, 2)  # back to (batch_size, seq_len, d_model)
        return out


# Inception Block remains unchanged
class InceptionBlock(nn.Module):
    def __init__(self, filters, input_channels=21):
        super(InceptionBlock, self).__init__()
        self.conv1x1 = nn.Conv1d(in_channels=input_channels, out_channels=filters, kernel_size=1)
        self.conv1x1_3x3 = nn.Conv1d(in_channels=input_channels, out_channels=filters, kernel_size=1)
        self.conv3x3 = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=3, padding=1)
        self.conv1x1_5x5 = nn.Conv1d(in_channels=input_channels, out_channels=filters, kernel_size=1)
        self.conv5x5 = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.pool_proj = nn.Conv1d(in_channels=input_channels, out_channels=filters, kernel_size=1)
        self.batch_norm = nn.BatchNorm1d(filters * 4)

    def forward(self, x):
        path1 = F.relu(self.conv1x1(x))
        path2 = F.relu(self.conv3x3(F.relu(self.conv1x1_3x3(x))))
        path3 = F.relu(self.conv5x5(F.relu(self.conv1x1_5x5(x))))
        path4 = F.relu(self.pool_proj(self.pool(x)))
        out = torch.cat((path1, path2, path3, path4), dim=1)
        out = self.batch_norm(out)
        return out

# Final Model with Transformer Branch
class FinalModel(nn.Module):
    def __init__(self, dropout_rate=0.3, embedding_dim=1280):
        super(FinalModel, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings=4, embedding_dim=32)
        
        # Inception blocks with progressively more filters
        self.inception1 = InceptionBlock(filters=32, input_channels=32)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.inception2 = InceptionBlock(filters=64, input_channels=32 * 4)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.inception3 = InceptionBlock(filters=128, input_channels=64 * 4)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()

         # Fully connected layers
        self.inception_fc1 = nn.Linear(512 * 4, 256)  # Dynamically set filters * 4
        self.inception_bn1 = nn.BatchNorm1d(256)
        self.inception_relu1 = nn.ReLU()
        self.inception_dropout1 = nn.Dropout(0.5)

        # Transformer block
        self.transformer_block = TransformerBlock(d_model=embedding_dim)
        self.pool_transformer = nn.MaxPool1d(kernel_size=3, stride=2)
        self.flatten_transformer = nn.Flatten()
        self.fc_transformer1 = nn.Linear(41*639, 512)
        self.bn_transformer1 = nn.BatchNorm1d(512)
        self.relu_transformer1 = nn.ReLU()
        self.dropout_transformer1 = nn.Dropout(dropout_rate)
        self.fc_transformer2 = nn.Linear(512, 256)
        self.bn_transformer2 = nn.BatchNorm1d(256)
        self.relu_transformer2 = nn.ReLU()
        self.dropout_transformer2 = nn.Dropout(dropout_rate)

        # Concatenation of Inception and Transformer branches
        self.fc_concat = nn.Linear(256 + 256, 128)
        self.bn_concat = nn.BatchNorm1d(128)
        self.relu_concat = nn.ReLU()
        self.dropout_concat = nn.Dropout(dropout_rate)
        
        # Output layer
        self.fc_out = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

        

    def forward(self, rna_embed, word_embed):
        # Embedding and Inception block processing
        x = self.embedding(word_embed)
        x = x.permute(0, 2, 1)
        x = self.inception1(x)
        x = self.pool1(x)
        x = self.inception2(x)
        x = self.pool2(x)
        x = self.inception3(x)
        x = self.pool3(x)
        x = self.flatten(x)

        # Fully connected layers for Inception branch
        x = self.inception_fc1(x)
        x = self.inception_bn1(x)
        x = self.inception_relu1(x)
        x = self.inception_dropout1(x)

        # Transformer block processing
        x1 = self.transformer_block(rna_embed)
        x1 = self.pool_transformer(x1)
        x1 = self.flatten_transformer(x1)

        x1 = self.fc_transformer1(x1)
        x1 = self.bn_transformer1(x1)
        x1 = self.relu_transformer1(x1)
        x1 = self.dropout_transformer1(x1)
        x1 = self.fc_transformer2(x1)
        x1 = self.bn_transformer2(x1)
        x1 = self.relu_transformer2(x1)
        x1 = self.dropout_transformer2(x1)
        
        # Concatenate Inception and Transformer branches
        z = torch.cat((x, x1), dim=1)
        z = self.fc_concat(z)
        z = self.bn_concat(z)
        z = self.relu_concat(z)
        z = self.dropout_concat(z)

        # Output layer
        x = self.fc_out(z)
        x = self.sigmoid(x)
        
        return x


word_embed = np.load('all_sequences.npy')
rna_embed = np.load('RiNALMo.npy')
rna_embed = rna_embed[:, 1:-1, :]  # Remove start and end tokens

print(f"Shape of word embeddings: {word_embed.shape}")
print(f"Shape of RNA embeddings: {rna_embed.shape}")

y = np.array([1]*662 + [0]*662)

X_train_indices = pd.read_csv("train_indices.txt", header=None)
X_test_indices = pd.read_csv("test_indices.txt", header=None)

X_train_indices = X_train_indices.values.flatten()
X_test_indices = X_test_indices.values.flatten()

word_embed_train = word_embed[X_train_indices]
word_embed_test = word_embed[X_test_indices]

rna_embed_train = rna_embed[X_train_indices]
rna_embed_test = rna_embed[X_test_indices]
y_train = y[X_train_indices]
y_test = y[X_test_indices]
print(f"Shape of word embeddings: {word_embed_train.shape}")
print(f"Shape of RNA embeddings: {rna_embed_train.shape}")
print(f"Shape of word embeddings: {word_embed_test.shape}")
print(f"Shape of RNA embeddings: {rna_embed_test.shape}")
print(f"Class distribution in y: {np.unique(y_train, return_counts=True)}")
print(f"Class distribution in y: {np.unique(y_test, return_counts=True)}")

X_train = list(zip(rna_embed_train, word_embed_train))
X_test = list(zip(rna_embed_test, word_embed_test))

# Convert to tensors
train_data = TensorDataset(torch.tensor(np.array([x[0] for x in X_train]), dtype=torch.float32),
                           torch.tensor(np.array([x[1] for x in X_train]), dtype=torch.long),
                           torch.tensor(y_train, dtype=torch.float32))

test_data = TensorDataset(torch.tensor(np.array([x[0] for x in X_test]), dtype=torch.float32),
                         torch.tensor(np.array([x[1] for x in X_test]), dtype=torch.long),
                         torch.tensor(y_test, dtype=torch.float32))

# DataLoader for batching
train_loader = DataLoader(train_data, batch_size=500, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

# Initialize model
model = FinalModel()

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.00001)
criterion = nn.BCELoss()  # For binary classification

# Training loop
epochs = 18

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for rna_batch, word_batch, label_batch in train_loader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(rna_batch, word_batch)
        loss = criterion(outputs.squeeze(), label_batch)

        # Backward pass
        loss.backward()

        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Update metrics
        running_loss += loss.item()
        preds = (outputs.squeeze() > 0.5).float()
        correct_preds += (preds == label_batch).sum().item()
        total_preds += label_batch.size(0)

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct_preds / total_preds

    # test
    model.eval()
    test_loss = 0.0
    test_correct_preds = 0
    test_total_preds = 0

    with torch.no_grad():
        y_test = []
        y_pred = []

        for rna_batch, word_batch, label_batch in test_loader:
            outputs = model(rna_batch, word_batch)
            loss = criterion(outputs.squeeze(), label_batch)

            test_loss += loss.item()
            preds = (outputs.squeeze() > 0.5).float()
            test_correct_preds += (preds == label_batch).sum().item()
            test_total_preds += label_batch.size(0)
            # Collect predictions and labels for metrics
            y_test.extend(label_batch.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())  # Convert tensors to lists

    print(f"Epoch [{epoch+1}/{epochs}]")

torch.save(model.state_dict(), 'model.pth')
