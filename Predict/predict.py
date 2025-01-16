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
import csv

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
    

# Initialize model
model = FinalModel()
model.load_state_dict(torch.load('model.pth'))

# test
model.eval()

rna_embed = np.load("RiNALMo.npy")[:, 1:-1, :]
word_embed = np.loadtxt("word_embedding.csv", delimiter=",")

print(rna_embed.shape)
print(word_embed.shape)

rna_embed = torch.tensor(rna_embed)
word_embed = torch.tensor(word_embed, dtype=torch.long)

outputs = model(rna_embed, word_embed)
outputs = (outputs > 0.5).float()
y_pred = outputs.detach().numpy()

print(y_pred)

# Save predictions to CSV
with open("predictions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Prediction"])
    for pred in y_pred:
        writer.writerow([pred[0]])