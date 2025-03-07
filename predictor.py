import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torchvision
from torchvision import models
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from keras.models import load_model

# Import with pandas
df = pd.read_csv('vegetable_prices.csv', parse_dates=['Date'], index_col="Date")

# Check if 'Commodity' column exists before proceeding
if 'Commodity' not in df.columns:
    print("Error: 'Commodity' column not found in dataset, not saving cleaned data.")
else:
    # Commodity is string value
    df['Commodity'] = df['Commodity'].astype(str)
    unique_commodities = df['Commodity'].unique()
    commodity_dict = {i: commodity for i, commodity in enumerate(unique_commodities)}
    
    print("Available commodities:")
    for index, commodity in commodity_dict.items():
        print(f"{index}: {commodity}")
    
    # Allow user to select a commodity by number
    try:
        commodity_index = int(input("Enter the number corresponding to the commodity you want to analyze: "))
        if commodity_index not in commodity_dict:
            print("Error: Selected number not found in dataset.")
        else:
            commodity = commodity_dict[commodity_index]
            df = df[df.Commodity == commodity].copy()
            df.drop(['Commodity'], axis=1, inplace=True)
    except ValueError:
        print("Error: Please enter a valid number.")

df.ffill(inplace=True)

df['prev_max_price'] = df['Maximum'].shift(1)
df['prev_min_price'] = df['Minimum'].shift(1)
df['prev_avg_price'] = df['Average'].shift(1)

df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
df['year'] = df.index.year

df.dropna(inplace=True)

scaler = MinMaxScaler()
numerical_features = ['Maximum', 'Minimum', 'Average', 'prev_max_price', 'prev_min_price', 'prev_avg_price']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in tqdm(range(len(data) - seq_length)):
        x = data.iloc[i:(i + seq_length)].drop(columns=['Maximum', 'Minimum', 'Average']).values.astype(np.float32)
        y = data.iloc[i + seq_length][['Maximum', 'Minimum', 'Average']].values.astype(np.float32)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys, dtype=np.float32)

seq_length = 30
X, y = create_sequences(df, seq_length)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, seq_length):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.BatchNorm1d(64)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(64 * (seq_length // 4), 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, out_channels)
        )
    
    def forward(self, x):
        x = self.convolution(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

X_train = torch.tensor(X_train.astype(np.float32), dtype=torch.float32).permute(0, 2, 1)
X_test = torch.tensor(X_test.astype(np.float32), dtype=torch.float32).permute(0, 2, 1)
y_train = torch.tensor(y_train.astype(np.float32), dtype=torch.float32)
y_test = torch.tensor(y_test.astype(np.float32), dtype=torch.float32)

model = ConvNet(in_channels=X_train.shape[1], out_channels=3, seq_length=seq_length)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete")

model.eval()
with torch.no_grad():
    predictions = model(X_test)

predicted_prices = predictions.detach().numpy()
y_test_np = y_test.numpy()

# Plot actual vs. predicted prices
plt.figure(figsize=(12, 6))
plt.plot(y_test_np[:, 0], label='Actual Max Price', color='blue')
plt.plot(predicted_prices[:, 0], label='Predicted Max Price', color='red', linestyle='--')
plt.title(f'Actual vs Predicted Max Price for {commodity}')
plt.xlabel('Index')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))