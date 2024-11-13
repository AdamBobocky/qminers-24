import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

def inverse_sigmoid(x):
    return np.log(x / (1 - x))

df = pd.read_csv('data/games.csv')

df['Benchmark'] = inverse_sigmoid(1 / df['OddsH'] / (1 / df['OddsH'] + 1 / df['OddsA']))

data_df = df[['HID', 'AID', 'Benchmark', 'H']]

# Step 1: Encode team IDs to integers
all_teams = pd.concat([data_df['HID'], data_df['AID']]).unique()
team_to_idx = {team: idx for idx, team in enumerate(all_teams)}
num_teams = len(team_to_idx)

# Convert team IDs in the DataFrame to integer indices
data_df['HID'] = data_df['HID'].map(team_to_idx)
data_df['AID'] = data_df['AID'].map(team_to_idx)

class NBAOutcomePredictor(nn.Module):
    def __init__(self, num_teams, embedding_dim=8, hidden_dim=16):
        super(NBAOutcomePredictor, self).__init__()

        # Embedding layers for home and away teams
        self.home_embedding = nn.Embedding(num_teams, embedding_dim)
        self.away_embedding = nn.Embedding(num_teams, embedding_dim)

        # Linear layer for combining embeddings and benchmark input
        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        # Sigmoid activation for the output
        self.sigmoid = nn.Sigmoid()

    def forward(self, home_id, away_id, benchmark):
        # Get team embeddings
        home_embedded = self.home_embedding(home_id)
        away_embedded = self.away_embedding(away_id)

        # Concatenate embeddings and benchmark input
        combined = torch.cat([home_embedded, away_embedded], dim=1)

        # Pass through fully connected layers
        x = torch.relu(self.fc1(combined))
        # x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        # Sigmoid to get probability
        outcome = self.sigmoid(x.squeeze() + benchmark)

        return outcome

# Example usage:
num_teams = 30  # Example number of NBA teams
embedding_dim = 4  # Dimensionality of team embeddings
hidden_dim = 6  # Hidden layer size

train_df, val_df = train_test_split(data_df, test_size=0.2, shuffle=False)

def df_to_tensors(df):
    home_id_tensor = torch.tensor(df['HID'].values, dtype=torch.long)
    away_id_tensor = torch.tensor(df['AID'].values, dtype=torch.long)
    benchmark_tensor = torch.tensor(df['Benchmark'].values, dtype=torch.float32)
    outcome_tensor = torch.tensor(df['H'].values, dtype=torch.float32)
    return home_id_tensor, away_id_tensor, benchmark_tensor, outcome_tensor

home_id_train, away_id_train, benchmark_train, outcome_train = df_to_tensors(train_df)
home_id_val, away_id_val, benchmark_val, outcome_val = df_to_tensors(val_df)

# Instantiate model
model = NBAOutcomePredictor(num_teams, embedding_dim, hidden_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 200
batch_size = 32
train_size = len(train_df)
num_batches = (train_size + batch_size - 1) // batch_size  # Calculate the number of batches

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    # Mini-batch training
    for i in range(0, train_size, batch_size):
        # Get batch data
        end_idx = i + batch_size
        home_id_batch = home_id_train[i:end_idx]
        away_id_batch = away_id_train[i:end_idx]
        benchmark_batch = benchmark_train[i:end_idx]
        outcome_batch = outcome_train[i:end_idx]

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(home_id_batch, away_id_batch, benchmark_batch).squeeze()

        # Compute loss
        loss = criterion(predictions, outcome_batch)
        epoch_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    # Calculate average training loss for the epoch
    avg_train_loss = epoch_loss / num_batches

    # Validation phase
    model.eval()
    with torch.no_grad():
        val_predictions = model(home_id_val, away_id_val, benchmark_val).squeeze()
        val_loss = criterion(val_predictions, outcome_val).item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

# BM values: train_loss: 0.2004, val_loss: 0.2057
