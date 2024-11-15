import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

enabled_log = False

with open('temp/keys.json', 'r') as f:
    keys = json.load(f)

class PlayerRatingModel(nn.Module):
    def __init__(self,
                 feature_size: int,
                 num_heads: int = 4,
                 hidden_dim: int = 12,
                 num_layers: int = 3,
                 lstm_hidden_dim: int = 8,
                 num_lstm_layers: int = 1,
                 dropout: float = 0.1):
        super(PlayerRatingModel, self).__init__()

        # Transformer layers for each player
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_size,  # Input feature dimension
            nhead=num_heads,       # Number of attention heads
            dim_feedforward=hidden_dim,  # Feedforward layer hidden dimension
            dropout=dropout        # Dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, num_layers=num_layers  # Number of transformer layers
        )

        # LSTM layer to capture long-term trends
        self.lstm = nn.LSTM(
            input_size=feature_size,   # Each game's features are passed in as the LSTM input
            hidden_size=lstm_hidden_dim,  # Size of LSTM hidden state
            num_layers=num_lstm_layers,  # Number of LSTM layers
            batch_first=True,          # The input and output tensors are provided as (batch_size, seq_len, features)
            dropout=dropout            # Dropout between LSTM layers
        )

        # Final linear layer to output a single player strength per player
        self.fc = nn.Linear(lstm_hidden_dim, 1)  # Reduce LSTM output to a scalar value for each player

    def forward(self, x):
        print('work...')
        # x: shape [batch_size, player_count, games_per_player, feature_size]

        batch_size, player_count, games_per_player, feature_size = x.shape

        # Reshape the input for the transformer: [batch_size * player_count, games_per_player, feature_size]
        x_reshaped = x.view(batch_size * player_count, games_per_player, feature_size)

        # Apply Transformer to each player's sequence of games
        transformer_out = self.transformer_encoder(x_reshaped)

        # Pass transformer output to LSTM
        lstm_out, (h_n, c_n) = self.lstm(transformer_out)

        # Get the final output of LSTM for each player (last hidden state)
        # We use the last LSTM output (from the last game)
        lstm_out_last = lstm_out[:, -1, :]  # Get the last timestep for each player

        # Pass through a final fully connected layer to get player strength
        player_strength = self.fc(lstm_out_last).squeeze(-1)  # shape [batch_size * player_count]

        # Reshape back to [batch_size, player_count]
        return player_strength.view(batch_size, player_count)

class GameRatingModel(nn.Module):
    def __init__(self):
        super(GameRatingModel, self).__init__()

        self.player_model = PlayerRatingModel(20)
        self.home_field_advantage = nn.Parameter(torch.tensor(4.5))

    def forward(self, home_team_stats, away_team_stats, home_play_times, away_play_times):
        home_outputs = self.player_model(home_team_stats)
        away_outputs = self.player_model(away_team_stats)

        home_ratings = home_outputs * home_play_times
        away_ratings = away_outputs * away_play_times

        home_team_rating = torch.sum(home_ratings, axis=1)
        away_team_rating = torch.sum(away_ratings, axis=1)

        score_diff = home_team_rating - away_team_rating

        return score_diff + self.home_field_advantage

def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for home_team_stats, away_team_stats, home_play_times, away_play_times, true_score_diff in dataloader:
        # Forward pass
        predicted_score_diff = model(home_team_stats, away_team_stats, home_play_times, away_play_times)

        # Compute loss
        loss = loss_fn(predicted_score_diff, true_score_diff)

        # Calculate binary accuracy (direction match)
        predicted_sign = torch.sign(predicted_score_diff)
        true_sign = torch.sign(true_score_diff)
        correct = (predicted_sign == true_sign).sum().item()
        total_correct += correct
        total_samples += true_score_diff.size(0)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()

    # Calculate average loss and binary accuracy for this epoch
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    predictions = []
    with torch.no_grad():
        for home_team_stats, away_team_stats, home_play_times, away_play_times, true_score_diff in dataloader:
            predicted_score_diff = model(home_team_stats, away_team_stats, home_play_times, away_play_times)
            loss = loss_fn(predicted_score_diff, true_score_diff)

            # Calculate binary accuracy (direction match)
            predicted_sign = torch.sign(predicted_score_diff)
            true_sign = torch.sign(true_score_diff)
            correct = (predicted_sign == true_sign).sum().item()
            total_correct += correct
            total_samples += true_score_diff.size(0)

            total_loss += loss.item()
            predictions.extend(predicted_score_diff.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy, predictions

# Training setup
device = torch.device('cpu')
model = GameRatingModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=2e-5)
loss_fn = nn.MSELoss()
val_loss_fn = nn.MSELoss()

np_home_team_stats = np.load('temp/nn_home_inputs.npy')
np_away_team_stats = np.load('temp/nn_away_inputs.npy')
np_home_play_times = np.load('temp/nn_home_playtimes.npy')
np_away_play_times = np.load('temp/nn_away_playtimes.npy')
np_true_score_diff = np.load('temp/nn_outputs.npy')

home_team_stats = torch.tensor(np_home_team_stats, dtype=torch.float32)
away_team_stats = torch.tensor(np_away_team_stats, dtype=torch.float32)
home_play_times = torch.tensor(np_home_play_times, dtype=torch.float32)
away_play_times = torch.tensor(np_away_play_times, dtype=torch.float32)
true_score_diff = torch.tensor(np_true_score_diff, dtype=torch.float32)

# Prepare DataLoader
split_index = int(0.7 * len(home_team_stats))
train_data = TensorDataset(home_team_stats[:split_index], away_team_stats[:split_index],
                           home_play_times[:split_index], away_play_times[:split_index],
                           true_score_diff[:split_index])
val_data = TensorDataset(home_team_stats[split_index:], away_team_stats[split_index:],
                         home_play_times[split_index:], away_play_times[split_index:],
                         true_score_diff[split_index:])
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    if epoch > 4:
        enabled_log = True

    train_loss, train_accuracy = train(model, train_loader, optimizer, loss_fn, device)
    val_loss, val_accuracy, val_predictions = validate(model, val_loader, val_loss_fn, device)
    print(f'Epoch {epoch+1} / {num_epochs}, train_loss: {train_loss:.4f}, train_accuracy: {train_accuracy:.4f}, val_loss: {val_loss:.4f}, val_accuracy: {val_accuracy:.4f}')

validation_results = {key: float(pred) for key, pred in zip(keys[split_index:], val_predictions)}

with open('temp/predictions.json', 'w') as json_file:
    json.dump(validation_results, json_file)
