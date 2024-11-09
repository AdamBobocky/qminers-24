import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset

enabled_log = False

class NBAGameDataset(Dataset):
    def __init__(self, home_team_stats, away_team_stats, home_game_weights, away_game_weights,
                 home_play_times, away_play_times, true_score_diff):
        """
        home_team_stats: Tensor of shape [num_games, 10, 50, 18] for home team stats (10 players, 50 games, 18 features)
        away_team_stats: Tensor of shape [num_games, 10, 50, 18] for away team stats (same dimensions as home_team_stats)
        home_game_weights: Tensor of shape [num_games, 10, 50] for game weights
        away_game_weights: Tensor of shape [num_games, 10, 50] for game weights
        home_play_times: Tensor of shape [num_games, 10] for expected play time for home players
        away_play_times: Tensor of shape [num_games, 10] for expected play time for away players
        true_score_diff: Tensor of shape [num_games, 1] for the actual score difference (home_score - away_score)
        """
        self.home_team_stats = home_team_stats
        self.away_team_stats = away_team_stats
        self.home_game_weights = home_game_weights
        self.away_game_weights = away_game_weights
        self.home_play_times = home_play_times
        self.away_play_times = away_play_times
        self.true_score_diff = true_score_diff

    def __len__(self):
        return len(self.true_score_diff)

    def __getitem__(self, idx):
        # Get data for a specific game
        return {
            'home_team_stats': self.home_team_stats[idx],
            'away_team_stats': self.away_team_stats[idx],
            'home_game_weights': self.home_game_weights[idx],
            'away_game_weights': self.away_game_weights[idx],
            'home_play_times': self.home_play_times[idx],
            'away_play_times': self.away_play_times[idx],
            'true_score_diff': self.true_score_diff[idx]
        }

class PlayerRatingModel(nn.Module):
    def __init__(self):
        super(PlayerRatingModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(15, 32),     # Input layer with 64 neurons
            nn.ReLU(),
            nn.Linear(32, 16),     # Hidden layer with 16 neurons
            nn.ReLU(),
            nn.Linear(16, 1)       # Output layer to produce a single rating value
        )

    def forward(self, player_stats, game_weight):
        player_output = self.layers(player_stats).squeeze()
        weighted_output = player_output * game_weight

        # weighted_output = self.linear(player_stats).squeeze() * game_weight
        return torch.sum(weighted_output) / (torch.sum(game_weight) + 0.001)

class GameRatingModel(nn.Module):
    def __init__(self):
        super(GameRatingModel, self).__init__()

        # Instantiate the player rating model
        self.player_model = PlayerRatingModel()
        self.home_field_advantage = nn.Parameter(torch.tensor(3.0))

    def forward(self, home_team_stats, away_team_stats, home_game_weights, away_game_weights,
                home_play_times, away_play_times):
        # Adjust shapes by removing leading dimension

        home_outputs = torch.stack([self.player_model(home_team_stats[:, i], home_game_weights[:, i]) for i in range(15)])
        away_outputs = torch.stack([self.player_model(away_team_stats[:, i], away_game_weights[:, i]) for i in range(15)])

        home_ratings = home_outputs * home_play_times
        away_ratings = away_outputs * away_play_times

        home_team_rating = torch.sum(home_ratings, axis=1)
        away_team_rating = torch.sum(away_ratings, axis=1)

        score_diff = home_team_rating - away_team_rating

        return score_diff + self.home_field_advantage

# Define the training function
def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:
        # Move batch data to the device
        home_team_stats = batch['home_team_stats'].to(device)
        away_team_stats = batch['away_team_stats'].to(device)
        home_game_weights = batch['home_game_weights'].to(device)
        away_game_weights = batch['away_game_weights'].to(device)
        home_play_times = batch['home_play_times'].to(device)
        away_play_times = batch['away_play_times'].to(device)
        true_score_diff = batch['true_score_diff'].to(device)

        # Forward pass
        predicted_score_diff = model(home_team_stats, away_team_stats, home_game_weights, away_game_weights,
                                     home_play_times, away_play_times)

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

# Training setup
device = torch.device('cpu')
model = GameRatingModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

np_home_team_stats = np.load('temp/nn_home_inputs.npy')
np_away_team_stats = np.load('temp/nn_away_inputs.npy')
np_home_play_times = np.load('temp/nn_home_playtimes.npy')
np_away_play_times = np.load('temp/nn_away_playtimes.npy')
np_true_score_diff = np.load('temp/nn_outputs.npy')

home_team_stats = torch.tensor(np_home_team_stats[:, :, :, 1:], dtype=torch.float32)
away_team_stats = torch.tensor(np_away_team_stats[:, :, :, 1:], dtype=torch.float32)
home_game_weights = torch.tensor(np_home_team_stats[:, :, :, 0], dtype=torch.float32)
away_game_weights = torch.tensor(np_away_team_stats[:, :, :, 0], dtype=torch.float32)
home_play_times = torch.tensor(np_home_play_times, dtype=torch.float32)
away_play_times = torch.tensor(np_away_play_times, dtype=torch.float32)
true_score_diff = torch.tensor(np_true_score_diff, dtype=torch.float32)

# Prepare DataLoader
train_dataset = NBAGameDataset(home_team_stats, away_team_stats, home_game_weights, away_game_weights,
                                home_play_times, away_play_times, true_score_diff)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Training loop
num_epochs = 10000
for epoch in range(num_epochs):
    if epoch > 4:
        enabled_log = True

    train_loss, train_accuracy = train(model, train_loader, optimizer, loss_fn, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, {train_accuracy:.4f}")
