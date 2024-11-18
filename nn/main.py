import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

with open('temp/keys.json', 'r') as f:
    keys = json.load(f)

class PlayerRatingModel(nn.Module):
    def __init__(self):
        super(PlayerRatingModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(19, 12),
            nn.ReLU(),
            nn.Linear(12, 6),
            nn.ReLU(),
            nn.Linear(6, 4),
            nn.ReLU()
        )

    def forward(self, player_stats, game_weight):
        player_output = self.layers(player_stats).squeeze()

        weighted_output = player_output * game_weight.unsqueeze(-1)

        return torch.sum(weighted_output, axis=2) / (torch.sum(game_weight, axis=2).unsqueeze(-1) + 0.001)

class GameRatingModel(nn.Module):
    def __init__(self):
        super(GameRatingModel, self).__init__()

        # Instantiate the player rating model
        self.player_model = PlayerRatingModel()
        self.home_field_advantage = nn.Parameter(torch.tensor(4.5))
        self.layers = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, home_team_stats, away_team_stats, home_game_weights, away_game_weights,
                home_play_times, away_play_times):
        home_outputs = self.player_model(home_team_stats, home_game_weights)
        away_outputs = self.player_model(away_team_stats, away_game_weights)

        home_ratings = home_outputs * home_play_times.unsqueeze(-1)
        away_ratings = away_outputs * away_play_times.unsqueeze(-1)

        home_team_rating = torch.sum(home_ratings, axis=1)
        away_team_rating = torch.sum(away_ratings, axis=1)

        x = self.layers(torch.cat((home_team_rating, away_team_rating), dim=-1)).squeeze()

        return x + self.home_field_advantage

def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (home_team_stats, away_team_stats, home_game_weights, away_game_weights, home_play_times, away_play_times, true_score_diff) in enumerate(dataloader):
        # Forward pass
        predicted_score_diff = model(home_team_stats, away_team_stats, home_game_weights, away_game_weights,
                                     home_play_times, away_play_times)

        # Compute loss
        loss = loss_fn(predicted_score_diff, true_score_diff)

        batch_size = true_score_diff.size(0)
        weights = torch.tensor([0.99984 ** (len(dataloader.dataset) - (batch_idx * batch_size + i))
                                for i in range(batch_size)], dtype=torch.float32)
        weights = weights.to(loss.device)
        weighted_loss = (loss * weights).mean()  # Apply weights to loss

        # Calculate binary accuracy (direction match)
        predicted_sign = torch.sign(predicted_score_diff)
        true_sign = torch.sign(true_score_diff)
        correct = (predicted_sign == true_sign).sum().item()
        total_correct += correct
        total_samples += true_score_diff.size(0)

        # Backpropagation and optimization
        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()

        # Accumulate loss
        total_loss += weighted_loss.item()

    # Calculate average loss and binary accuracy for this epoch
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    # predictions = []
    with torch.no_grad():
        for home_team_stats, away_team_stats, home_game_weights, away_game_weights, home_play_times, away_play_times, true_score_diff in dataloader:
            predicted_score_diff = model(home_team_stats, away_team_stats, home_game_weights, away_game_weights,
                                         home_play_times, away_play_times)
            loss = loss_fn(predicted_score_diff, true_score_diff)

            # Calculate binary accuracy (direction match)
            predicted_sign = torch.sign(predicted_score_diff)
            true_sign = torch.sign(true_score_diff)
            correct = (predicted_sign == true_sign).sum().item()
            total_correct += correct
            total_samples += true_score_diff.size(0)

            total_loss += loss.item()
            # predictions.extend(predicted_score_diff.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy # , predictions

# Training setup
device = torch.device('cpu')
model = GameRatingModel().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.0001) # , weight_decay=2e-5)
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
split_index = int(0.85 * len(home_team_stats))

# Training loop
num_epochs = 40

train_data = TensorDataset(home_team_stats[:split_index], away_team_stats[:split_index],
                        home_game_weights[:split_index], away_game_weights[:split_index],
                        home_play_times[:split_index], away_play_times[:split_index],
                        true_score_diff[:split_index])
val_data = TensorDataset(home_team_stats[split_index:], away_team_stats[split_index:],
                        home_game_weights[split_index:], away_game_weights[split_index:],
                        home_play_times[split_index:], away_play_times[split_index:],
                        true_score_diff[split_index:])
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

for epoch in range(num_epochs):
    train_loss, train_accuracy = train(model, train_loader, optimizer, loss_fn, device)
    # val_loss, val_accuracy, val_predictions = validate(model, val_loader, loss_fn, device)
    val_loss, val_accuracy = validate(model, val_loader, loss_fn, device)
    print(f'Epoch {epoch+1} / {num_epochs}, train_loss: {train_loss:.4f}, train_accuracy: {train_accuracy:.4f}, val_loss: {val_loss:.4f}, val_accuracy: {val_accuracy:.4f}')

# validation_results = {key: float(pred) for key, pred in zip(keys[split_index:], val_predictions)}

# with open('temp/predictions.json', 'w') as json_file:
#     json.dump(validation_results, json_file)

state_dict = model.state_dict()
state_dict_json = {k: v.tolist() for k, v in state_dict.items()}  # Convert tensors to lists

# Save to a JSON file
with open("pretrained_model.json", "w") as f:
    json.dump(state_dict_json, f, indent=2)
