import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

with open('temp/keys.json', 'r') as f:
    keys = json.load(f)

class GameRatingModel(nn.Module):
    def __init__(self):
        super(GameRatingModel, self).__init__()

        self.home_field_advantage = nn.Parameter(torch.tensor(4.5))

        self.layers = nn.Sequential(
            nn.Linear(38, 6),
            # nn.ReLU(),
            # nn.Linear(12, 8),
            # nn.ReLU(),
            # nn.Linear(8, 5)
        )

        self.rnn = nn.LSTM(input_size=6, hidden_size=6, num_layers=1, batch_first=True)

        # self.end_layers = nn.Sequential(
        #     nn.Linear(12, 8),
        #     nn.ReLU(),
        #     nn.Linear(8, 1)
        # )
        self.end_layers = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, home_team_stats, away_team_stats):
        home_outputs = self.layers(home_team_stats)
        away_outputs = self.layers(away_team_stats)

        home_rnn_out, home_hidden = self.rnn(home_outputs)
        away_rnn_out, away_hidden = self.rnn(away_outputs)

        home_final_output = home_rnn_out[:, -1, :]
        away_final_output = away_rnn_out[:, -1, :]

        return self.end_layers(torch.cat((home_final_output, away_final_output), dim=-1)).squeeze() + self.home_field_advantage

def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for home_team_stats, away_team_stats, true_score_diff in dataloader:
        # Forward pass
        predicted_score_diff = model(home_team_stats, away_team_stats)

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
        for home_team_stats, away_team_stats, true_score_diff in dataloader:
            predicted_score_diff = model(home_team_stats, away_team_stats)
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
loss_fn = nn.MSELoss()

np_home_team_stats = np.load('temp/team_nn_home_inputs.npy')
np_away_team_stats = np.load('temp/team_nn_away_inputs.npy')
np_true_score_diff = np.load('temp/team_nn_outputs.npy')

home_team_stats = torch.tensor(np_home_team_stats, dtype=torch.float32)
away_team_stats = torch.tensor(np_away_team_stats, dtype=torch.float32)
true_score_diff = torch.tensor(np_true_score_diff, dtype=torch.float32)

# Prepare DataLoader
split_index = int(0.7 * len(home_team_stats))

# Training loop
num_epochs = 100

train_data = TensorDataset(home_team_stats[:split_index], away_team_stats[:split_index],
                        true_score_diff[:split_index])
val_data = TensorDataset(home_team_stats[split_index:], away_team_stats[split_index:],
                        true_score_diff[split_index:])
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

model = GameRatingModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=2e-5)

for epoch in range(num_epochs):
    train_loss, train_accuracy = train(model, train_loader, optimizer, loss_fn, device)
    val_loss, val_accuracy, val_predictions = validate(model, val_loader, loss_fn, device)
    print(f'Epoch {epoch+1} / {num_epochs}, train_loss: {train_loss:.4f}, train_accuracy: {train_accuracy:.4f}, val_loss: {val_loss:.4f}, val_accuracy: {val_accuracy:.4f}')

validation_results = {key: float(pred) for key, pred in zip(keys[split_index:], val_predictions)}

with open('temp/team_predictions.json', 'w') as json_file:
    json.dump(validation_results, json_file)
