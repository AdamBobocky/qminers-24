import math
import torch
import copy
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
from datetime import datetime

class PlayerRatingModel(nn.Module):
    def __init__(self):
        super(PlayerRatingModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(19, 12),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(12, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
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
            nn.Linear(8, 12),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(12, 1)
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

class NeuralNetwork:
    def __init__(self, elo):
        self.elo = elo

        self.INPUTS_DIM = 19

        self.model = GameRatingModel().to(torch.device('cpu'))
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.003, weight_decay=2e-5)
        self.loss_fn = nn.MSELoss()
        self.retrain_countdown = 0
        self.first_training = True

        self.team_rosters = {}
        self.player_data = defaultdict(list)

        self.home_inputs = np.empty((30000, 12, 26, self.INPUTS_DIM + 1), np.float32)
        self.away_inputs = np.empty((30000, 12, 26, self.INPUTS_DIM + 1), np.float32)
        self.home_playtimes = []
        self.away_playtimes = []
        self.outputs = []
        self.training_data = 0

    def _row_to_inputs(self, row, am_home, my_id, opponent_id, season):
        return [
            self.elo.get_team_strength(my_id, am_home, season) / 100,
            self.elo.get_team_strength(opponent_id, not am_home, season) / 100,
            1 if am_home else 0,        # Whether player is part of home team
            row['MIN'],
            row['FGM'],

            row['FGA'],
            row['FG3M'],
            row['FG3A'],
            row['FTM'],
            row['FTA'],

            row['ORB'],
            row['DRB'],
            row['RB'],
            row['AST'],
            row['STL'],

            row['BLK'],
            row['TOV'],
            row['PF'],
            row['PTS']
        ]

    def _get_team_roster(self, season, team_id, date):
        rosters = self.team_rosters[season][team_id][-5:]

        roster = defaultdict(int)

        for c_roster in rosters:
            for pid, mins in c_roster:
                roster[pid] += mins

        roster = sorted(roster.items(), key=lambda x: x[1], reverse=True)[:12]

        while len(roster) < 12:
            roster.append([-1, 0])

        total_mins = sum(x[1] for x in roster)

        return roster, total_mins

    def _get_game_frame(self, season, date, home_id, away_id):
        season_valid = season in self.team_rosters
        home_valid = season_valid and home_id in self.team_rosters[season] and len(self.team_rosters[season][home_id]) >= 5
        away_valid = season_valid and away_id in self.team_rosters[season] and len(self.team_rosters[season][away_id]) >= 5

        if season_valid and home_valid and away_valid:
            home_roster, home_total_mins = self._get_team_roster(season, home_id, date)
            away_roster, away_total_mins = self._get_team_roster(season, away_id, date)

            if home_total_mins >= 500 and away_total_mins >= 500:
                c_home_inputs = []
                c_home_playtimes = []
                c_away_inputs = []
                c_away_playtimes = []

                for pid, mins in home_roster:
                    c_player_data = []

                    if pid != -1 and pid in self.player_data:
                        c_player_data = copy.deepcopy(self.player_data[pid][-26:])

                    for i in range(len(c_player_data)):
                        point_date, point_mins = c_player_data[i][0]
                        time_weight = 0.9965 ** abs((date - point_date).days)
                        c_player_data[i][0] = round(point_mins * time_weight, 3) # Apply time decay

                    while len(c_player_data) < 26:
                        c_player_data.append([0] * (self.INPUTS_DIM + 1))

                    c_home_inputs.append(c_player_data)
                    c_home_playtimes.append(mins / home_total_mins)

                for pid, mins in away_roster:
                    c_player_data = []

                    if pid != -1 and pid in self.player_data:
                        c_player_data = copy.deepcopy(self.player_data[pid][-26:])

                    for i in range(len(c_player_data)):
                        point_date, point_mins = c_player_data[i][0]
                        time_weight = 0.9965 ** abs((date - point_date).days)
                        c_player_data[i][0] = round(point_mins * time_weight, 3) # Apply time decay

                    while len(c_player_data) < 26:
                        c_player_data.append([0] * (self.INPUTS_DIM + 1))

                    c_away_inputs.append(c_player_data)
                    c_away_playtimes.append(mins / away_total_mins)

                return c_home_inputs, c_home_playtimes, c_away_inputs, c_away_playtimes

        return None

    def _train(self, dataloader):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, (home_team_stats, away_team_stats, home_game_weights, away_game_weights, home_play_times, away_play_times, true_score_diff) in enumerate(dataloader):
            # Forward pass
            predicted_score_diff = self.model(home_team_stats, away_team_stats, home_game_weights, away_game_weights,
                                        home_play_times, away_play_times)

            # Compute loss
            loss = self.loss_fn(predicted_score_diff, true_score_diff)

            # Apply sample weights: weight decays with distance from the most recent sample
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
            self.optimizer.zero_grad()
            weighted_loss.backward()
            self.optimizer.step()

            # Accumulate loss
            total_loss += weighted_loss.item()

        # Calculate average loss and binary accuracy for this epoch
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    def pre_add_game(self, current, current_players):
        season = current['Season']
        home_id = current['HID']
        away_id = current['AID']
        home_score = current['HSC']
        away_score = current['ASC']
        date = current['Date']

        game_frame = self._get_game_frame(season, date, home_id, away_id)

        if game_frame is not None:
            c_home_inputs, c_home_playtimes, c_away_inputs, c_away_playtimes = game_frame

            self.home_inputs[self.training_data] = np.array(c_home_inputs, np.float32)
            self.away_inputs[self.training_data] = np.array(c_away_inputs, np.float32)
            self.home_playtimes.append(c_home_playtimes)
            self.away_playtimes.append(c_away_playtimes)
            self.outputs.append((abs(home_score - away_score) + 3) ** 0.7 * (1 if home_score > away_score else -1))
            self.retrain_countdown -= 1
            self.training_data += 1

    def add_game(self, current, current_players):
        season = current['Season']
        home_id = current['HID']
        away_id = current['AID']
        date = current['Date']

        home_players = current_players[current_players['Team'] == home_id]
        away_players = current_players[current_players['Team'] == away_id]

        if season not in self.team_rosters:
            self.team_rosters[season] = {}

        if home_id not in self.team_rosters[season]:
            self.team_rosters[season][home_id] = []

        if away_id not in self.team_rosters[season]:
            self.team_rosters[season][away_id] = []

        self.team_rosters[season][home_id].append([[x['Player'], x['MIN']] for _, x in home_players.iterrows()])
        self.team_rosters[season][away_id].append([[x['Player'], x['MIN']] for _, x in away_players.iterrows()])

        mapped_home_players = [{
            'pid': row['Player'],
            'mins': row['MIN'],
            'inputs': self._row_to_inputs(row, True, home_id, away_id, season)
        } for _, row in home_players.iterrows()]
        mapped_away_players = [{
            'pid': row['Player'],
            'mins': row['MIN'],
            'inputs': self._row_to_inputs(row, False, away_id, home_id, season)
        } for _, row in away_players.iterrows()]

        for data in [*mapped_home_players, *mapped_away_players]:
            if not any(math.isnan(x) for x in data['inputs']):
                self.player_data[data['pid']].append([[date, data['mins']], *data['inputs']])

    def get_input_data(self, home_id, away_id, season, date):
        game_frame = self._get_game_frame(season, date, home_id, away_id)

        if game_frame is None:
            return None

        if self.retrain_countdown <= 0:
            self.retrain_countdown = 2500

            print('\nRetraining! Preparing dataset...')

            home_team_stats = torch.from_numpy(self.home_inputs[:self.training_data, :, :, 1:])
            away_team_stats = torch.from_numpy(self.away_inputs[:self.training_data, :, :, 1:])
            home_game_weights = torch.from_numpy(self.home_inputs[:self.training_data, :, :, 0])
            away_game_weights = torch.from_numpy(self.away_inputs[:self.training_data, :, :, 0])
            home_play_times = torch.tensor(np.array(self.home_playtimes).astype(np.float32), dtype=torch.float32)
            away_play_times = torch.tensor(np.array(self.away_playtimes).astype(np.float32), dtype=torch.float32)
            true_score_diff = torch.tensor(np.array(self.outputs), dtype=torch.float32)

            # Prepare DataLoader
            train_data = TensorDataset(home_team_stats, away_team_stats, home_game_weights, away_game_weights, home_play_times, away_play_times, true_score_diff)
            train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

            print('\nRetraining!')

            num_epochs = 40 if self.first_training else 10
            self.first_training = False
            for epoch in range(num_epochs):
                train_loss, train_accuracy = self._train(train_loader)

                print(f'Epoch {epoch + 1} / {num_epochs}, train_loss: {train_loss:.4f}, train_accuracy: {train_accuracy:.4f}; N={self.training_data}')

        c_home_inputs, c_home_playtimes, c_away_inputs, c_away_playtimes = game_frame
        np_array_home_inputs = np.array(c_home_inputs).astype(np.float32)
        np_array_away_inputs = np.array(c_away_inputs).astype(np.float32)

        home_team_stats = torch.tensor(np_array_home_inputs[:, :, 1:], dtype=torch.float32).unsqueeze(0)
        away_team_stats = torch.tensor(np_array_away_inputs[:, :, 1:], dtype=torch.float32).unsqueeze(0)
        home_game_weights = torch.tensor(np_array_home_inputs[:, :, 0], dtype=torch.float32).unsqueeze(0)
        away_game_weights = torch.tensor(np_array_away_inputs[:, :, 0], dtype=torch.float32).unsqueeze(0)
        home_play_times = torch.tensor(np.array(c_home_playtimes).astype(np.float32), dtype=torch.float32).unsqueeze(0)
        away_play_times = torch.tensor(np.array(c_away_playtimes).astype(np.float32), dtype=torch.float32).unsqueeze(0)

        self.model.eval()

        with torch.no_grad():
            prediction = self.model(home_team_stats, away_team_stats, home_game_weights, away_game_weights, home_play_times, away_play_times)

        return [
            prediction.item()
        ]

