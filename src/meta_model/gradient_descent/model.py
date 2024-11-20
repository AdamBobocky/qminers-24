import math
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import softplus
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from collections import defaultdict

class GradientDescent:
    def __init__(self, num_teams=30, monthly_decay=0.8):
        self.num_teams = num_teams
        self.monthly_decay = monthly_decay

        self.team_mus = nn.Parameter(torch.zeros(num_teams))
        self.team_sigmas = torch.ones(num_teams) * 40
        self.home_advantage = nn.Parameter(torch.tensor(5.0))
        self.sigma = nn.Parameter(torch.tensor(12.0))

        self.games = torch.zeros((50000, 5), dtype=torch.long)
        self.game_count = 0

        self.my_team_id = {}
        self.last_season = -1
        self.fit_date = None

    def _check_season(self, season):
        if self.last_season != season:
            self.last_season = season

            self.team_sigmas = torch.ones_like(self.team_sigmas) * 20

    def _get_time_weights(self):
        timestamps = self.games[:self.game_count, 0].to(torch.float32)
        last_ts = timestamps[-1]

        return (self.monthly_decay ** (torch.abs(timestamps - last_ts) / (30 * 24 * 60 * 60 * 1000))).to(timestamps.device)

    def forward(self, weights, idx_start):
        games = self.games[idx_start:self.game_count]
        home_ratings = self.team_mus[games[:, 1]]
        away_ratings = self.team_mus[games[:, 2]]

        expectations_home = self.home_advantage + home_ratings - away_ratings
        realities_home = games[:, 3] - games[:, 4]

        game_sigmas = torch.sqrt(
            self.team_sigmas[games[:, 1]] ** 2 +
            self.team_sigmas[games[:, 2]] ** 2 +
            self.sigma ** 2
        )

        log_value = (-0.5 * ((realities_home - expectations_home) / game_sigmas) ** 2 - torch.log(game_sigmas) - 0.5 * torch.log(torch.tensor(2 * np.pi)))
        return torch.sum(log_value * weights[idx_start:])

    def _fit(self, max_epochs=30):
        weights = self._get_time_weights()
        idx_start = torch.nonzero(weights > 0.004, as_tuple=True)[0][0].item()
        optimizer = optim.Adam([self.team_mus, self.sigma, self.home_advantage], lr=0.02)

        for epoch in range(max_epochs):
            optimizer.zero_grad()
            loss = -self.forward(weights, idx_start)
            loss.backward()
            optimizer.step()

    def add_game(self, current, current_players):
        self._check_season(current['Season'])

        timestamp = int(current['Date'].timestamp() * 1000)
        team_home = self._map_team_id(current['HID'])
        team_away = self._map_team_id(current['AID'])
        score_home = current['HSC']
        score_away = current['ASC']

        self.games[self.game_count] = torch.tensor(
            [timestamp, team_home, team_away, score_home, score_away], dtype=torch.long
        )
        self.game_count += 1

        game_sigma2 = math.sqrt(
            self.team_sigmas[team_home] ** 2 +
            self.team_sigmas[team_away] ** 2 +
            self.sigma ** 2
        )

        self.team_sigmas[team_home] = 1 / torch.sqrt(
            1 / self.team_sigmas[team_home] ** 2 + 1 / game_sigma2)
        self.team_sigmas[team_away] = 1 / torch.sqrt(
            1 / self.team_sigmas[team_away] ** 2 + 1 / game_sigma2)

        self.fit_date = None

    def _map_team_id(self, team_id):
        if team_id not in self.my_team_id:
            self.my_team_id[team_id] = len(self.my_team_id)

        return self.my_team_id[team_id]

    def pre_add_game(self, current, current_players):
        pass

    def get_input_data(self, home_id, away_id, season, date):
        self._check_season(season)

        if self.game_count < 2000:
            return None

        if self.fit_date is None or self.fit_date != date:
            self.fit_date = date
            self._fit()

        team_home = self._map_team_id(home_id)
        team_away = self._map_team_id(away_id)

        game_exp = self.home_advantage + self.team_mus[team_home] - self.team_mus[team_away]
        game_sigma = torch.sqrt(
            self.team_sigmas[team_home] ** 2 +
            self.team_sigmas[team_away] ** 2 +
            self.sigma ** 2
        )

        return [game_exp.item() / game_sigma.item()]
