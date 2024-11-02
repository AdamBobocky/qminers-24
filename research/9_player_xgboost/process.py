import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from collections import defaultdict
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

players_df = pd.read_csv('data/players.csv')
games_df = pd.read_csv('data/games.csv')

player_stats_average = {} # Dict where first index is [season] and second is [player_id] and then its list of performances

indices = []
frames_X = []
frames_y = []

total = len(players_df)

for i in players_df.index:
  current = players_df.loc[i]

  game_id = current['Game']
  season = current['Season']
  team_id = current['Team']
  player_id = current['Player']

  if season not in player_stats_average:
    player_stats_average[season] = {}
  if player_id not in player_stats_average[season]:
    player_stats_average[season][player_id] = []

  matching_game = games_df.loc[game_id]
  home_id = matching_game['HID']
  home_score = matching_game['HSC']
  away_score = matching_game['ASC']

  player_is_home_team = home_id == team_id

  player_score_delta = (home_score - away_score) if player_is_home_team else (away_score - home_score)

  if len(player_stats_average[season][player_id]) >= 24:
    # print(player_stats_average[season][player_id])
    df = pd.DataFrame(player_stats_average[season][player_id])
    inputs = df[['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'ORB', 'DRB', 'RB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].mean().to_numpy()
    target = player_score_delta

    indices.append([game_id, player_is_home_team])
    frames_X.append(inputs)
    frames_y.append(target)

  # Add this game stats to the player
  player_stats_average[season][player_id].append(current)

  if (i + 1) % (total // 1000) == 0:
    progress = (i + 1) / total * 100
    print(f"Progress: {progress:.1f}%")

np_indices = np.array(indices)
np_frames_X = np.array(frames_X)
np_frames_y = np.array(frames_y)

np.save('temp/np_indices.npy', np_indices)
np.save('temp/np_frames_X.npy', np_frames_X)
np.save('temp/np_frames_y.npy', np_frames_y)

print(len(frames_X))
