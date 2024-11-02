import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from collections import defaultdict

# Load the CSV file into a DataFrame
file_path = 'data/players.csv'  # Change this to your file path
df = pd.read_csv(file_path)

games = df.groupby('Game')

players_teams = {}

for game_id, data in games:
  unique_teams = data['Team'].unique()
  if len(unique_teams) == 2:
    team1, team2 = unique_teams

    team_1_players = data[data['Team'] == team1]['Player'].to_list()
    team_2_players = data[data['Team'] == team2]['Player'].to_list()

    for player in team_1_players:
      if player not in players_teams:
        players_teams[player] = []
      if team1 not in players_teams[player]:
        players_teams[player].append(team1)

    for player in team_2_players:
      if player not in players_teams:
        players_teams[player] = []
      if team2 not in players_teams[player]:
        players_teams[player].append(team2)

team_player_count = defaultdict(int)

for teams in players_teams.values():
    team_player_count[len(teams)] += 1

# Step 2: Sort the dictionary by the number of players (values)
sorted_occurrences = dict(sorted(team_player_count.items(), key=lambda item: item[0]))
cumulative_sum = len(players_teams)
frequencies = {key: value / cumulative_sum for key, value in sorted_occurrences.items()}

print("Frequencies based on occurrences:")
for key, freq in frequencies.items():
    print(f"{key}: {freq:.4f}")

# print(players_teams)
