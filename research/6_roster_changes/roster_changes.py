import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

def list_difference_count(A, B):
    set_A = set(A)
    set_B = set(B)

    only_in_A = set_A - set_B
    only_in_B = set_B - set_A

    return len(only_in_A) + len(only_in_B)

# Load the CSV file into a DataFrame
file_path = 'data/players.csv'  # Change this to your file path
df = pd.read_csv(file_path)

games = df.groupby('Game')

last_rosters = {}

roster_changes = {}

for game_id, data in games:
  unique_teams = data['Team'].unique()
  if len(unique_teams) == 2:
    team1, team2 = unique_teams

    team_1_players = sorted(data[data['Team'] == team1]['Player'].to_list())
    team_2_players = sorted(data[data['Team'] == team2]['Player'].to_list())

    if team1 in last_rosters:
      delta = list_difference_count(last_rosters[team1], team_1_players)
      if delta not in roster_changes:
        roster_changes[delta] = 0
      roster_changes[delta] += 1

    if team2 in last_rosters:
      delta = list_difference_count(last_rosters[team2], team_2_players)
      if delta not in roster_changes:
        roster_changes[delta] = 0
      roster_changes[delta] += 1

    last_rosters[team1] = team_1_players
    last_rosters[team2] = team_2_players

sorted_occurrences = dict(sorted(roster_changes.items(), key=lambda item: item[0]))
cumulative_sum = sum(sorted_occurrences.values())
frequencies = {key: value / cumulative_sum for key, value in sorted_occurrences.items()}

print("Frequencies based on occurrences:")
for key, freq in frequencies.items():
    print(f"{key}: {freq:.4f}")
