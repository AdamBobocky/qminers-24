import pandas as pd
import numpy as np

players_df = pd.read_csv('data/players.csv')

players_df = players_df[players_df['MIN'] > 3]

players_df = players_df.groupby('Game')

player_counts = []
total_minutes = []

for _, group in players_df:
    for _, players in group.groupby('Team'):
        player_counts.append(len(players))
        total_minutes.append(round(sum(players['MIN'])))

print(min(player_counts), max(player_counts))
print(np.unique(total_minutes, return_counts=True))
