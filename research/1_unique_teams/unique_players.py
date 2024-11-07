import pandas as pd

df = pd.read_csv('data/players.csv')

unique_ids = df['Player'].unique()

# Print the unique IDs
print(f'Unique players: {len(unique_ids)}')
