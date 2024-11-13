import pandas as pd
import pandas as np

# Load the CSV file into a DataFrame
file_path = 'data/games.csv'  # Change this to your file path
df = pd.read_csv(file_path)

df = df[(df['OddsH'] != 0) & (df['OddsA'] != 0)]

# Calculate de-vigged probabilities for both home and away teams
# Convert odds to probabilities (1/odds)
df['probH'] = 1 / df['OddsH']
df['probA'] = 1 / df['OddsA']

# Normalize the probabilities to remove the vig
df['overround'] = df['probH'] + df['probA'] - 1.0

df['devig_probH'] = 1 / df['OddsH'] / (df['overround'] + 1)
df['devig_probA'] = 1 / df['OddsA'] / (df['overround'] + 1)

mse_map = {}
overrounds = []

for index, row in df.iterrows():
    rounded = round(row['overround'], 2)
    overrounds.append(rounded)

    if rounded not in mse_map:
        mse_map[rounded] = [0, 0]

    mse_map[rounded][0] += (row['devig_probH'] - row['H']) ** 2
    mse_map[rounded][1] += 1

    if rounded == 0.07:
        print(row['devig_probH'])

print(np.unique(overrounds))

for key, value in mse_map.items():
    print(key, value[0] / value[1], value[1])
