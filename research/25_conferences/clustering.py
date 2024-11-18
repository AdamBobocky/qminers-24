import pandas as pd

df = pd.read_csv('data/games.csv')

df = df[df['POFF'] == 0]

for index, current in df.iterrows():
    season = current['Season']
    home_id = current['HID']
    away_id = current['AID']

    print(home_id, away_id)

# Iterate seasonal non-poff games of all teams, and find ones which played against each other the least
# Now place them in opposing conferences
# Now iterate again, and find teams that play more with
