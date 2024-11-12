import pandas as pd

df = pd.read_csv('data/games.csv')

print(df['POFF'].value_counts())
