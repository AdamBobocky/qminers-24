import pandas as pd

file_path = 'data/games.csv'
df = pd.read_csv(file_path)

df['dateA'] = pd.to_datetime(df['Date'])
df['dateB'] = pd.to_datetime(df['Open'])

df['days_difference'] = (df['dateB'] - df['dateA']).dt.days

print(df['days_difference'].unique()) # [-1]
