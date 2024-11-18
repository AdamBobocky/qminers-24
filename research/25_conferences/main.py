import pandas as pd
import numpy as np

df = pd.read_csv('data/games.csv')

filtered_df = df[(df['POFF'] == 0) & (df['Season'] == 10)]

filtered_df[['Team1', 'Team2']] = filtered_df[['HID', 'AID']].apply(sorted, axis=1, result_type='expand')

frequencies = filtered_df.groupby(['Team1', 'Team2']).size().reset_index(name='Frequency')

pd.set_option('display.max_rows', None)

print(frequencies)
