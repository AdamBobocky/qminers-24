import pandas as pd
import numpy as np

def inverse_sigmoid(x):
    return np.log(x / (1 - x))

df = pd.read_csv('data/games.csv')

df['Benchmark'] = inverse_sigmoid(1 / df['OddsH'] / (1 / df['OddsH'] + 1 / df['OddsA']))

# 13,28

new_df = df[((df['HID'] == 13) & (df['AID'] == 28) | (df['HID'] == 28) & (df['AID'] == 13))][['Season', 'HID', 'AID', 'OddsH', 'OddsA', 'Benchmark']]

print(new_df.to_string())

print(df['Benchmark'].mean())
