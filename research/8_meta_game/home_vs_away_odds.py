# Track inverse sigmoid of implied odds of each team during season, see how accurately it predicts
# outcomes into the future.

import pandas as pd
import numpy as np

def inverse_sigmoid(x):
    return np.log(x / (1 - x))

file_path = 'data/games.csv'
df = pd.read_csv(file_path)

df['Inv_OddsH'] = inverse_sigmoid(1 / df['OddsH'])
df['Inv_OddsA'] = inverse_sigmoid(1 / df['OddsA'])

print(df['Inv_OddsH'].mean() - df['Inv_OddsA'].mean())
