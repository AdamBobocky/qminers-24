import numpy as np
import pandas as pd
import string
from datetime import datetime

class Model:
    def __init__(self):
        pass

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame]):
        games_increment, players_increment = inc

        min_bet = summary.iloc[0]['Min_bet']
        max_bet = summary.iloc[0]['Max_bet']

        bets = pd.DataFrame(data=np.zeros((len(opps), 2)), columns=['BetH', 'BetA'], index=opps.index)

        for i in opps.index:
            current = opps.loc[i]

            home_id = current['HID']
            away_id = current['AID']
            home_odds = current['OddsH']
            away_odds = current['OddsA']
            date = current['Date']
            playoff = current['POFF']

            if playoff == 1:
                if home_odds < 1.8:
                    bets.at[i, 'BetH'] = min_bet * 2
                if away_odds < 1.8:
                    bets.at[i, 'BetA'] = min_bet * 2

        return bets
