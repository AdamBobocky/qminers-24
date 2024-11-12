import numpy as np
import pandas as pd
import string
import random
from datetime import datetime

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
   return ''.join(random.choice(chars) for _ in range(size))

class Model:
    def __init__(self):
        self.last_season = -1
        self.season_start = 0
        self.team_rest = {}
        self.noise = {}

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame]):
        games_increment, players_increment = inc

        min_bet = summary.iloc[0]['Min_bet']
        max_bet = summary.iloc[0]['Max_bet']

        bets = pd.DataFrame(data=np.zeros((len(opps), 2)), columns=['BetH', 'BetA'], index=opps.index)

        for idx in games_increment.index:
            current = games_increment.loc[idx]

            home_id = current['HID']
            away_id = current['AID']
            date = current['Date']

            self.team_rest[home_id] = date
            self.team_rest[away_id] = date

        for i in opps.index:
            current = opps.loc[i]

            if self.last_season != current['Season']:
                self.last_season = current['Season']
                self.season_start = current['Date']

        for _ in range(10000):
            self.noise[id_generator(4)] = 1
            id_generator(12)

        for i in opps.index:
            current = opps.loc[i]

            home_id = current['HID']
            away_id = current['AID']
            date = current['Date']

            season_days = (date - self.season_start).days

            if season_days >= 5 and home_id in self.team_rest and away_id in self.team_rest:
                home_days = (date - self.team_rest[home_id]).days
                away_days = (date - self.team_rest[away_id]).days

                if home_days + away_days > 50:
                    pass
                else:
                    if home_days == 1 and away_days == 1:
                        pass
                    elif home_days == 1:
                        bets.at[i, 'BetA'] = min_bet * 2
                    elif away_days == 1:
                        bets.at[i, 'BetH'] = min_bet * 2

        return bets
