import numpy as np
import pandas as pd
from datetime import datetime

class Model:
    def __init__(self):
        self.last_season = -1
        self.season_start = 0
        self.bet_count = 0

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame]):
        min_bet = summary.iloc[0]['Min_bet']
        max_bet = summary.iloc[0]['Max_bet']

        bets = pd.DataFrame(data=np.zeros((len(opps), 2)), columns=['BetH', 'BetA'], index=opps.index)

        for i in opps.index:
            current = opps.loc[i]

            if self.last_season != current['Season']:
                self.last_season = current['Season']
                self.season_start = current['Date']

        for i in opps.index:
            current = opps.loc[i]

            if current['Date'] == summary.iloc[0]['Date']:
                odds_home = current['OddsH']
                odds_away = current['OddsA']
                overround = 1 / odds_home + 1 / odds_away

                if overround >= 1.065:
                    if odds_home >= 2.3:
                        bets.at[i, 'BetH'] = min_bet
                        self.bet_count += 1

                        print('\nself.bet_count:', self.bet_count)

                    if odds_away >= 2.3:
                        bets.at[i, 'BetA'] = min_bet
                        self.bet_count += 1

                        print('\nself.bet_count:', self.bet_count)

                # if (current['Date'] - self.season_start).days <= 4:
                #     if odds_home >= 2.5:
                #         bets.at[i, 'BetH'] = min_bet
                #         self.bet_count += 1

                #         print('\nself.bet_count:', self.bet_count)

                #     if odds_away >= 2.5:
                #         bets.at[i, 'BetA'] = min_bet
                #         self.bet_count += 1

                #         print('\nself.bet_count:', self.bet_count)

        return bets
