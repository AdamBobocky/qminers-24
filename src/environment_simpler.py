from typing import Optional

import numpy as np
import pandas as pd

class IModel:
    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: pd.DataFrame):
        raise NotImplementedError()

class Environment:

    result_cols = ["H", "A"]

    odds_cols = ["OddsH", "OddsA"]

    score_cols = ["HSC", "ASC"]

    # fmt: off
    feature_cols = [
        "HFGM", "AFGM", "HFGA", "AFGA", "HFG3M", "AFG3M", "HFG3A", "AFG3A",
        "HFTM", "AFTM", "HFTA", "AFTA", "HORB", "AORB", "HDRB", "ADRB", "HRB", "ARB", "HAST",
        "AAST", "HSTL", "ASTL", "HBLK", "ABLK", "HTOV", "ATOV", "HPF", "APF",
    ]
    # fmt: on

    def __init__(
        self,
        games: pd.DataFrame,
        players: pd.DataFrame,
        model: IModel,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ):
        self.games = games
        self.players = players

        self.start_date: pd.Timestamp = (
            start_date if start_date is not None else self.games["Open"].min()
        )
        self.end_date: pd.Timestamp = (
            end_date if end_date is not None else self.games["Date"].max()
        )

        self.model = model

        self.last_seen = pd.to_datetime("1900-01-01")

    def run(self):
        print(f"Start: {self.start_date}, End: {self.end_date}")
        for date in pd.date_range(self.start_date, self.end_date):
            inc = self._next_date(date)
            opps = self._get_options(date)
            if opps.empty:
                continue

            summary = self._generate_summary(date)

            self.model.place_bets(summary, opps, inc)

        self._next_date(self.end_date + pd.to_timedelta(1, "days"))

        self.model.end()

        return self.games

    def _next_date(self, date: pd.Timestamp):
        games = self.games.loc[
            (self.games["Date"] > self.last_seen) & (self.games["Date"] < date)
        ]
        players = self.players.loc[
            (self.players["Date"] > self.last_seen) & (self.players["Date"] < date)
        ]
        self.last_seen = games["Date"].max() if not games.empty else self.last_seen

        return games.drop(["Open"], axis=1), players

    def _get_options(self, date: pd.Timestamp):
        opps = self.games.loc[
            (self.games["Open"] <= date) & (self.games["Date"] >= date)
        ]
        opps = opps.loc[opps[self.odds_cols].sum(axis=1) > 0]
        return opps.drop(
            [*self.score_cols, *self.result_cols, *self.feature_cols, "Open"],
            axis=1,
        )

    def _generate_summary(self, date: pd.Timestamp):
        summary = {
            "Date": date
        }
        return pd.Series(summary).to_frame().T
