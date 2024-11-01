import math
import numpy as np
import pandas as pd
from trueskill import Rating, rate, global_env

def p_win_1v1(
    p1: Rating,
    p2: Rating,
    n: int = 2
) -> float:
    """Calculate the probability that p1 wins the game."""
    env = global_env()
    return env.cdf(
        (p1.mu - p2.mu) /
        math.sqrt(n * env.beta**2 + p1.sigma**2 + p2.sigma**2)
    )

def p_win_team(
    team1: list[Rating],
    team2: list[Rating],
) -> float:
    """Calculate the probability that team1 wins the game."""
    n = len(team1) + len(team2)
    p1 = team_rating(team1)
    p2 = team_rating(team2)
    return p_win_1v1(p1, p2, n)

def avg_rating(team: list[Rating]) -> Rating:
    return Rating(
        sum([p.mu for p in team]) / len(team),
        sum([p.sigma for p in team]) / len(team),
        # math.sqrt(sum([p.sigma**2 for p in team]) / len(team)),
    )

def team_rating(team: list[Rating]) -> Rating:
    """Return sum of ratings as a Rating, i.e. the sum of Gaussians."""
    return Rating(
        sum([p.mu for p in team]),
        math.sqrt(sum([p.sigma**2 for p in team])),
    )

class Model:
    rating_map = {}
    games_data = []
    last_rosters = {}
    predictions = []
    metrics = {
        'rating_mse': 0,
        'odds_mse': 0,
        'n': 0
    }

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame]):
        games_increment, players_increment = inc
        print(summary['Date'])
        # print(opps) # Contains betting opportunities
        # print(games_increment) # New games data
        # print(players_increment) # New players data

        self.games_data.append(games_increment)

        for i in players_increment.index:
            current = players_increment.loc[i]

            if current['Player'] not in self.rating_map:
                self.rating_map[current['Player']] = Rating()

        players_by_game = players_increment.groupby('Game')

        for _, data in players_by_game:
            unique_teams = data['Team'].unique()

            if len(unique_teams) == 2:
                team1, team2 = unique_teams

                team_1_players = sorted(data[data['Team'] == team1]['Player'].to_list())
                team_2_players = sorted(data[data['Team'] == team2]['Player'].to_list())

                self.last_rosters[team1] = team_1_players
                self.last_rosters[team2] = team_2_players

        for i in games_increment.index:
            current = games_increment.loc[i]

            home_id = current['HID']
            away_id = current['AID']
            home_win = current['H']

            home_players = self.last_rosters[home_id]
            away_players = self.last_rosters[away_id]

            home_ratings = [self.rating_map[index] for index in home_players]
            away_ratings = [self.rating_map[index] for index in away_players]

            home_avg_rating = avg_rating(home_ratings)
            away_avg_rating = avg_rating(away_ratings)

            while len(home_ratings) < len(away_ratings):
                home_ratings.append(home_avg_rating)
            while len(away_ratings) < len(home_ratings):
                away_ratings.append(away_avg_rating)

            new_home_ratings, new_away_ratings = rate([home_ratings, away_ratings], ranks=([0, 1] if home_win else [1, 0]))

            for player, new_rating in zip(home_players, new_home_ratings):
                self.rating_map[player] = new_rating

            for player, new_rating in zip(away_players, new_away_ratings):
                self.rating_map[player] = new_rating

        for i in opps.index:
            current = opps.loc[i]

            home_id = current['HID']
            away_id = current['AID']
            home_odds = current['OddsH']
            away_odds = current['OddsA']

            prediction = 0.5
            odds_prediction = 1 / home_odds / (1 / home_odds + 1 / away_odds)

            if home_id in self.last_rosters and away_id in self.last_rosters:
                home_players = self.last_rosters[home_id]
                away_players = self.last_rosters[away_id]

                home_ratings = [self.rating_map.get(index, Rating()) for index in home_players]
                away_ratings = [self.rating_map.get(index, Rating()) for index in away_players]

                home_avg_rating = avg_rating(home_ratings)
                away_avg_rating = avg_rating(away_ratings)

                while len(home_ratings) < len(away_ratings):
                    home_ratings.append(home_avg_rating)
                while len(away_ratings) < len(home_ratings):
                    away_ratings.append(away_avg_rating)

                prediction = p_win_team(home_ratings, away_ratings)

            self.predictions.append({
                'index': i,
                'rating_pred': prediction,
                'odds_pred': odds_prediction,
                'home_odds': home_odds,
                'away_odds': away_odds
            })

    def end(self):
        merge = pd.concat(self.games_data)

        backtests = {
            'pnl_0%': 0,
            'bets_0%': 0,
            'vig_0%': 0,
            'pnl_10%': 0,
            'bets_10%': 0,
            'vig_10%': 0,
            'pnl_20%': 0,
            'bets_20%': 0,
            'vig_20%': 0
        }

        for current in self.predictions:
            if current['index'] in merge.index:
                home_win = merge.at[current['index'], 'H']
                away_win = merge.at[current['index'], 'A']
                rating_pred = current['rating_pred']
                odds_pred = current['odds_pred']
                home_odds = current['home_odds']
                away_odds = current['away_odds']
                self.metrics['rating_mse'] += (rating_pred - home_win) ** 2
                self.metrics['odds_mse'] += (odds_pred - home_win) ** 2
                self.metrics['n'] += 1

                if rating_pred * home_odds > 1:
                    backtests['pnl_0%'] -= 1
                    if home_win:
                        backtests['pnl_0%'] += home_odds
                    backtests['vig_0%'] += 1 / home_odds + 1 / away_odds - 1
                    backtests['bets_0%'] += 1
                if (1 - rating_pred) * away_odds > 1:
                    backtests['pnl_0%'] -= 1
                    if away_win:
                        backtests['pnl_0%'] += away_odds
                    backtests['vig_0%'] += 1 / home_odds + 1 / away_odds - 1
                    backtests['bets_0%'] += 1

                if rating_pred * home_odds > 1.1:
                    backtests['pnl_10%'] -= 1
                    if home_win:
                        backtests['pnl_10%'] += home_odds
                    backtests['vig_10%'] += 1 / home_odds + 1 / away_odds - 1
                    backtests['bets_10%'] += 1
                if (1 - rating_pred) * away_odds > 1.1:
                    backtests['pnl_10%'] -= 1
                    if away_win:
                        backtests['pnl_10%'] += away_odds
                    backtests['vig_10%'] += 1 / home_odds + 1 / away_odds - 1
                    backtests['bets_10%'] += 1

                if rating_pred * home_odds > 1.2:
                    backtests['pnl_20%'] -= 1
                    if home_win:
                        backtests['pnl_20%'] += home_odds
                    backtests['vig_20%'] += 1 / home_odds + 1 / away_odds - 1
                    backtests['bets_20%'] += 1
                if (1 - rating_pred) * away_odds > 1.2:
                    backtests['pnl_20%'] -= 1
                    if away_win:
                        backtests['pnl_20%'] += away_odds
                    backtests['vig_20%'] += 1 / home_odds + 1 / away_odds - 1
                    backtests['bets_20%'] += 1

        print('rating_mse:', self.metrics['rating_mse'] / self.metrics['n'])
        print('odds_mse:', self.metrics['odds_mse'] / self.metrics['n'])
        print(backtests)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))
