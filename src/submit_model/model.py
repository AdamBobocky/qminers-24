import warnings

warnings.filterwarnings('ignore')
from collections import defaultdict
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import softplus
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
import math
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from datetime import datetime

class Exhaustion:
    def __init__(self):
        self.team_rest = {}

    def pre_add_game(self, current, current_players):
        pass

    def add_game(self, current, current_players):
        date = current['Date']
        home_id = current['HID']
        away_id = current['AID']

        self.team_rest[home_id] = date
        self.team_rest[away_id] = date

    def get_input_data(self, home_id, away_id, season, date):
        if home_id not in self.team_rest or away_id not in self.team_rest:
            return None

        home_days = (date - self.team_rest[home_id]).days
        away_days = (date - self.team_rest[away_id]).days

        factor = 0.0

        if home_days <= 1:
            factor += 1.0
        if away_days <= 1:
            factor -= 1.0

        return [
            factor
        ]

def inverse_sigmoid(x):
    return math.log(x / (1 - x))

class FourFactor:
    def __init__(self):
        # Hyperparameters
        self.lr_required_n = 2000
        # End

        self.team_stats_average = defaultdict(list)
        self.opponent_stats_average = defaultdict(list)

        self.past_pred = []
        self.lr = None
        self.lr_retrain = 0

    def _get_stats(self, date, stats):
        totals = {
            'FieldGoalsMade': 0,
            '3PFieldGoalsMade': 0,
            'FieldGoalAttempts': 0,
            'Turnovers': 0,
            'OffensiveRebounds': 0,
            'OpponentsDefensiveRebounds': 0,
            'FreeThrowAttempts': 0,
            'Score': 0,
            'Win': 0,
            'Weight': 0
        }

        # Iterate over each dictionary in the list
        for stat in stats:
            weight = 0.994 ** abs((date - stat['Date']).days)

            # Multiply each relevant field by the weight and add to totals
            totals['FieldGoalsMade'] += stat['FieldGoalsMade'] * weight
            totals['3PFieldGoalsMade'] += stat['3PFieldGoalsMade'] * weight
            totals['FieldGoalAttempts'] += stat['FieldGoalAttempts'] * weight
            totals['Turnovers'] += stat['Turnovers'] * weight
            totals['OffensiveRebounds'] += stat['OffensiveRebounds'] * weight
            totals['OpponentsDefensiveRebounds'] += stat['OpponentsDefensiveRebounds'] * weight
            totals['FreeThrowAttempts'] += stat['FreeThrowAttempts'] * weight
            totals['Score'] += stat['Score'] * weight
            totals['Win'] += stat['Win'] * weight
            totals['Weight'] += weight

        return totals

    def pre_add_game(self, current, current_players):
        pass

    def add_game(self, current, current_players):
        val1 = self._get_team_input_data(current['HID'], current['Date'])
        val2 = self._get_team_input_data(current['AID'], current['Date'])

        if val1 is not None:
            self.past_pred.append([*val1, current['H']])
            self.lr_retrain -= 1

        if val2 is not None:
            self.past_pred.append([*val2, current['A']])
            self.lr_retrain -= 1

        self.team_stats_average[current['HID']].append({
            'Date': current['Date'],
            'FieldGoalsMade': current['HFGM'],
            '3PFieldGoalsMade': current['HFG3M'],
            'FieldGoalAttempts': current['HFGA'],
            'Turnovers': current['HTOV'],
            'OffensiveRebounds': current['HORB'],
            'OpponentsDefensiveRebounds': current['ADRB'],
            'FreeThrowAttempts': current['HFTA'],
            'Score': current['HSC'],
            'Win': current['H']
        })
        self.team_stats_average[current['AID']].append({
            'Date': current['Date'],
            'FieldGoalsMade': current['AFGM'],
            '3PFieldGoalsMade': current['AFG3M'],
            'FieldGoalAttempts': current['AFGA'],
            'Turnovers': current['ATOV'],
            'OffensiveRebounds': current['AORB'],
            'OpponentsDefensiveRebounds': current['ADRB'],
            'FreeThrowAttempts': current['AFTA'],
            'Score': current['ASC'],
            'Win': current['A']
        })
        # Opponent
        self.opponent_stats_average[current['AID']].append({
            'Date': current['Date'],
            'FieldGoalsMade': current['HFGM'],
            '3PFieldGoalsMade': current['HFG3M'],
            'FieldGoalAttempts': current['HFGA'],
            'Turnovers': current['HTOV'],
            'OffensiveRebounds': current['HORB'],
            'OpponentsDefensiveRebounds': current['ADRB'],
            'FreeThrowAttempts': current['HFTA'],
            'Score': current['HSC'],
            'Win': current['H']
        })
        self.opponent_stats_average[current['HID']].append({
            'Date': current['Date'],
            'FieldGoalsMade': current['AFGM'],
            '3PFieldGoalsMade': current['AFG3M'],
            'FieldGoalAttempts': current['AFGA'],
            'Turnovers': current['ATOV'],
            'OffensiveRebounds': current['AORB'],
            'OpponentsDefensiveRebounds': current['ADRB'],
            'FreeThrowAttempts': current['AFTA'],
            'Score': current['ASC'],
            'Win': current['A']
        })

    def _get_team_input_data(self, team_id, date):
        if len(self.team_stats_average[team_id]) <= 5:
            return None

        stats = self._get_stats(date, self.team_stats_average[team_id][-100:])
        opp_stats = self._get_stats(date, self.opponent_stats_average[team_id][-100:])

        return [
            (stats['FieldGoalsMade'] + 0.5 * stats['3PFieldGoalsMade']) / stats['FieldGoalAttempts'],
            stats['Turnovers'] / (stats['FieldGoalAttempts'] + 0.44 * stats['FreeThrowAttempts'] + stats['Turnovers']),
            stats['OffensiveRebounds'] / (stats['OffensiveRebounds'] + stats['OpponentsDefensiveRebounds']),
            stats['FreeThrowAttempts'] / stats['FieldGoalAttempts'],
            stats['Score'] / stats['Weight'],
            (opp_stats['FieldGoalsMade'] + 0.5 * opp_stats['3PFieldGoalsMade']) / opp_stats['FieldGoalAttempts'],
            opp_stats['Turnovers'] / (opp_stats['FieldGoalAttempts'] + 0.44 * opp_stats['FreeThrowAttempts'] + opp_stats['Turnovers']),
            opp_stats['OffensiveRebounds'] / (opp_stats['OffensiveRebounds'] + opp_stats['OpponentsDefensiveRebounds']),
            opp_stats['FreeThrowAttempts'] / opp_stats['FieldGoalAttempts'],
            opp_stats['Score'] / opp_stats['Weight']
        ]

    def get_input_data(self, home_id, away_id, season, date):
        val1 = self._get_team_input_data(home_id, date)
        val2 = self._get_team_input_data(away_id, date)

        if val1 is None or val2 is None:
            return None

        if len(self.past_pred) < self.lr_required_n:
            return None

        if self.lr_retrain <= 0:
            self.lr_retrain = 200

            np_array = np.array(self.past_pred)
            sample_weights = np.exp(-0.0003 * np.arange(len(self.past_pred)))
            self.lr = LogisticRegression(max_iter=10000)
            self.lr.fit(np_array[:, :-1], np_array[:, -1], sample_weight=sample_weights[::-1])

        return [
            inverse_sigmoid(self.lr.predict_proba(np.array([val1]))[0, 1]) -
            inverse_sigmoid(self.lr.predict_proba(np.array([val2]))[0, 1])
        ]

class GradientDescent:
    def __init__(self, num_teams=30, monthly_decay=0.8, long_term_decay=0.97):
        self.num_teams = num_teams
        self.monthly_decay = monthly_decay
        self.long_term_decay = long_term_decay

        self.team_mus = nn.Parameter(torch.zeros(num_teams))
        self.team_sigmas = torch.ones(num_teams) * 40
        self.team_home_advantages = torch.ones(num_teams) * 0
        self.home_advantage = nn.Parameter(torch.tensor(5.0))
        self.sigma = nn.Parameter(torch.tensor(12.0))

        self.games = torch.zeros((50000, 5), dtype=torch.long)
        self.game_count = 0

        self.my_team_id = {}
        self.last_season = -1
        self.fit_date = None

    def _check_season(self, season):
        if self.last_season != season:
            self.last_season = season

            self.team_sigmas = torch.ones_like(self.team_sigmas) * 20

    def _get_time_weights(self):
        timestamps = self.games[:self.game_count, 0].to(torch.float32)
        last_ts = timestamps[-1]

        return (self.monthly_decay ** (torch.abs(timestamps - last_ts) / (30 * 24 * 60 * 60 * 1000))).to(timestamps.device)

    def _get_long_term_weights(self):
        timestamps = self.games[:self.game_count, 0].to(torch.float32)
        last_ts = timestamps[-1]

        return (self.long_term_decay ** (torch.abs(timestamps - last_ts) / (30 * 24 * 60 * 60 * 1000))).to(timestamps.device)


    def forward(self, weights, long_term_weights, idx_start):
        games = self.games[idx_start:self.game_count]
        home_ratings = self.team_mus[games[:, 1]]
        away_ratings = self.team_mus[games[:, 2]]
        teams_home_advs = self.team_home_advantages[games[:, 1]]

        expectations_home = (
            self.home_advantage * long_term_weights[idx_start:]
            + teams_home_advs * long_term_weights[idx_start:]
            + home_ratings - away_ratings
        )
        realities_home = games[:, 3] - games[:, 4]

        game_sigmas = torch.sqrt(
            self.team_sigmas[games[:, 1]] ** 2 +
            self.team_sigmas[games[:, 2]] ** 2 +
            self.sigma ** 2
        )

        log_value = (-0.5 * ((realities_home - expectations_home) / game_sigmas) ** 2
                     - torch.log(game_sigmas)
                     - 0.5 * torch.log(torch.tensor(2 * np.pi)))
        return torch.sum(log_value * weights[idx_start:])

    def _fit(self, max_epochs=30):
        weights = self._get_time_weights()
        long_term_weights = self._get_long_term_weights()
        idx_start = torch.nonzero(weights > 0.004, as_tuple=True)[0][0].item()
        optimizer = optim.Adam([self.team_mus, self.sigma, self.home_advantage, self.team_home_advantages], lr=0.02)

        for epoch in range(max_epochs):
            optimizer.zero_grad()
            loss = -self.forward(weights, long_term_weights, idx_start)
            loss.backward()
            optimizer.step()

    def add_game(self, current, current_players):
        self._check_season(current['Season'])

        timestamp = int(current['Date'].timestamp() * 1000)
        team_home = self._map_team_id(current['HID'])
        team_away = self._map_team_id(current['AID'])
        score_home = current['HSC']
        score_away = current['ASC']

        self.games[self.game_count] = torch.tensor(
            [timestamp, team_home, team_away, score_home, score_away], dtype=torch.long
        )
        self.game_count += 1

        game_sigma2 = math.sqrt(
            self.team_sigmas[team_home] ** 2 +
            self.team_sigmas[team_away] ** 2 +
            self.sigma ** 2
        )

        self.team_sigmas[team_home] = 1 / torch.sqrt(
            1 / self.team_sigmas[team_home] ** 2 + 1 / game_sigma2)
        self.team_sigmas[team_away] = 1 / torch.sqrt(
            1 / self.team_sigmas[team_away] ** 2 + 1 / game_sigma2)

        self.fit_date = None

    def _map_team_id(self, team_id):
        if team_id not in self.my_team_id:
            self.my_team_id[team_id] = len(self.my_team_id)

        return self.my_team_id[team_id]

    def pre_add_game(self, current, current_players):
        pass

    def get_input_data(self, home_id, away_id, season, date):
        self._check_season(season)

        if self.game_count < 2000:
            return None

        if self.fit_date is None or self.fit_date != date:
            self.fit_date = date
            self._fit()

        team_home = self._map_team_id(home_id)
        team_away = self._map_team_id(away_id)

        game_exp = self.home_advantage + self.team_mus[team_home] - self.team_mus[team_away]
        game_sigma = torch.sqrt(
            self.team_sigmas[team_home] ** 2 +
            self.team_sigmas[team_away] ** 2 +
            self.sigma ** 2
        )

        return [game_exp.item() / game_sigma.item()]

# Top is new
# Bottom is OG

class NateSilverElo:
    def __init__(self):
        self.elo_map = defaultdict(float)
        self.last_season = -1

    def _new_season(self):
        for key in self.elo_map:
            self.elo_map[key] *= 0.75

    def _win_probability(self, x):
        return 1 / (1 + (math.exp(-x / 175)))

    def pre_add_game(self, current, current_players):
        pass

    def add_game(self, current, current_players):
        season = current['Season']
        home_id = current['HID']
        away_id = current['AID']
        home_score = current['HSC']
        away_score = current['ASC']

        if season > self.last_season:
            self.last_season = season
            self._new_season()

        home_prediction = self._win_probability(self.elo_map[home_id] + 100 - self.elo_map[away_id])
        away_prediction = 1 - home_prediction

        k_factor = self.get_k_factor(home_score - away_score, self.elo_map[home_id] + 100, self.elo_map[away_id])

        self.elo_map[home_id] += k_factor * (current['H'] - home_prediction)
        self.elo_map[away_id] += k_factor * (current['A'] - away_prediction)

    def get_input_data(self, home_id, away_id, season, date):
        if season > self.last_season:
            self.last_season = season
            self._new_season()

        return [
            self.elo_map[home_id] - self.elo_map[away_id] + 100
        ]

    def get_team_strength(self, team_id, is_home, season):
        if season > self.last_season:
            self.last_season = season
            self._new_season()

        return self.elo_map[team_id] + 100 * (0.5 if is_home else -0.5)

    def get_k_factor(self, score_difference, elo_home, elo_away):
        if score_difference > 0:
            return 20 * (score_difference + 3) ** 0.8 / (7.5 + 0.006 * (elo_home - elo_away))
        else:
            return 20 * (-score_difference + 3) ** 0.8 / (7.5 + 0.006 * (elo_away - elo_home))

import random
import math
import torch
import copy
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
from datetime import datetime

pretrained_weights = {
  "home_field_advantage": 4.3094072341918945,
  "player_model.layers.0.weight": [
    [
      -0.40628108382225037,
      0.07856258004903793,
      0.325394868850708,
      -0.04651965573430061,
      0.22802461683750153,
      0.08993364125490189,
      -0.012805229052901268,
      0.2985721230506897,
      -0.114983469247818,
      0.11324682831764221,
      0.17688260972499847,
      -0.15939097106456757,
      0.14691346883773804,
      -0.021689845249056816,
      -0.16040724515914917,
      -0.45319804549217224,
      0.041913509368896484,
      0.04578346014022827,
      0.1634901762008667,
      -0.06727402657270432
    ],
    [
      0.3928733468055725,
      0.1215740516781807,
      -0.17076091468334198,
      -0.11281996220350266,
      0.2011859118938446,
      -0.018632585182785988,
      0.10615407675504684,
      0.023888597264885902,
      0.11094790697097778,
      0.1317845731973648,
      0.2049129605293274,
      0.06919250637292862,
      0.09743773192167282,
      0.35388123989105225,
      0.42477938532829285,
      0.18658334016799927,
      0.01861433871090412,
      0.309643030166626,
      0.2717139422893524,
      0.04014481604099274
    ],
    [
      0.41789430379867554,
      0.0024627188686281443,
      -0.14035896956920624,
      -0.13930851221084595,
      0.12681876122951508,
      -0.005851545371115208,
      0.19314418733119965,
      0.2079148292541504,
      0.14916671812534332,
      0.19649715721607208,
      0.056214217096567154,
      0.2513090670108795,
      0.1668471097946167,
      0.3315052092075348,
      0.4643332064151764,
      0.4694107472896576,
      -0.1311037689447403,
      0.23952819406986237,
      0.14797721803188324,
      0.11224479228258133
    ],
    [
      -0.34502580761909485,
      0.12925435602664948,
      0.654212474822998,
      0.058163709938526154,
      -0.06222563982009888,
      0.11980241537094116,
      0.15323615074157715,
      0.23244594037532806,
      0.23196552693843842,
      0.09697479009628296,
      0.1184496060013771,
      -0.06311710178852081,
      0.03624590113759041,
      -0.14447911083698273,
      -0.27396151423454285,
      -0.03828718513250351,
      0.1291438341140747,
      0.11578258126974106,
      -0.11147730052471161,
      -0.08478951454162598
    ],
    [
      0.3196898400783539,
      0.1613921970129013,
      0.05859702453017235,
      0.21447919309139252,
      0.11481557041406631,
      -0.08189275860786438,
      0.3539499044418335,
      0.5197039246559143,
      -0.2196812778711319,
      0.032281726598739624,
      -0.08282342553138733,
      0.1349317729473114,
      -0.05505254119634628,
      0.10598566383123398,
      0.22474165260791779,
      0.2963859736919403,
      0.1250125765800476,
      0.10720179975032806,
      0.050101954489946365,
      0.2545809745788574
    ],
    [
      0.3572973608970642,
      0.0703587457537651,
      -0.12714727222919464,
      -0.04367605224251747,
      -0.07395759224891663,
      0.07068043947219849,
      0.41835394501686096,
      0.43159985542297363,
      0.10478450357913971,
      0.05105776712298393,
      0.15983158349990845,
      0.1436811238527298,
      0.09200671315193176,
      0.22244232892990112,
      -0.05399100482463837,
      0.23051412403583527,
      0.1412379890680313,
      0.2128426730632782,
      0.08620784431695938,
      -0.03438577428460121
    ],
    [
      0.4455743134021759,
      -0.1133473590016365,
      -0.19943171739578247,
      0.12229631096124649,
      0.015001267194747925,
      0.10240188986063004,
      0.5370718836784363,
      0.1659574657678604,
      0.05063004046678543,
      -0.12466218322515488,
      -0.05443756282329559,
      0.2823655605316162,
      0.024729840457439423,
      -0.07596263289451599,
      0.03752243518829346,
      0.11681242287158966,
      0.05034288018941879,
      -0.039553213864564896,
      0.16105565428733826,
      0.33484330773353577
    ],
    [
      -0.28674307465553284,
      -0.03965514153242111,
      0.33466988801956177,
      0.10774684697389603,
      0.056912533938884735,
      -0.10310427099466324,
      0.3119896650314331,
      0.45400673151016235,
      0.04732339084148407,
      0.09541021287441254,
      0.003992387093603611,
      -0.1004420593380928,
      0.16769854724407196,
      -0.043334897607564926,
      -0.1540612429380417,
      -0.08407316356897354,
      0.19665977358818054,
      0.004578695632517338,
      0.06783322989940643,
      0.1936808079481125
    ],
    [
      0.5484738945960999,
      -0.15244503319263458,
      -0.2027621567249298,
      0.06208739057183266,
      0.23243814706802368,
      -0.2571946680545807,
      0.1083764135837555,
      0.3879607021808624,
      -0.0702187716960907,
      -0.021276092156767845,
      0.2096250057220459,
      0.1756255179643631,
      0.17116934061050415,
      0.2902088463306427,
      0.4971604347229004,
      0.26992297172546387,
      -0.13799309730529785,
      0.24512922763824463,
      0.0029479230288416147,
      0.3626019358634949
    ],
    [
      0.12722691893577576,
      0.15191568434238434,
      0.21158671379089355,
      -0.15945899486541748,
      -0.164481982588768,
      -0.2693355977535248,
      0.008363070897758007,
      0.22649335861206055,
      -0.14795581996440887,
      -0.06377823650836945,
      0.14885762333869934,
      -0.16310659050941467,
      0.19377771019935608,
      0.1329144388437271,
      0.12349480390548706,
      0.12981435656547546,
      -0.11585695296525955,
      -0.02503357082605362,
      -0.19860798120498657,
      -0.1974385678768158
    ],
    [
      -0.16966378688812256,
      0.14018979668617249,
      0.2268741875886917,
      -0.1614578664302826,
      -0.13636355102062225,
      -0.15552766621112823,
      0.11979736387729645,
      0.04349113628268242,
      -0.07724402844905853,
      -0.14406174421310425,
      -0.02987593226134777,
      -0.15521648526191711,
      -0.15629540383815765,
      0.204724982380867,
      -0.10234754532575607,
      -0.07079306244850159,
      0.05816899985074997,
      -0.004951317794620991,
      -0.04280285909771919,
      0.06530462205410004
    ],
    [
      0.24800074100494385,
      -0.43903660774230957,
      0.04329640418291092,
      -0.03980958089232445,
      -0.14698441326618195,
      0.20964641869068146,
      0.25149643421173096,
      0.22565563023090363,
      0.01959618739783764,
      0.23476146161556244,
      0.22463364899158478,
      0.2390316128730774,
      0.20820708572864532,
      -0.1600300371646881,
      -0.0881115049123764,
      0.058833975344896317,
      0.10963530093431473,
      0.08977818489074707,
      0.10369562357664108,
      0.33692631125450134
    ],
    [
      -0.5062018036842346,
      0.066086046397686,
      0.17920580506324768,
      0.21727430820465088,
      0.12692715227603912,
      -0.09898432344198227,
      0.006501663476228714,
      0.24014051258563995,
      -0.10145845264196396,
      0.027333766222000122,
      -0.08032329380512238,
      0.03139592707157135,
      -0.009787927381694317,
      -0.28415152430534363,
      -0.06960422545671463,
      -0.1768156886100769,
      0.20414535701274872,
      -0.1294177919626236,
      -0.11243646591901779,
      -0.13331082463264465
    ],
    [
      0.37416815757751465,
      0.021618284285068512,
      -0.3037388026714325,
      0.165911003947258,
      -0.14670902490615845,
      -0.1453988403081894,
      0.01473803911358118,
      0.17373809218406677,
      -0.1814654916524887,
      0.10341779887676239,
      -0.20151139795780182,
      0.23795689642429352,
      0.22841079533100128,
      -0.0450904443860054,
      0.08601666986942291,
      0.1862129271030426,
      -0.3902186155319214,
      0.15078364312648773,
      0.19853438436985016,
      -0.14968395233154297
    ],
    [
      0.5433275103569031,
      0.09401623159646988,
      -0.2296002060174942,
      -0.04844160005450249,
      -0.05336228013038635,
      0.1475583165884018,
      0.14478223025798798,
      0.12311045080423355,
      -0.05968298390507698,
      0.031938981264829636,
      0.01650242879986763,
      0.10356912016868591,
      0.2715165913105011,
      0.060604799538850784,
      0.12473363429307938,
      0.4419455826282501,
      -0.00969152245670557,
      0.046557482331991196,
      0.09137513488531113,
      0.15335257351398468
    ],
    [
      -0.2714052200317383,
      -0.051823269575834274,
      0.36628004908561707,
      0.12246937304735184,
      0.21005982160568237,
      0.25585421919822693,
      0.143593430519104,
      0.18992221355438232,
      0.05789532884955406,
      -0.097569540143013,
      -0.0733136534690857,
      0.12812687456607819,
      -0.1622532457113266,
      -0.23963263630867004,
      -0.12671427428722382,
      -0.09902216494083405,
      0.06179652363061905,
      0.013818947598338127,
      -0.17494270205497742,
      -0.1792289763689041
    ]
  ],
  "player_model.layers.0.bias": [
    0.08794743567705154,
    0.10999306291341782,
    0.2343338131904602,
    0.33256348967552185,
    0.10032830387353897,
    0.21090805530548096,
    -0.07330162823200226,
    0.10210084170103073,
    0.20729485154151917,
    0.18456260859966278,
    -0.08346591144800186,
    0.32541146874427795,
    0.10115228593349457,
    -0.10315486788749695,
    -0.14599928259849548,
    -0.024419452995061874
  ],
  "player_model.layers.2.weight": [
    [
      -0.27746695280075073,
      -0.02676437608897686,
      0.319453626871109,
      -0.16773271560668945,
      -0.0863225981593132,
      0.0543908029794693,
      0.15903693437576294,
      0.20512893795967102,
      0.32378697395324707,
      -0.1565398871898651,
      -0.043305959552526474,
      0.03197691962122917,
      -0.253505140542984,
      0.3075566291809082,
      0.24705296754837036,
      0.04965822398662567
    ],
    [
      -0.18605248630046844,
      0.3025258481502533,
      0.21577204763889313,
      -0.08042564243078232,
      0.1427793651819229,
      0.2591501772403717,
      0.24975697696208954,
      0.21411414444446564,
      0.21442490816116333,
      0.07705513387918472,
      -0.1314847618341446,
      0.25793391466140747,
      -0.018998457118868828,
      -0.023292051628232002,
      0.21836023032665253,
      -0.1880127638578415
    ],
    [
      -0.16257382929325104,
      -0.045232076197862625,
      0.20158803462982178,
      0.18528057634830475,
      0.22362098097801208,
      0.18624302744865417,
      0.14058992266654968,
      0.11778935045003891,
      -0.0832444280385971,
      -0.060571227222681046,
      0.11513679474592209,
      0.22877025604248047,
      -0.06679096817970276,
      0.16793382167816162,
      0.0725557804107666,
      -0.23204566538333893
    ],
    [
      0.03831877186894417,
      0.5118112564086914,
      0.5164991021156311,
      -0.11555247008800507,
      0.06559963524341583,
      0.25242340564727783,
      -0.004394876770675182,
      0.11528041958808899,
      0.5840326547622681,
      0.04793361574411392,
      0.09488262236118317,
      0.04144531488418579,
      -0.17478318512439728,
      -0.02469741925597191,
      0.16929425299167633,
      -0.334975928068161
    ],
    [
      0.04974634200334549,
      -0.23119981586933136,
      -0.08658786863088608,
      -0.019636375829577446,
      -0.1366724967956543,
      0.14145004749298096,
      -0.00023819210764486343,
      0.21556982398033142,
      0.16289091110229492,
      -0.1612900346517563,
      0.1932946890592575,
      -0.25672197341918945,
      -0.06419892609119415,
      -0.25183621048927307,
      0.1752787083387375,
      -0.40702807903289795
    ],
    [
      -0.12960292398929596,
      0.07065378129482269,
      0.09582730382680893,
      0.022987088188529015,
      0.08614534139633179,
      0.23498909175395966,
      0.12963944673538208,
      0.2584710717201233,
      0.25133568048477173,
      -0.33486872911453247,
      -0.17716084420681,
      0.10210026055574417,
      0.21465997397899628,
      0.08823790401220322,
      0.06620696932077408,
      0.24837975203990936
    ],
    [
      -0.3945326805114746,
      0.26264363527297974,
      0.364901065826416,
      -0.3049011528491974,
      0.06905525177717209,
      0.0551825575530529,
      0.1363847702741623,
      -0.1168384701013565,
      0.2651750147342682,
      -0.10428318381309509,
      0.27626901865005493,
      -0.09515644609928131,
      -0.0501607246696949,
      0.16902567446231842,
      -0.06728246062994003,
      -0.15408670902252197
    ],
    [
      -0.23340021073818207,
      0.27115023136138916,
      0.34969210624694824,
      -0.4915331304073334,
      -0.05750895291566849,
      0.09145598113536835,
      0.10582268238067627,
      -0.06413092464208603,
      0.338711678981781,
      -0.17943546175956726,
      0.1596212238073349,
      0.25770431756973267,
      -0.340376079082489,
      0.32763800024986267,
      0.5596314072608948,
      -0.5250003337860107
    ],
    [
      0.23401986062526703,
      0.01802758499979973,
      0.03921467438340187,
      0.1629827320575714,
      -0.15302744507789612,
      0.013980724848806858,
      -0.001350213075056672,
      0.27418968081474304,
      -0.11180860549211502,
      0.19303832948207855,
      -0.17179162800312042,
      0.13299213349819183,
      0.015112901106476784,
      -0.23256167769432068,
      -0.031225159764289856,
      0.1431037038564682
    ],
    [
      0.08786697685718536,
      -0.24513743817806244,
      -0.19820231199264526,
      0.17678967118263245,
      0.2520892918109894,
      -0.08852711319923401,
      -0.08486202359199524,
      0.032729558646678925,
      0.115509033203125,
      -0.1023489385843277,
      0.11271939426660538,
      0.1699495166540146,
      0.1918666809797287,
      -0.03407539799809456,
      -0.13536463677883148,
      0.33414116501808167
    ],
    [
      0.37223654985427856,
      -0.260934978723526,
      -0.3756137490272522,
      0.39443856477737427,
      0.08001770079135895,
      0.086018867790699,
      0.10857027769088745,
      0.2718571722507477,
      -0.03580961748957634,
      -0.06260997802019119,
      -0.2880888283252716,
      0.12338299304246902,
      0.24724476039409637,
      -0.2144414782524109,
      -0.2260747253894806,
      0.417849600315094
    ],
    [
      0.18632443249225616,
      0.1754692941904068,
      -0.05637195333838463,
      0.1762605756521225,
      -0.18562181293964386,
      -0.22669704258441925,
      0.050629228353500366,
      0.26010605692863464,
      -0.01114695705473423,
      0.2660233974456787,
      -0.06672888994216919,
      0.11895853281021118,
      0.14454421401023865,
      -0.22310230135917664,
      -0.23905137181282043,
      0.34413692355155945
    ]
  ],
  "player_model.layers.2.bias": [
    0.15562614798545837,
    0.1678888350725174,
    -0.07601993530988693,
    -0.1428881138563156,
    0.006373410113155842,
    0.2633109390735626,
    0.08159586787223816,
    -0.14982381463050842,
    -0.07022005319595337,
    -0.0337778776884079,
    -0.011498593725264072,
    0.09761454164981842
  ],
  "player_model.layers.4.weight": [
    [
      -0.1930568665266037,
      0.21401552855968475,
      0.23321028053760529,
      -0.1826091706752777,
      0.1419225037097931,
      0.2900327444076538,
      0.2176252007484436,
      -0.046192627400159836,
      -0.3418322801589966,
      -0.11968158185482025,
      0.010555948130786419,
      -0.2182541787624359
    ],
    [
      0.38547247648239136,
      0.40346771478652954,
      0.0074646067805588245,
      0.49673858284950256,
      0.5145406126976013,
      0.0729864239692688,
      0.42907482385635376,
      0.4286305010318756,
      -0.31943485140800476,
      0.022825350984930992,
      -0.43070805072784424,
      -0.6331722736358643
    ],
    [
      -0.19246283173561096,
      -0.061919569969177246,
      -0.009879776276648045,
      -0.268829345703125,
      -0.4975477457046509,
      0.11547698825597763,
      -0.05253829434514046,
      0.053414396941661835,
      0.20861127972602844,
      -0.12037672847509384,
      0.2632844150066376,
      -0.13869516551494598
    ],
    [
      0.12424123287200928,
      0.11346487700939178,
      -0.17557798326015472,
      -0.03147859871387482,
      -0.4333250820636749,
      0.32323262095451355,
      -0.22209914028644562,
      -0.24107173085212708,
      0.014995343051850796,
      0.2523224353790283,
      0.2730155289173126,
      -0.07330304384231567
    ]
  ],
  "player_model.layers.4.bias": [
    -0.00839944463223219,
    0.11146636307239532,
    0.17724576592445374,
    0.004085450433194637
  ],
  "layers.0.weight": [
    [
      -0.10078301280736923,
      0.23462419211864471,
      0.2861880660057068,
      -0.06887756288051605,
      0.25640538334846497,
      -0.037376768887043,
      -0.06091867387294769,
      -0.08438482135534286
    ],
    [
      0.12126979231834412,
      -0.07848096638917923,
      -0.2049761414527893,
      -0.11710754781961441,
      -0.22952723503112793,
      -0.1617956906557083,
      -0.1318512260913849,
      0.07914092391729355
    ],
    [
      0.2738158404827118,
      0.4118530750274658,
      0.18086683750152588,
      0.12536923587322235,
      0.03142452985048294,
      0.1444665640592575,
      0.23033805191516876,
      0.32752272486686707
    ],
    [
      0.11104724556207657,
      -0.17042531073093414,
      0.3877195417881012,
      0.12282896041870117,
      0.13556434214115143,
      0.316124826669693,
      0.052320800721645355,
      0.040639229118824005
    ],
    [
      0.0901518389582634,
      -0.2398773580789566,
      0.26962125301361084,
      0.36462482810020447,
      0.3794559836387634,
      0.08381714671850204,
      -0.3019927144050598,
      -0.04058102145791054
    ],
    [
      0.08293215930461884,
      0.23919427394866943,
      -0.29633644223213196,
      -0.4060794711112976,
      0.13080035150051117,
      0.19103507697582245,
      0.0733085498213768,
      0.019574012607336044
    ],
    [
      0.20661818981170654,
      -0.2993183732032776,
      -0.04697749763727188,
      0.11862613260746002,
      -0.03371891751885414,
      -0.3270721435546875,
      0.22856462001800537,
      0.17607085406780243
    ],
    [
      -0.02454221062362194,
      -0.20483818650245667,
      0.20404057204723358,
      0.11478785425424576,
      -0.28730472922325134,
      0.0596514567732811,
      0.030657609924674034,
      0.10697679966688156
    ],
    [
      0.09719282388687134,
      0.0008757594623602927,
      0.20415067672729492,
      -0.04254965856671333,
      -0.09255670011043549,
      0.20845742523670197,
      -0.5205865502357483,
      -0.3419281244277954
    ],
    [
      -0.07076828926801682,
      -0.04939241707324982,
      0.3044818341732025,
      -0.11844300478696823,
      0.2441040724515915,
      -0.1323791742324829,
      -0.018087876960635185,
      -0.14833644032478333
    ],
    [
      -0.3167954683303833,
      -0.035589881241321564,
      0.5678966641426086,
      0.16428500413894653,
      0.03056907095015049,
      0.3281480371952057,
      -0.06456172466278076,
      -0.3664450943470001
    ],
    [
      0.016440441831946373,
      -0.24026305973529816,
      -0.18504324555397034,
      0.22863009572029114,
      0.15456824004650116,
      -0.12673556804656982,
      0.21311555802822113,
      0.2274317890405655
    ],
    [
      0.21400998532772064,
      -0.16743691265583038,
      0.03422316163778305,
      -0.2777220606803894,
      -0.2874804437160492,
      0.07729250937700272,
      0.17848563194274902,
      0.3256843090057373
    ],
    [
      -0.18705666065216064,
      -0.2676231563091278,
      0.04829960688948631,
      0.3873558044433594,
      -0.107689768075943,
      0.4230426251888275,
      0.15991054475307465,
      0.07624389976263046
    ],
    [
      0.14734135568141937,
      -0.35355260968208313,
      -0.1521139144897461,
      0.35652902722358704,
      0.26079660654067993,
      0.4542412757873535,
      -0.528472900390625,
      0.1417391300201416
    ],
    [
      0.19733312726020813,
      -0.0966641828417778,
      -0.038770340383052826,
      0.2668136656284332,
      0.14723996818065643,
      0.19699598848819733,
      0.08554449677467346,
      -0.32452166080474854
    ]
  ],
  "layers.0.bias": [
    0.24437491595745087,
    0.2443685382604599,
    -0.12394119799137115,
    0.22541098296642303,
    0.25416800379753113,
    -0.2524382174015045,
    0.23530012369155884,
    0.039002835750579834,
    -0.07427021116018295,
    -0.263823539018631,
    -0.11206241697072983,
    -0.10125448554754257,
    0.09616999328136444,
    -0.1914626657962799,
    -0.04643852636218071,
    0.34713032841682434
  ],
  "layers.2.weight": [
    [
      0.024978695437312126,
      0.12467417865991592,
      0.09086883813142776,
      -0.17428983747959137,
      -0.3038393259048462,
      0.23502230644226074,
      0.11998336017131805,
      0.016407396644353867,
      -0.006751313339918852,
      0.08650071173906326,
      -0.1661667823791504,
      0.14999078214168549,
      0.2743881642818451,
      -0.2794640362262726,
      -0.3935360312461853,
      -0.28193140029907227
    ]
  ],
  "layers.2.bias": [
    0.13551369309425354
  ]
}

class PlayerRatingModel(nn.Module):
    def __init__(self):
        super(PlayerRatingModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(20, 16),
            nn.ReLU(),
            nn.Linear(16, 12),
            nn.ReLU(),
            nn.Linear(12, 4),
            nn.ReLU()
        )

    def forward(self, player_stats, game_weight):
        player_output = self.layers(player_stats).squeeze()

        weighted_output = player_output * game_weight.unsqueeze(-1)

        return torch.sum(weighted_output, axis=2) / (torch.sum(game_weight, axis=2).unsqueeze(-1) + 0.001)

class GameRatingModel(nn.Module):
    def __init__(self):
        super(GameRatingModel, self).__init__()

        # Instantiate the player rating model
        self.player_model = PlayerRatingModel()
        self.home_field_advantage = nn.Parameter(torch.tensor(4.5))
        self.layers = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(16, 1)
        )

    def forward(self, home_team_stats, away_team_stats, home_game_weights, away_game_weights,
                home_play_times, away_play_times):
        home_outputs = self.player_model(home_team_stats, home_game_weights)
        away_outputs = self.player_model(away_team_stats, away_game_weights)

        home_ratings = home_outputs * home_play_times.unsqueeze(-1)
        away_ratings = away_outputs * away_play_times.unsqueeze(-1)

        home_team_rating = torch.sum(home_ratings, axis=1)
        away_team_rating = torch.sum(away_ratings, axis=1)

        x = self.layers(torch.cat((home_team_rating, away_team_rating), dim=-1)).squeeze()

        return x + self.home_field_advantage

class NeuralNetwork:
    def __init__(self, elo):
        self.elo = elo

        self.INPUTS_DIM = 19

        self.model = GameRatingModel().to(torch.device('cpu'))
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.003, weight_decay=2e-5)
        self.loss_fn = nn.MSELoss()
        self.retrain_countdown = 0
        self.first_training = True

        self.team_rosters = {}
        self.player_data = defaultdict(list)
        self.player_teams = defaultdict(int)

        self.home_inputs = np.empty((30000, 12, 40, self.INPUTS_DIM + 2), np.float32)
        self.away_inputs = np.empty((30000, 12, 40, self.INPUTS_DIM + 2), np.float32)
        self.home_playtimes = []
        self.away_playtimes = []
        self.outputs = []
        self.training_data = 0

        state_dict = {k: torch.tensor(v) for k, v in pretrained_weights.items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _row_to_inputs(self, row, am_home, my_id, opponent_id, season):
        return [
            self.elo.get_team_strength(my_id, am_home, season) / 100,
            self.elo.get_team_strength(opponent_id, not am_home, season) / 100,
            1 if am_home else 0,        # Whether player is part of home team
            row['MIN'],
            row['FGM'],

            row['FGA'],
            row['FG3M'],
            row['FG3A'],
            row['FTM'],
            row['FTA'],

            row['ORB'],
            row['DRB'],
            row['RB'],
            row['AST'],
            row['STL'],

            row['BLK'],
            row['TOV'],
            row['PF'],
            row['PTS']
        ]

    def _get_team_roster(self, season, team_id, date):
        rosters = self.team_rosters[season][team_id][-5:]

        roster = defaultdict(int)

        for c_roster in rosters:
            for pid, mins in c_roster:
                roster[pid] += mins

        roster = sorted(roster.items(), key=lambda x: x[1], reverse=True)[:12]

        while len(roster) < 12:
            roster.append([-1, 0])

        total_mins = sum(x[1] for x in roster)

        return roster, total_mins

    def _get_game_frame(self, season, date, home_id, away_id):
        season_valid = season in self.team_rosters
        home_valid = season_valid and home_id in self.team_rosters[season] and len(self.team_rosters[season][home_id]) >= 3
        away_valid = season_valid and away_id in self.team_rosters[season] and len(self.team_rosters[season][away_id]) >= 3

        if season_valid and home_valid and away_valid:
            home_roster, home_total_mins = self._get_team_roster(season, home_id, date)
            away_roster, away_total_mins = self._get_team_roster(season, away_id, date)

            if home_total_mins >= 500 and away_total_mins >= 500:
                c_home_inputs = []
                c_home_playtimes = []
                c_away_inputs = []
                c_away_playtimes = []

                for pid, mins in home_roster:
                    c_player_data = []

                    if pid != -1 and pid in self.player_data:
                        c_player_data = copy.deepcopy(self.player_data[pid])

                    for i in range(len(c_player_data)):
                        point_date, point_mins = c_player_data[i][0]
                        time_weight = 0.9965 ** abs((date - point_date).days)
                        c_player_data[i][0] = round(point_mins * time_weight, 3) # Apply time decay

                    while len(c_player_data) < 40:
                        c_player_data.append([0] * (self.INPUTS_DIM + 2))

                    c_home_inputs.append(c_player_data)
                    c_home_playtimes.append(mins / home_total_mins)

                for pid, mins in away_roster:
                    c_player_data = []

                    if pid != -1 and pid in self.player_data:
                        c_player_data = copy.deepcopy(self.player_data[pid])

                    for i in range(len(c_player_data)):
                        point_date, point_mins = c_player_data[i][0]
                        time_weight = 0.9965 ** abs((date - point_date).days)
                        c_player_data[i][0] = round(point_mins * time_weight, 3) # Apply time decay

                    while len(c_player_data) < 40:
                        c_player_data.append([0] * (self.INPUTS_DIM + 2))

                    c_away_inputs.append(c_player_data)
                    c_away_playtimes.append(mins / away_total_mins)

                return c_home_inputs, c_home_playtimes, c_away_inputs, c_away_playtimes

        return None

    # def _train(self, dataloader):
    #     self.model.train()
    #     total_loss = 0.0
    #     total_correct = 0
    #     total_samples = 0

    #     for batch_idx, (home_team_stats, away_team_stats, home_game_weights, away_game_weights, home_play_times, away_play_times, true_score_diff) in enumerate(dataloader):
    #         # Forward pass
    #         predicted_score_diff = self.model(home_team_stats, away_team_stats, home_game_weights, away_game_weights,
    #                                     home_play_times, away_play_times)

    #         # Compute loss
    #         loss = self.loss_fn(predicted_score_diff, true_score_diff)

    #         # Apply sample weights: weight decays with distance from the most recent sample
    #         batch_size = true_score_diff.size(0)
    #         weights = torch.tensor([0.99984 ** (len(dataloader.dataset) - (batch_idx * batch_size + i))
    #                                 for i in range(batch_size)], dtype=torch.float32)
    #         weights = weights.to(loss.device)
    #         weighted_loss = (loss * weights).mean()  # Apply weights to loss

    #         # Calculate binary accuracy (direction match)
    #         predicted_sign = torch.sign(predicted_score_diff)
    #         true_sign = torch.sign(true_score_diff)
    #         correct = (predicted_sign == true_sign).sum().item()
    #         total_correct += correct
    #         total_samples += true_score_diff.size(0)

    #         # Backpropagation and optimization
    #         self.optimizer.zero_grad()
    #         weighted_loss.backward()
    #         self.optimizer.step()

    #         # Accumulate loss
    #         total_loss += weighted_loss.item()

    #     # Calculate average loss and binary accuracy for this epoch
    #     avg_loss = total_loss / len(dataloader)
    #     accuracy = total_correct / total_samples
    #     return avg_loss, accuracy

    def pre_add_game(self, current, current_players):
        pass
        # season = current['Season']
        # home_id = current['HID']
        # away_id = current['AID']
        # home_score = current['HSC']
        # away_score = current['ASC']
        # date = current['Date']

        # game_frame = self._get_game_frame(season, date, home_id, away_id)

        # if game_frame is not None:
        #     c_home_inputs, c_home_playtimes, c_away_inputs, c_away_playtimes = game_frame

        #     self.home_inputs[self.training_data] = np.array(c_home_inputs, np.float32)
        #     self.away_inputs[self.training_data] = np.array(c_away_inputs, np.float32)
        #     self.home_playtimes.append(c_home_playtimes)
        #     self.away_playtimes.append(c_away_playtimes)
        #     self.outputs.append((abs(home_score - away_score) + 3) ** 0.7 * (1 if home_score > away_score else -1))
        #     self.retrain_countdown -= 1
        #     self.training_data += 1

    def add_game(self, current, current_players):
        season = current['Season']
        home_id = current['HID']
        away_id = current['AID']
        date = current['Date']

        home_players = current_players[current_players['Team'] == home_id]
        away_players = current_players[current_players['Team'] == away_id]

        players_on_a_team_map = {}

        for _, player in current_players.iterrows():
            key = f"{player['Player']}|{player['Team']}"
            players_on_a_team_map[player['Player']] = math.log(1 + self.player_teams[key])
            self.player_teams[key] += 1

        if season not in self.team_rosters:
            self.team_rosters[season] = {}

        if home_id not in self.team_rosters[season]:
            self.team_rosters[season][home_id] = []

        if away_id not in self.team_rosters[season]:
            self.team_rosters[season][away_id] = []

        self.team_rosters[season][home_id].append([[x['Player'], x['MIN']] for _, x in home_players.iterrows()])
        self.team_rosters[season][away_id].append([[x['Player'], x['MIN']] for _, x in away_players.iterrows()])

        mapped_home_players = [{
            'pid': row['Player'],
            'mins': row['MIN'],
            'inputs': self._row_to_inputs(row, True, home_id, away_id, season)
        } for _, row in home_players.iterrows()]
        mapped_away_players = [{
            'pid': row['Player'],
            'mins': row['MIN'],
            'inputs': self._row_to_inputs(row, False, away_id, home_id, season)
        } for _, row in away_players.iterrows()]

        for data in [*mapped_home_players, *mapped_away_players]:
            if not any(math.isnan(x) for x in data['inputs']):
                self.player_data[data['pid']].append([[date, data['mins']], *data['inputs'], players_on_a_team_map[data['pid']]])
                self.player_data[data['pid']] = self.player_data[data['pid']][-40:]

    def get_input_data(self, home_id, away_id, season, date):
        game_frame = self._get_game_frame(season, date, home_id, away_id)

        if game_frame is None:
            return None

        # if self.retrain_countdown <= 0:
        #     self.retrain_countdown = 2500

        #     print('\nRetraining! Preparing dataset...')

        #     home_team_stats = torch.from_numpy(self.home_inputs[:self.training_data, :, :, 1:])
        #     away_team_stats = torch.from_numpy(self.away_inputs[:self.training_data, :, :, 1:])
        #     home_game_weights = torch.from_numpy(self.home_inputs[:self.training_data, :, :, 0])
        #     away_game_weights = torch.from_numpy(self.away_inputs[:self.training_data, :, :, 0])
        #     home_play_times = torch.tensor(np.array(self.home_playtimes).astype(np.float32), dtype=torch.float32)
        #     away_play_times = torch.tensor(np.array(self.away_playtimes).astype(np.float32), dtype=torch.float32)
        #     true_score_diff = torch.tensor(np.array(self.outputs), dtype=torch.float32)

        #     # Prepare DataLoader
        #     train_data = TensorDataset(home_team_stats, away_team_stats, home_game_weights, away_game_weights, home_play_times, away_play_times, true_score_diff)
        #     train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

        #     print('\nRetraining!')

        #     num_epochs = 40 if self.first_training else 10
        #     self.first_training = False
        #     for epoch in range(num_epochs):
        #         train_loss, train_accuracy = self._train(train_loader)

        #         print(f'Epoch {epoch + 1} / {num_epochs}, train_loss: {train_loss:.4f}, train_accuracy: {train_accuracy:.4f}; N={self.training_data}')

        c_home_inputs, c_home_playtimes, c_away_inputs, c_away_playtimes = game_frame
        np_array_home_inputs = np.array(c_home_inputs).astype(np.float32)
        np_array_away_inputs = np.array(c_away_inputs).astype(np.float32)

        home_team_stats = torch.tensor(np_array_home_inputs[:, :, 1:], dtype=torch.float32).unsqueeze(0)
        away_team_stats = torch.tensor(np_array_away_inputs[:, :, 1:], dtype=torch.float32).unsqueeze(0)
        home_game_weights = torch.tensor(np_array_home_inputs[:, :, 0], dtype=torch.float32).unsqueeze(0)
        away_game_weights = torch.tensor(np_array_away_inputs[:, :, 0], dtype=torch.float32).unsqueeze(0)
        home_play_times = torch.tensor(np.array(c_home_playtimes).astype(np.float32), dtype=torch.float32).unsqueeze(0)
        away_play_times = torch.tensor(np.array(c_away_playtimes).astype(np.float32), dtype=torch.float32).unsqueeze(0)

        self.model.eval()

        with torch.no_grad():
            prediction = self.model(home_team_stats, away_team_stats, home_game_weights, away_game_weights, home_play_times, away_play_times)

        if season <= 24:
            return [
                prediction.item() + random.random() - 0.5
            ]
        else:
            return [
                prediction.item()
            ]



from collections import defaultdict
from datetime import datetime

class Pythagorean:
    def __init__(self, power=16.5, regularization=120, daily_decay=0.992):
        self.power = power
        self.regularization = regularization
        self.daily_decay = daily_decay

        self.team_map = defaultdict(list)

    def pre_add_game(self, current, current_players):
        pass

    def add_game(self, current, current_players):
        date = current['Date']
        home_id = current['HID']
        away_id = current['AID']
        home_score = current['HSC']
        away_score = current['ASC']

        if type(date) == 'str':
            date = datetime.strptime(date, '%Y-%m-%d')

        home_difference = abs(home_score - away_score) ** self.power * (1 if home_score > away_score else -1)
        away_difference = -home_difference

        self.team_map[home_id].append([date, home_score, away_score])
        self.team_map[away_id].append([date, away_score, home_score])

    def _get_weighted(self, team_id, idx, date):
        return sum([x[idx] * (self.daily_decay ** abs((date - x[0]).days)) for x in self.team_map[team_id][-100:]])

    def get_input_data(self, home_id, away_id, season, date):
        if type(date) == 'str':
            date = datetime.strptime(date, '%Y-%m-%d')

        home_scored = self.regularization + self._get_weighted(home_id, 1, date)
        home_allowed = self.regularization + self._get_weighted(home_id, 2, date)
        away_scored = self.regularization + self._get_weighted(away_id, 1, date)
        away_allowed = self.regularization + self._get_weighted(away_id, 2, date)

        return [
            (home_scored ** self.power) / (home_scored ** self.power + home_allowed ** self.power),
            (away_scored ** self.power) / (away_scored ** self.power + away_allowed ** self.power)
        ]


import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def inverse_sigmoid(x):
    return math.log(x / (1 - x))

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class Model:
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode

        # Hyperparameters
        self.ensamble_required_n = 2000
        nate_silver_elo = NateSilverElo()
        self.model_list = [
            Pythagorean(),      # 0.8464013687751296, -0.06697869116809001
            FourFactor(),       # -0.037466710615323806
            GradientDescent(),  # 0.8539410540350695
            Exhaustion(),       # -0.30556362733411674
            nate_silver_elo,    # 0.002608191859624124
            NeuralNetwork(nate_silver_elo)
        ]
        # End

        self.coef_list = []

        self.prediction_map = {}
        self.input_map = {}
        self.coef_map = {}
        self.past_pred = []
        self.ensamble = None
        self.ensamble_retrain = 0

        self.pred_list = []

        self.pred_metrics = {
            'my_ba': 0,
            'mkt_ba': 0,
            'my_mse': 0,
            'mkt_mse': 0,
            'corr_me': [],
            'corr_mkt': [],
            'n': 0
        }

        self.bet_metrics = {
            'exp_pnl': 0,
            'opps': 0,
            'count': 0,
            'volume': 0,
            'sum_odds': 0
        }

    def _get_input_features(self, home_id, away_id, season, date):
        input_data = []

        for model in self.model_list:
            data = model.get_input_data(home_id, away_id, season, date)

            if data is None:
                return None

            input_data = [
                *input_data,
                *data
            ]

        return input_data

    def _handle_metrics(self, idx, current):
        if idx in self.prediction_map:
            pred = self.prediction_map[idx]

            home_win = current['H']
            odds_home = current['OddsH']
            odds_away = current['OddsA']
            overround = 1 / odds_home + 1 / odds_away
            mkt_pred = 1 / odds_home / overround

            if pred == 0.5:
                self.pred_metrics['my_ba'] += 0.5
            elif (pred > 0.5) == home_win:
                self.pred_metrics['my_ba'] += 1

            if mkt_pred == 0.5:
                self.pred_metrics['mkt_ba'] += 0.5
            elif (mkt_pred > 0.5) == home_win:
                self.pred_metrics['mkt_ba'] += 1

            self.pred_metrics['my_mse'] += (pred - home_win) ** 2
            self.pred_metrics['mkt_mse'] += (mkt_pred - home_win) ** 2
            self.pred_metrics['n'] += 1

            self.pred_metrics['corr_me'].append(pred)
            self.pred_metrics['corr_mkt'].append(mkt_pred)

            self.pred_list.append({
                'index': str(idx),
                'neutral': int(current['N']),
                'playoff': int(current['POFF']),
                'date': str(current['Date']),
                'season': int(current['Season']),
                'score': int(current['HSC'] - current['ASC']),
                'my_pred': pred,
                'mkt_pred': mkt_pred,
                'odds_home': float(odds_home),
                'odds_away': float(odds_away),
                'outcome': int(home_win),
                'inputs': self.input_map[idx],
                'coefs': self.coef_map[idx]
            })

    def _game_increment(self, idx, current, current_players):
        season = current['Season']
        date = current['Date']
        home_id = current['HID']
        away_id = current['AID']
        home_win = current['H']
        year = int(str(current['Date'])[0:4])

        if year >= 1996:
            input_arr = self._get_input_features(home_id, away_id, season, date)

            if input_arr is not None:
                self.past_pred.append([*input_arr, home_win])
                self.ensamble_retrain -= 1

        self._handle_metrics(idx, current)

        if year >= 1990:
            # Let the models create training frames before new data arrives
            for model in self.model_list:
                model.pre_add_game(current, current_players)

            # Add new data to the models
            for model in self.model_list:
                model.add_game(current, current_players)

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame]):
        games_increment, players_increment = inc

        done = 0
        total = len(games_increment)

        for idx in games_increment.index:
            current = games_increment.loc[idx]
            current_players = players_increment[(players_increment['Game'] == idx) & (players_increment['MIN'] >= 3)]

            self._game_increment(idx, current, current_players)
            done += 1

        min_bet = summary.iloc[0]['Min_bet']
        max_bet = summary.iloc[0]['Max_bet']
        bankroll = summary.iloc[0]['Bankroll']
        my_bet = max(min_bet, min(max_bet, summary.iloc[0]['Bankroll'] * 0.08))

        bets = pd.DataFrame(data=np.zeros((len(opps), 2)), columns=['BetH', 'BetA'], index=opps.index)

        for i in opps.index:
            current = opps.loc[i]

            season = current['Season']
            date = current['Date']
            home_id = current['HID']
            away_id = current['AID']
            playoff = current['POFF'] == 1

            if len(self.past_pred) >= self.ensamble_required_n:
                input_arr = self._get_input_features(home_id, away_id, season, date)

                if input_arr is not None:
                    if self.ensamble_retrain <= 0:
                        self.ensamble_retrain = 400
                        np_array = np.array(self.past_pred)
                        sample_weights = np.exp(-0.0003 * np.arange(len(self.past_pred)))
                        self.ensamble = LogisticRegression(max_iter=10000)
                        self.ensamble.fit(np_array[:, :-1], np_array[:, -1], sample_weight=sample_weights[::-1])

                        self.coef_list.append({
                            'index': i,
                            'date': str(date),
                            'coefs': self.ensamble.coef_.tolist(),
                            'intercept': self.ensamble.intercept_.tolist(),
                            'sum_weight': sample_weights.sum(),
                            'len': len(self.past_pred)
                        })

                    self.bet_metrics['opps'] += 1

                    pred = self.ensamble.predict_proba(np.array([input_arr]))[0, 1]

                    self.prediction_map[i] = pred
                    self.input_map[i] = input_arr
                    self.coef_map[i] = [self.ensamble.intercept_.tolist(), *self.ensamble.coef_.tolist()]

                    # Adjust for playoffs
                    # adj_pred = pred # sigmoid((inverse_sigmoid(pred) + 0.2) * 1.1) if playoff else pred
                    adj_pred = sigmoid((inverse_sigmoid(pred) + 0.1) * 1.05) if playoff else sigmoid(inverse_sigmoid(pred) * 1.03)

                    odds_home = current['OddsH']
                    odds_away = current['OddsA']

                    min_home_odds = (1 / adj_pred + 0.02) if bankroll > 4000 else ((1 / adj_pred - 1) * 1.1 + 1.04)
                    min_away_odds = (1 / (1 - adj_pred) + 0.02) if bankroll > 4000 else ((1 / (1 - adj_pred) - 1) * 1.1 + 1.04)

                    if odds_home >= min_home_odds:
                        bets.at[i, 'BetH'] = my_bet

                        self.bet_metrics['exp_pnl'] += adj_pred * odds_home - 1
                        self.bet_metrics['volume'] += my_bet
                        self.bet_metrics['count'] += 1
                        self.bet_metrics['sum_odds'] += odds_home

                    if odds_away >= min_away_odds:
                        bets.at[i, 'BetA'] = my_bet

                        self.bet_metrics['exp_pnl'] += (1 - adj_pred) * odds_away - 1
                        self.bet_metrics['volume'] += my_bet
                        self.bet_metrics['count'] += 1
                        self.bet_metrics['sum_odds'] += odds_away

        return bets
