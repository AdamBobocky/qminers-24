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
    def __init__(self, num_teams=50, monthly_decay=0.8, long_term_decay=0.97):
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

    def _fit(self, max_epochs=24):
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
  "home_field_advantage": 4.319807052612305,
  "player_model.layers.0.weight": [
    [
      0.37389135360717773,
      0.11988319456577301,
      -0.34011974930763245,
      0.12640340626239777,
      0.2692384123802185,
      0.022081924602389336,
      0.31633269786834717,
      0.17248649895191193,
      0.12291429936885834,
      0.04477522149682045,
      -0.10811631381511688,
      0.196944460272789,
      -0.04629145935177803,
      0.0402073860168457,
      0.0797380805015564,
      0.23847530782222748,
      -0.15636901557445526,
      -0.12447795271873474,
      0.02687535434961319,
      0.20644980669021606
    ],
    [
      0.29864683747291565,
      0.09291031956672668,
      -0.28557175397872925,
      0.07264544814825058,
      0.003734811209142208,
      0.11825118958950043,
      0.1570771485567093,
      0.0002458452945575118,
      -0.01769884116947651,
      -0.11278612911701202,
      0.19086609780788422,
      -0.051299937069416046,
      0.06396471709012985,
      0.011142655275762081,
      -0.01866457611322403,
      0.13744784891605377,
      -0.03829261288046837,
      0.029169507324695587,
      0.030294353142380714,
      -0.07931890338659286
    ],
    [
      0.02693902887403965,
      0.0024827448651194572,
      0.24131731688976288,
      0.2555604875087738,
      -0.11979260295629501,
      0.12212089449167252,
      0.19890046119689941,
      0.23617681860923767,
      -0.20316636562347412,
      -0.09990095347166061,
      0.14195360243320465,
      0.17850002646446228,
      0.13315004110336304,
      -0.13740329444408417,
      0.014850021339952946,
      -0.23906797170639038,
      -0.044790420681238174,
      -0.03517507389187813,
      0.08271241188049316,
      -0.08524283021688461
    ],
    [
      0.15744948387145996,
      -0.022720104083418846,
      -0.3548385798931122,
      0.1338207721710205,
      -0.19096536934375763,
      -0.1524030566215515,
      0.10152382403612137,
      -0.006234739441424608,
      0.018528278917074203,
      0.15681537985801697,
      0.05110105499625206,
      0.0937313660979271,
      0.05423959344625473,
      -0.002458835719153285,
      0.28067049384117126,
      0.05640852823853493,
      -0.19987478852272034,
      -0.20913313329219818,
      0.15318477153778076,
      -0.011048495769500732
    ],
    [
      0.45154431462287903,
      -0.019775094464421272,
      -0.4699123203754425,
      -0.00727640837430954,
      -0.08123307675123215,
      0.17183838784694672,
      0.20481976866722107,
      0.32088345289230347,
      0.15966570377349854,
      0.08459346741437912,
      -0.01895103044807911,
      0.20862185955047607,
      0.13563768565654755,
      0.13329274952411652,
      0.1859484165906906,
      0.528512716293335,
      -0.03444962948560715,
      -0.07901006191968918,
      -0.012757716700434685,
      0.04570210725069046
    ],
    [
      -0.2699805200099945,
      -0.12024735659360886,
      0.6468359231948853,
      0.2774874269962311,
      -0.08414441347122192,
      0.2847532033920288,
      0.02740437723696232,
      -0.10455577075481415,
      0.05968422442674637,
      0.21712948381900787,
      0.16610965132713318,
      -0.08717956393957138,
      0.0007216626545414329,
      -0.10644692182540894,
      0.06144983321428299,
      -0.02490982413291931,
      -0.07756339013576508,
      0.19177107512950897,
      0.05027597025036812,
      0.048062123358249664
    ],
    [
      -0.3439634144306183,
      0.03043367527425289,
      0.3716602623462677,
      0.20473507046699524,
      -0.10401300340890884,
      0.07386911660432816,
      0.0637153908610344,
      0.09142804145812988,
      -0.00470511382445693,
      0.020349912345409393,
      -0.1501891016960144,
      -0.12858127057552338,
      0.14608442783355713,
      -0.12642022967338562,
      0.036372601985931396,
      -0.032443296164274216,
      0.01459538284689188,
      0.2676667273044586,
      0.05033895745873451,
      -0.0725986585021019
    ],
    [
      0.33239760994911194,
      -0.06723570078611374,
      -0.6584975719451904,
      0.05843202769756317,
      0.18231065571308136,
      -0.1729595810174942,
      0.2043548822402954,
      0.04611169546842575,
      0.10749813169240952,
      -0.1109543889760971,
      0.0009854619856923819,
      0.07291288673877716,
      0.07703252881765366,
      0.21156492829322815,
      0.18500621616840363,
      0.14662402868270874,
      -0.20837090909481049,
      0.08503574877977371,
      0.13113810122013092,
      0.1283464878797531
    ],
    [
      -0.07209129631519318,
      -0.15427568554878235,
      0.46097248792648315,
      0.18897110223770142,
      -0.036384765058755875,
      0.17629119753837585,
      0.221999391913414,
      -0.04295622929930687,
      0.05012589693069458,
      -0.17281077802181244,
      0.12443332374095917,
      0.15025006234645844,
      0.022470496594905853,
      0.1779414415359497,
      -0.11805625259876251,
      -0.16705270111560822,
      0.049284398555755615,
      0.2728060781955719,
      -0.06393484771251678,
      0.002143179066479206
    ],
    [
      -0.3087228536605835,
      0.08426835387945175,
      0.34985604882240295,
      0.15110161900520325,
      0.1703380048274994,
      0.30089879035949707,
      -0.08355028182268143,
      0.07619039714336395,
      -0.10608082264661789,
      0.21145036816596985,
      0.18120962381362915,
      0.27090832591056824,
      0.1569015383720398,
      -0.11791951209306717,
      0.014930575154721737,
      -0.11917694658041,
      -0.012563377618789673,
      -0.0008306218660436571,
      0.10992124676704407,
      0.35202690958976746
    ],
    [
      0.3869883716106415,
      0.02825726941227913,
      -0.3295108377933502,
      -0.050808195024728775,
      0.12425056844949722,
      0.13366717100143433,
      0.1750984638929367,
      0.08890961855649948,
      0.19246752560138702,
      0.07447995990514755,
      -0.023515766486525536,
      0.07564892619848251,
      0.21340033411979675,
      0.2558361291885376,
      0.3119346797466278,
      0.252532422542572,
      -0.06458856910467148,
      -0.10173272341489792,
      0.1137487143278122,
      0.15721531212329865
    ],
    [
      -0.13908261060714722,
      0.08509223163127899,
      0.4868204593658447,
      0.07668604701757431,
      -0.04604581370949745,
      0.23691463470458984,
      0.031086552888154984,
      0.2118554711341858,
      -0.15856534242630005,
      -0.19403791427612305,
      -0.080867238342762,
      0.05919799208641052,
      -0.04999587684869766,
      -0.16075854003429413,
      -0.29334113001823425,
      0.016570284962654114,
      0.011145500466227531,
      -0.03521528095006943,
      -0.011491009034216404,
      0.14527802169322968
    ],
    [
      -0.34721556305885315,
      -0.09931347519159317,
      0.43927180767059326,
      0.06577851623296738,
      0.17749804258346558,
      -0.04257313907146454,
      -0.22321270406246185,
      0.0507514663040638,
      -0.018085042014718056,
      -0.16977283358573914,
      -0.12360019981861115,
      0.06263862550258636,
      -0.038016147911548615,
      -0.32154417037963867,
      -0.01775326207280159,
      -0.06988157331943512,
      0.18211747705936432,
      0.2936754524707794,
      0.05435242876410484,
      -0.11096874624490738
    ],
    [
      -0.31551480293273926,
      -0.033916231244802475,
      0.6637702584266663,
      -0.0037313976790755987,
      0.1746395379304886,
      0.17482803761959076,
      0.09339966624975204,
      -0.17614005506038666,
      -0.13902249932289124,
      -0.01905808039009571,
      0.07181733846664429,
      -0.11689239740371704,
      0.1546342372894287,
      -0.15421482920646667,
      -0.009965021163225174,
      -0.2500615417957306,
      0.09636514633893967,
      0.2180270105600357,
      0.048432160168886185,
      0.1354074627161026
    ],
    [
      0.22436100244522095,
      0.03176896646618843,
      -0.3116568922996521,
      -0.09672002494335175,
      0.0772867277264595,
      -0.005550897214561701,
      -0.08410739153623581,
      -0.16477814316749573,
      0.23450864851474762,
      -0.1118999570608139,
      0.042162325233221054,
      0.07064886391162872,
      0.23524309694766998,
      0.3543098568916321,
      0.25572583079338074,
      0.17283253371715546,
      0.06650911271572113,
      0.14207690954208374,
      0.2230011224746704,
      -0.14893899857997894
    ],
    [
      0.05982903763651848,
      -0.17984022200107574,
      -0.29889124631881714,
      -0.19174160063266754,
      0.11742246896028519,
      0.0002157585695385933,
      0.13736356794834137,
      -0.038707587867975235,
      0.018926402553915977,
      -0.25380977988243103,
      -0.16374951601028442,
      -0.21814855933189392,
      -0.12484470009803772,
      -0.16046026349067688,
      0.1264430731534958,
      -0.18707376718521118,
      -0.005053089465945959,
      -0.05674360319972038,
      0.09544471651315689,
      -0.2861688733100891
    ]
  ],
  "player_model.layers.0.bias": [
    -0.05893964320421219,
    -0.5701621174812317,
    0.47184261679649353,
    -0.5351146459579468,
    -0.39362502098083496,
    0.6977140307426453,
    0.3732665777206421,
    -0.5951244235038757,
    0.08013898134231567,
    0.38736972212791443,
    -0.40624600648880005,
    0.4749740660190582,
    0.6920029520988464,
    0.7880406379699707,
    -0.22116056084632874,
    0.09278508275747299
  ],
  "player_model.layers.2.weight": [
    [
      0.0940244048833847,
      -0.17661158740520477,
      -0.10038086771965027,
      -0.30146753787994385,
      -0.2792845070362091,
      0.36468252539634705,
      0.22328005731105804,
      -0.41232967376708984,
      0.15983571112155914,
      0.053327061235904694,
      -0.35859277844429016,
      0.3365015387535095,
      0.4967169463634491,
      0.16009046137332916,
      -0.3263615667819977,
      -0.32606637477874756
    ],
    [
      -0.07929801940917969,
      -0.18554475903511047,
      0.22325845062732697,
      -0.20732742547988892,
      0.04950624704360962,
      0.11527501046657562,
      0.10869163274765015,
      -0.30144959688186646,
      0.3174549341201782,
      0.21324731409549713,
      -0.12754827737808228,
      0.32106655836105347,
      -0.026691075414419174,
      0.11225514858961105,
      -0.23910121619701385,
      -0.03795464336872101
    ],
    [
      0.18490320444107056,
      0.26553258299827576,
      0.05808371677994728,
      0.061332087963819504,
      0.05631687864661217,
      -0.08629864454269409,
      0.11170874536037445,
      0.42344552278518677,
      -0.14817069470882416,
      0.08988445997238159,
      0.29956215620040894,
      -0.02996893785893917,
      -0.08149264007806778,
      -0.2903056740760803,
      0.17875146865844727,
      0.09940223395824432
    ],
    [
      0.12378643453121185,
      -0.06840407103300095,
      0.03470179811120033,
      0.15721257030963898,
      -0.11804171651601791,
      -0.20015256106853485,
      0.06968314200639725,
      -0.06116228178143501,
      -0.15635231137275696,
      0.13769739866256714,
      0.1823887676000595,
      0.015631427988409996,
      0.23300941288471222,
      -0.09932965040206909,
      -0.19412921369075775,
      -0.14890606701374054
    ],
    [
      -0.12531037628650665,
      -0.02495519071817398,
      0.17351509630680084,
      -0.15588432550430298,
      -0.18504638969898224,
      0.32777145504951477,
      0.2819557189941406,
      -0.4963802993297577,
      0.18728117644786835,
      0.06201654300093651,
      -0.08893566578626633,
      0.028289316222071648,
      0.4801463484764099,
      0.41794058680534363,
      -0.3846699893474579,
      -0.27981817722320557
    ],
    [
      0.2518689036369324,
      0.1121487021446228,
      0.0031461790204048157,
      -0.05519532039761543,
      0.3610527515411377,
      -0.1228405088186264,
      0.1965676248073578,
      -0.0073067788034677505,
      0.244467630982399,
      -0.1024237871170044,
      0.15355053544044495,
      0.22138839960098267,
      0.04820859804749489,
      -0.35974204540252686,
      -0.11207567900419235,
      -0.022391831502318382
    ],
    [
      0.027292262762784958,
      0.04646054655313492,
      0.014716905541718006,
      0.14935436844825745,
      0.037699613720178604,
      0.01743161678314209,
      -0.17629393935203552,
      0.017086589708924294,
      0.12034326046705246,
      -0.0019047284731641412,
      0.2844284176826477,
      -0.156748965382576,
      0.05373214930295944,
      -0.27691134810447693,
      0.33077818155288696,
      0.48096558451652527
    ],
    [
      0.32400748133659363,
      0.15669451653957367,
      0.12553884088993073,
      0.14752499759197235,
      0.42823106050491333,
      -0.08786383271217346,
      -0.1848987191915512,
      0.45134028792381287,
      0.06895550340414047,
      0.09426426142454147,
      0.39240673184394836,
      -0.1660914272069931,
      -0.5021517872810364,
      -0.09516231715679169,
      0.43836045265197754,
      -0.18058352172374725
    ],
    [
      0.17225363850593567,
      -0.19048850238323212,
      0.05043991655111313,
      0.09686318039894104,
      -0.28025153279304504,
      0.27312198281288147,
      0.2994367480278015,
      -0.2391519695520401,
      0.2709318995475769,
      0.12876468896865845,
      -0.09529197961091995,
      0.15234678983688354,
      0.16784194111824036,
      0.1290750801563263,
      -0.13613440096378326,
      -0.17364658415317535
    ],
    [
      0.07048056274652481,
      -0.1944051831960678,
      0.12228671461343765,
      -0.21727094054222107,
      -0.1531529724597931,
      0.08043363690376282,
      0.010432856157422066,
      -0.04897434264421463,
      0.16108092665672302,
      0.01900634542107582,
      -0.014229552820324898,
      -0.07497675716876984,
      0.24923034012317657,
      0.3132748007774353,
      -0.06532134115695953,
      -0.26295626163482666
    ],
    [
      0.17436926066875458,
      0.18744981288909912,
      -0.03296302258968353,
      0.3452405035495758,
      0.19048745930194855,
      -0.025623967871069908,
      -0.13507363200187683,
      0.3941490948200226,
      -0.09485787898302078,
      0.026519063860177994,
      0.011849900707602501,
      -0.08209353685379028,
      0.17154096066951752,
      0.09195487201213837,
      0.11261170357465744,
      0.16767802834510803
    ],
    [
      0.208695650100708,
      -0.10922366380691528,
      -0.2078099250793457,
      0.23088552057743073,
      0.0812453106045723,
      -0.10290313512086868,
      0.1137319952249527,
      0.35284990072250366,
      0.25055384635925293,
      0.15091359615325928,
      0.37046077847480774,
      -0.1349339336156845,
      -0.24368640780448914,
      -0.09392492473125458,
      0.25870048999786377,
      -0.1344081163406372
    ]
  ],
  "player_model.layers.2.bias": [
    0.7178093194961548,
    0.5103786587715149,
    -0.333748996257782,
    0.17889715731143951,
    0.691243588924408,
    -0.22625795006752014,
    0.07248515635728836,
    -0.4616697132587433,
    0.5192729830741882,
    0.1474040150642395,
    -0.1600911021232605,
    -0.4567277133464813
  ],
  "player_model.layers.4.weight": [
    [
      0.6491722464561462,
      0.37938275933265686,
      -0.3428036570549011,
      -0.12898755073547363,
      0.297767698764801,
      -0.24133989214897156,
      -0.15467000007629395,
      -0.49498215317726135,
      0.4592229723930359,
      0.4715689420700073,
      -0.3908722698688507,
      -0.47881558537483215
    ],
    [
      0.31964433193206787,
      0.13925911486148834,
      0.02131134271621704,
      0.3330099284648895,
      0.19308023154735565,
      0.05641528591513634,
      -0.021739987656474113,
      -0.05807502940297127,
      0.27570512890815735,
      0.1255892813205719,
      -0.25853443145751953,
      -0.25268155336380005
    ],
    [
      0.31551623344421387,
      0.420623779296875,
      -0.28420788049697876,
      0.22869333624839783,
      0.5735945701599121,
      -0.1084773913025856,
      -0.5097799301147461,
      -0.544208824634552,
      0.4610756039619446,
      0.18917511403560638,
      -0.2693314850330353,
      -0.34257468581199646
    ],
    [
      -0.15230174362659454,
      0.20118822157382965,
      0.11663538217544556,
      0.03267214819788933,
      -0.10340960323810577,
      0.32534945011138916,
      -0.24024182558059692,
      0.042140617966651917,
      0.26924434304237366,
      0.29039308428764343,
      0.0035408437252044678,
      0.018783478066325188
    ]
  ],
  "player_model.layers.4.bias": [
    0.4317394196987152,
    0.1741950809955597,
    0.7098283171653748,
    -0.05828246474266052
  ],
  "layers.0.weight": [
    [
      -0.13931095600128174,
      0.3028431236743927,
      0.08319688588380814,
      0.030986012890934944,
      -0.04215754196047783,
      -0.29516535997390747,
      -0.2576371133327484,
      -0.34199124574661255
    ],
    [
      0.34783339500427246,
      0.0935913696885109,
      0.023227263242006302,
      0.1581783890724182,
      0.06348421424627304,
      0.358211874961853,
      0.07615787535905838,
      0.26462993025779724
    ],
    [
      0.06523840129375458,
      0.08800099790096283,
      -0.10511672496795654,
      -0.2851060926914215,
      -0.33426418900489807,
      0.08361256867647171,
      -0.15361833572387695,
      0.07260791957378387
    ],
    [
      -0.3370916545391083,
      -0.08051106333732605,
      0.11607253551483154,
      -0.33671534061431885,
      0.0729348361492157,
      -0.3331694006919861,
      -0.04175027459859848,
      0.019758770242333412
    ],
    [
      0.40907397866249084,
      -0.057940881699323654,
      -0.004797422327101231,
      -0.02187163010239601,
      0.020176948979496956,
      0.31125426292419434,
      -0.0335070863366127,
      0.18593396246433258
    ],
    [
      -0.16967158019542694,
      -0.17253807187080383,
      -0.27007144689559937,
      0.23534223437309265,
      0.03171251341700554,
      -0.029453841969370842,
      -0.3129371404647827,
      -0.06792832165956497
    ],
    [
      -0.03482658043503761,
      0.02689289301633835,
      -0.16464126110076904,
      0.19504407048225403,
      0.07068828493356705,
      0.3103516399860382,
      0.3687938451766968,
      -0.25098419189453125
    ],
    [
      -0.33813899755477905,
      0.24729833006858826,
      0.1994592249393463,
      -0.2910372018814087,
      -0.07130687683820724,
      -0.26069536805152893,
      0.1077120453119278,
      -0.3339792490005493
    ],
    [
      0.018863338977098465,
      -0.18294601142406464,
      0.08711659908294678,
      -0.13650859892368317,
      -0.2272995114326477,
      0.32916519045829773,
      -0.3132982552051544,
      -0.14231480658054352
    ],
    [
      0.10894249379634857,
      -0.026149295270442963,
      -0.13138456642627716,
      -0.07013383507728577,
      0.27963149547576904,
      0.04144966974854469,
      0.35683363676071167,
      0.13351325690746307
    ],
    [
      0.43084990978240967,
      0.25319308042526245,
      0.5264074206352234,
      0.13415443897247314,
      -0.18318751454353333,
      0.20514079928398132,
      -0.22193317115306854,
      0.29349732398986816
    ],
    [
      -0.22030064463615417,
      0.07593195885419846,
      -0.12062668055295944,
      0.03673162683844566,
      -0.14230485260486603,
      0.17400206625461578,
      0.02355160564184189,
      0.2806553244590759
    ],
    [
      -0.2187766134738922,
      0.2327573150396347,
      0.032669778913259506,
      0.08747897297143936,
      0.225210502743721,
      -0.33946943283081055,
      -0.02004111558198929,
      -0.22463533282279968
    ],
    [
      0.02967781014740467,
      0.17782041430473328,
      0.30431580543518066,
      -0.029171762987971306,
      -0.3293571174144745,
      -0.038661979138851166,
      -0.21470221877098083,
      0.4179690182209015
    ],
    [
      -0.034518759697675705,
      -0.3465452492237091,
      -0.017326008528470993,
      -0.11122157424688339,
      -0.24855151772499084,
      0.21300970017910004,
      -0.1499750316143036,
      0.25560227036476135
    ],
    [
      -0.28322547674179077,
      0.010763905942440033,
      -0.11759250611066818,
      -0.10489720851182938,
      0.3784435987472534,
      0.28034985065460205,
      0.18243014812469482,
      0.27069905400276184
    ]
  ],
  "layers.0.bias": [
    -0.1768643707036972,
    -0.2732515335083008,
    -0.09850584715604782,
    0.06216082721948624,
    -0.21761953830718994,
    0.2347216159105301,
    -0.18201090395450592,
    -0.2974637448787689,
    -0.22906531393527985,
    -0.3772487938404083,
    -0.030543314293026924,
    0.22122901678085327,
    -0.21309828758239746,
    -0.051603302359580994,
    -0.06868170946836472,
    -0.31598329544067383
  ],
  "layers.2.weight": [
    [
      0.18083876371383667,
      -0.10387269407510757,
      0.0875229611992836,
      0.17474877834320068,
      -0.10957291722297668,
      -0.15400837361812592,
      0.28269296884536743,
      -0.10803216695785522,
      0.21201656758785248,
      0.18741562962532043,
      -0.3504253923892975,
      0.2016758918762207,
      0.24121588468551636,
      -0.3242780864238739,
      0.10979773849248886,
      0.24536584317684174
    ]
  ],
  "layers.2.bias": [
    -0.29125040769577026
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
    def __init__(self):
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

        if year >= 2002:
            input_arr = self._get_input_features(home_id, away_id, season, date)

            if input_arr is not None:
                self.past_pred.append([*input_arr, home_win])
                self.ensamble_retrain -= 1

        self._handle_metrics(idx, current)

        if year >= 2000:
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
        my_bet = max(min_bet, min(max_bet / 2, summary.iloc[0]['Bankroll'] * 0.02))

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
                    adj_pred = sigmoid((inverse_sigmoid(pred) + 0.1) * 1.05) if playoff else pred

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
