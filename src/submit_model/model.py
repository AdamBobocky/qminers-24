import warnings

warnings.filterwarnings('ignore')
import math
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
    def __init__(self, num_teams=30, monthly_decay=0.8):
        self.num_teams = num_teams
        self.monthly_decay = monthly_decay

        self.team_mus = nn.Parameter(torch.zeros(num_teams))
        self.team_sigmas = torch.ones(num_teams) * 40
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

    def forward(self, weights, idx_start):
        games = self.games[idx_start:self.game_count]
        home_ratings = self.team_mus[games[:, 1]]
        away_ratings = self.team_mus[games[:, 2]]

        expectations_home = self.home_advantage + home_ratings - away_ratings
        realities_home = games[:, 3] - games[:, 4]

        game_sigmas = torch.sqrt(
            self.team_sigmas[games[:, 1]] ** 2 +
            self.team_sigmas[games[:, 2]] ** 2 +
            self.sigma ** 2
        )

        log_value = (-0.5 * ((realities_home - expectations_home) / game_sigmas) ** 2 - torch.log(game_sigmas) - 0.5 * torch.log(torch.tensor(2 * np.pi)))
        return torch.sum(log_value * weights[idx_start:])

    def _fit(self, max_epochs=30):
        weights = self._get_time_weights()
        idx_start = torch.nonzero(weights > 0.004, as_tuple=True)[0][0].item()
        optimizer = optim.Adam([self.team_mus, self.sigma, self.home_advantage], lr=0.02)

        for epoch in range(max_epochs):
            optimizer.zero_grad()
            loss = -self.forward(weights, idx_start)
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

from collections import defaultdict
from datetime import datetime
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
  "home_field_advantage": 4.273768901824951,
  "player_model.layers.0.weight": [
    [
      -0.46132466197013855,
      0.14314395189285278,
      0.0969933569431305,
      -0.0025559880305081606,
      -0.0010695202508941293,
      0.07897372543811798,
      -0.10150393098592758,
      0.28524020314216614,
      -0.016492562368512154,
      0.024918444454669952,
      0.24745190143585205,
      0.233917236328125,
      -0.1279875487089157,
      -0.1544625163078308,
      0.01708388328552246,
      -0.11058883368968964,
      0.2825421690940857,
      -0.0008825539262033999,
      0.05452527850866318
    ],
    [
      0.4944504499435425,
      -0.11379224807024002,
      0.025597723200917244,
      0.041924849152565,
      0.2036632001399994,
      -0.0432777926325798,
      0.3509446084499359,
      0.2831759452819824,
      -0.0038330580573529005,
      -0.11064938455820084,
      0.19283133745193481,
      0.1942169964313507,
      0.3142022490501404,
      0.0003423029265832156,
      0.21947111189365387,
      0.08574233204126358,
      0.22310057282447815,
      0.20986813306808472,
      0.2077830284833908
    ],
    [
      0.6206061244010925,
      0.1583482325077057,
      0.19039912521839142,
      -0.06429808586835861,
      0.26799917221069336,
      0.13746267557144165,
      -0.05716133117675781,
      -0.02492421120405197,
      0.07645440101623535,
      -0.06437697261571884,
      -0.1035865843296051,
      0.07382678240537643,
      0.04680756852030754,
      0.10258776694536209,
      0.21785669028759003,
      0.10186921805143356,
      0.16716650128364563,
      0.4019826352596283,
      0.010406022891402245
    ],
    [
      0.4816531240940094,
      -0.1902429163455963,
      0.08985863626003265,
      -0.008190132677555084,
      -0.006650008261203766,
      -0.042139261960983276,
      0.2792526185512543,
      0.2350957989692688,
      0.27587974071502686,
      -0.006455853581428528,
      0.2830427885055542,
      -0.04345907270908356,
      0.21349626779556274,
      0.21137990057468414,
      0.22953200340270996,
      0.14638729393482208,
      -0.018971631303429604,
      0.056408870965242386,
      0.1303374171257019
    ],
    [
      0.562394380569458,
      0.05477524921298027,
      0.2810579240322113,
      -0.09159336239099503,
      0.11469974368810654,
      0.1419934630393982,
      0.06280618906021118,
      0.18257296085357666,
      0.24173583090305328,
      0.29858410358428955,
      0.11435690522193909,
      0.19627009332180023,
      0.22774310410022736,
      0.03296620398759842,
      0.14266787469387054,
      0.24175019562244415,
      0.1998787820339203,
      0.2705686092376709,
      0.04148826748132706
    ],
    [
      -0.4663291573524475,
      0.11719981580972672,
      -0.059291329234838486,
      -0.06335223466157913,
      0.09885196387767792,
      0.17596019804477692,
      -0.059525929391384125,
      0.19924296438694,
      -0.11656607687473297,
      0.050621695816516876,
      0.29718488454818726,
      0.10077078640460968,
      0.16147258877754211,
      -0.10296448320150375,
      -0.05239323526620865,
      -0.04170166328549385,
      0.33941400051116943,
      0.16701024770736694,
      0.14254648983478546
    ],
    [
      -0.5088132619857788,
      -0.007849408313632011,
      0.2202838808298111,
      0.05364477261900902,
      0.04453333839774132,
      0.04903711751103401,
      0.21197645366191864,
      0.13348683714866638,
      0.21344897150993347,
      -0.07651360332965851,
      0.19362638890743256,
      -0.06340659409761429,
      0.041101742535829544,
      -0.16029910743236542,
      0.13627266883850098,
      -0.2673618495464325,
      0.05097339674830437,
      0.15672975778579712,
      0.12006644904613495
    ],
    [
      0.6221994757652283,
      -0.011087113060057163,
      -0.09661169350147247,
      0.013888380490243435,
      0.1913718432188034,
      -0.1005922332406044,
      0.00041351091931574047,
      0.17329691350460052,
      0.13117720186710358,
      -0.01645677536725998,
      0.24505017697811127,
      0.08355771005153656,
      0.22951959073543549,
      0.20818331837654114,
      0.1341397762298584,
      0.14987987279891968,
      -0.06952380388975143,
      0.24108390510082245,
      -0.08747979253530502
    ],
    [
      0.21360236406326294,
      0.1424025297164917,
      -0.0768454298377037,
      0.2881695032119751,
      0.16803079843521118,
      0.23893749713897705,
      0.1471974104642868,
      0.2549327313899994,
      -0.06231151148676872,
      0.2043115645647049,
      0.17873327434062958,
      0.24315857887268066,
      -0.021837273612618446,
      0.24410025775432587,
      0.06188255548477173,
      0.4101611077785492,
      -0.04745335876941681,
      0.09783804416656494,
      0.25398775935173035
    ],
    [
      0.2874046266078949,
      -0.07816904038190842,
      -0.08565454185009003,
      0.032875657081604004,
      -0.14738905429840088,
      0.10044209659099579,
      -0.1094449982047081,
      -0.13255427777767181,
      0.24369889497756958,
      0.25558483600616455,
      0.1926112025976181,
      0.33505964279174805,
      0.2988196909427643,
      0.339863657951355,
      0.3778199851512909,
      0.3183732032775879,
      -0.10276823490858078,
      0.14543966948986053,
      0.20791393518447876
    ],
    [
      -0.053276799619197845,
      -0.03142334893345833,
      0.32004794478416443,
      0.061930857598781586,
      0.214658722281456,
      0.14547215402126312,
      0.2629673182964325,
      0.45046472549438477,
      -0.00567471981048584,
      0.06262896955013275,
      0.07311118394136429,
      0.09452550858259201,
      0.149872824549675,
      0.11077436804771423,
      0.20250314474105835,
      -0.1219499409198761,
      0.20569074153900146,
      0.2524121403694153,
      -0.047210659831762314
    ],
    [
      0.02704518474638462,
      0.13604943454265594,
      0.19328108429908752,
      0.2473263442516327,
      0.01610465906560421,
      0.23730351030826569,
      0.2536811828613281,
      0.3183937072753906,
      -0.03935166075825691,
      0.24618501961231232,
      0.20563346147537231,
      0.18990010023117065,
      -0.029130969196558,
      0.15960165858268738,
      0.30573806166648865,
      0.2722054719924927,
      0.2028971016407013,
      0.1359601765871048,
      -0.1254817098379135
    ]
  ],
  "player_model.layers.0.bias": [
    0.020220765843987465,
    0.08574437350034714,
    0.37870052456855774,
    0.19378480315208435,
    0.217021182179451,
    -0.07736591249704361,
    -0.08452315628528595,
    0.3670296370983124,
    0.3782435357570648,
    0.34008899331092834,
    -0.03203906491398811,
    0.22966966032981873
  ],
  "player_model.layers.2.weight": [
    [
      -0.09400201588869095,
      -0.23743018507957458,
      0.04104377701878548,
      0.021526729688048363,
      -0.18440204858779907,
      0.06104385107755661,
      -0.04273957759141922,
      0.21302583813667297,
      -0.22441019117832184,
      -0.08624333143234253,
      -0.15752318501472473,
      -0.18277955055236816
    ],
    [
      -0.6382842063903809,
      0.01264105923473835,
      0.4246429204940796,
      0.1608283668756485,
      0.4581577479839325,
      -0.31222838163375854,
      -0.4537745714187622,
      0.23259346187114716,
      0.0006477724527940154,
      0.13517703115940094,
      -0.15733909606933594,
      0.16902542114257812
    ],
    [
      -0.12811526656150818,
      0.007861371152102947,
      0.4715477228164673,
      0.3786522448062897,
      0.3188493251800537,
      -0.24935011565685272,
      -0.20153629779815674,
      0.5237644910812378,
      -0.1138695701956749,
      0.08966069668531418,
      0.08773737400770187,
      -0.0645654946565628
    ],
    [
      -0.24401430785655975,
      0.2806129455566406,
      0.49629753828048706,
      0.2085573822259903,
      0.06374436616897583,
      -0.5164902806282043,
      -0.5466634631156921,
      0.2901211380958557,
      0.2779448926448822,
      0.1193939819931984,
      -0.06741482019424438,
      -0.2606578469276428
    ],
    [
      -0.14686743915081024,
      0.09295734763145447,
      0.48585909605026245,
      0.25841572880744934,
      0.09449049830436707,
      -0.2947482466697693,
      0.03772997483611107,
      0.16724638640880585,
      0.07832644134759903,
      -0.0724082887172699,
      -0.12183545529842377,
      -0.001297096605412662
    ],
    [
      -0.35918542742729187,
      0.3162637948989868,
      0.2561104893684387,
      0.3448092043399811,
      0.37380170822143555,
      -0.2927223742008209,
      -0.11531826108694077,
      0.4447111487388611,
      0.29259490966796875,
      0.12411238998174667,
      0.2367764562368393,
      0.3655611276626587
    ]
  ],
  "player_model.layers.2.bias": [
    -0.05319960042834282,
    0.42732593417167664,
    0.18045386672019958,
    0.0530986487865448,
    0.5027801990509033,
    0.11539766192436218
  ],
  "player_model.layers.4.weight": [
    [
      0.17099247872829437,
      0.5611874461174011,
      0.6757961511611938,
      0.5115221738815308,
      0.5384373068809509,
      0.29965898394584656
    ],
    [
      0.39582327008247375,
      0.08285092562437057,
      0.4918971359729767,
      0.264239102602005,
      0.37985363602638245,
      0.38056886196136475
    ],
    [
      0.38702628016471863,
      0.2648528516292572,
      0.027833139523863792,
      0.11079459637403488,
      -0.31591472029685974,
      -0.19477267563343048
    ],
    [
      -0.16427898406982422,
      0.4259629249572754,
      0.5133938789367676,
      0.42201218008995056,
      0.01886879839003086,
      -0.026161113753914833
    ]
  ],
  "player_model.layers.4.bias": [
    0.04609839245676994,
    -0.2042640745639801,
    -0.3023581802845001,
    -0.15613144636154175
  ],
  "layers.0.weight": [
    [
      -0.29342252016067505,
      0.11669932305812836,
      0.22577492892742157,
      -0.08625336736440659,
      0.31954360008239746,
      -0.015772098675370216,
      -0.16215157508850098,
      -0.14668121933937073
    ],
    [
      0.01932557485997677,
      0.08739083260297775,
      0.3332729637622833,
      -0.004385114647448063,
      0.44301849603652954,
      0.2013770341873169,
      -0.06352437287569046,
      -0.19700783491134644
    ],
    [
      -0.18141847848892212,
      -0.12128943204879761,
      -0.012811878696084023,
      -0.028456008061766624,
      -0.22739700973033905,
      0.27241143584251404,
      0.018632296472787857,
      -0.04895592853426933
    ],
    [
      -0.19490580260753632,
      -0.17100632190704346,
      -0.23293550312519073,
      -0.029279645532369614,
      -0.19281379878520966,
      0.2757982611656189,
      0.18753857910633087,
      0.07305910438299179
    ],
    [
      0.42399585247039795,
      0.3829096555709839,
      0.000750741979572922,
      0.07093271613121033,
      -0.1668015867471695,
      -0.15665467083454132,
      0.23136375844478607,
      -0.17902375757694244
    ],
    [
      -0.3422153890132904,
      -0.1732456237077713,
      0.07320240139961243,
      -0.16463056206703186,
      -0.020777352154254913,
      0.01171827595680952,
      0.12251149863004684,
      0.1525115817785263
    ],
    [
      -0.024868778884410858,
      0.2563345432281494,
      -0.10508754104375839,
      0.11854599416255951,
      -0.04536003619432449,
      0.21793346107006073,
      -0.20813632011413574,
      -0.037182893604040146
    ],
    [
      -0.18316592276096344,
      -0.3031435012817383,
      0.3279156982898712,
      -0.17012785375118256,
      -0.31657901406288147,
      -0.029661748558282852,
      -0.2888486385345459,
      0.013483800925314426
    ],
    [
      -0.23388881981372833,
      0.011172330938279629,
      -0.07767385989427567,
      -0.006800463423132896,
      -0.16340236365795135,
      0.12625697255134583,
      0.3438524305820465,
      0.15357425808906555
    ],
    [
      0.22400586307048798,
      0.16165730357170105,
      -0.1066962406039238,
      -0.0639805868268013,
      -0.16932815313339233,
      0.4075947701931,
      -0.017314493656158447,
      0.26974576711654663
    ],
    [
      0.0778738409280777,
      0.33508679270744324,
      0.3082113265991211,
      0.23339179158210754,
      -0.19675348699092865,
      -0.09485837817192078,
      0.12423158437013626,
      0.23367218673229218
    ],
    [
      -0.3543935716152191,
      0.1256803572177887,
      -0.024976549670100212,
      -0.2166333943605423,
      0.1482192426919937,
      0.4182755649089813,
      -0.17849694192409515,
      -0.034679725766181946
    ],
    [
      0.3601365089416504,
      0.20222347974777222,
      -0.29262575507164,
      0.3905227482318878,
      -0.21489819884300232,
      -0.10842520743608475,
      0.18875829875469208,
      -0.024151736870408058
    ],
    [
      0.17474836111068726,
      0.13612239062786102,
      0.09648815542459488,
      0.2877526879310608,
      -0.19236145913600922,
      0.40313830971717834,
      -0.0843532383441925,
      -0.20529291033744812
    ],
    [
      -0.01916719786822796,
      -0.2969306409358978,
      -0.24596504867076874,
      0.20457150042057037,
      0.20799145102500916,
      -0.321767657995224,
      -0.22610485553741455,
      -0.062152888625860214
    ],
    [
      -0.22800473868846893,
      0.26013118028640747,
      -0.2285522222518921,
      -0.32030874490737915,
      -0.1819762885570526,
      -0.18787173926830292,
      0.1446540206670761,
      -0.15292105078697205
    ]
  ],
  "layers.0.bias": [
    -0.14129199087619781,
    0.3875221312046051,
    0.19046597182750702,
    0.10361554473638535,
    -0.2200271636247635,
    0.2294715791940689,
    0.030026011168956757,
    -0.2757928669452667,
    -0.21449638903141022,
    0.3923634886741638,
    0.06665293127298355,
    0.1054273471236229,
    0.16960082948207855,
    0.304372102022171,
    -0.043309055268764496,
    -0.13229042291641235
  ],
  "layers.2.weight": [
    [
      -0.42097008228302,
      -0.23848432302474976,
      -0.13252922892570496,
      -0.2202393114566803,
      0.2386821061372757,
      0.036666274070739746,
      0.13786998391151428,
      0.22291040420532227,
      0.20028844475746155,
      -0.2214047610759735,
      0.2789594829082489,
      -0.3781520128250122,
      0.3373568654060364,
      -0.2178739458322525,
      -0.1274261176586151,
      -0.16731327772140503
    ]
  ],
  "layers.2.bias": [
    0.029644742608070374
  ]
}

class PlayerRatingModel(nn.Module):
    def __init__(self):
        super(PlayerRatingModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(19, 12),
            nn.ReLU(),
            nn.Linear(12, 6),
            nn.ReLU(),
            nn.Linear(6, 4),
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

        self.home_inputs = np.empty((30000, 12, 40, self.INPUTS_DIM + 1), np.float32)
        self.away_inputs = np.empty((30000, 12, 40, self.INPUTS_DIM + 1), np.float32)
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
                        c_player_data.append([0] * (self.INPUTS_DIM + 1))

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
                        c_player_data.append([0] * (self.INPUTS_DIM + 1))

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
                self.player_data[data['pid']].append([[date, data['mins']], *data['inputs']])
                self.player_data[data['pid']] = self.player_data[data['pid']][-40:]

    def get_input_data(self, home_id, away_id, season, date):
        game_frame = self._get_game_frame(season, date, home_id, away_id)

        if game_frame is None:
            return None

        # if self.retrain_countdown <= 0:
        #     self.retrain_countdown = 2500


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


        #     num_epochs = 40 if self.first_training else 10
        #     self.first_training = False
        #     for epoch in range(num_epochs):
        #         train_loss, train_accuracy = self._train(train_loader)

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
        my_bet = max(min_bet, min(max_bet * 0.3, bankroll * 0.02))

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

                    min_home_odds = (1 / adj_pred + 0.025) if bankroll > 4000 else ((1 / adj_pred - 1) * 1.1 + 1.04)
                    min_away_odds = (1 / (1 - adj_pred) + 0.025) if bankroll > 4000 else ((1 / (1 - adj_pred) - 1) * 1.1 + 1.04)

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
