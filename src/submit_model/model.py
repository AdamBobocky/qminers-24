import math
import torch
import copy
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

pretrained_weights = {
  "home_field_advantage": 4.351552486419678,
  "player_model.layers.0.weight": [
    [
      0.5463494658470154,
      0.21004889905452728,
      -0.10237139463424683,
      0.2855411171913147,
      0.21399399638175964,
      -0.09893327951431274,
      0.11855059862136841,
      0.23714177310466766,
      -0.04737051948904991,
      0.1088726818561554,
      0.09510402381420135,
      0.3404611647129059,
      0.14236553013324738,
      0.05329460650682449,
      0.39961275458335876,
      0.39027971029281616,
      0.16180992126464844,
      0.17904847860336304,
      0.26020675897598267
    ],
    [
      -0.24722597002983093,
      0.03559093549847603,
      0.14142729341983795,
      0.04110291227698326,
      -0.040509629994630814,
      0.08156641572713852,
      0.17939229309558868,
      -0.0688062533736229,
      -0.22093962132930756,
      -0.12551195919513702,
      0.04998317360877991,
      0.22626037895679474,
      -0.1488441377878189,
      0.08036384731531143,
      0.19162431359291077,
      0.011051448993384838,
      0.2327592819929123,
      -0.09107749909162521,
      -0.033524040132761
    ],
    [
      0.571209192276001,
      0.24322658777236938,
      0.027390720322728157,
      0.20799373090267181,
      0.26624181866645813,
      -0.053353242576122284,
      0.12470794469118118,
      0.3742470145225525,
      0.17594671249389648,
      0.31941699981689453,
      -0.12094316631555557,
      0.36269617080688477,
      -0.06397394835948944,
      -0.0036695401649922132,
      0.3200127184391022,
      0.10882008820772171,
      0.11640860140323639,
      0.11151029169559479,
      0.34349435567855835
    ],
    [
      -0.17881490290164948,
      0.13807040452957153,
      0.4607223868370056,
      0.2381916493177414,
      0.14648942649364471,
      0.033397261053323746,
      0.17777547240257263,
      0.13334542512893677,
      0.1506119668483734,
      0.017560461536049843,
      0.29294300079345703,
      0.012693159282207489,
      0.0652390792965889,
      0.10378185659646988,
      -0.10726186633110046,
      0.032076723873615265,
      -0.02927631326019764,
      0.07239599525928497,
      0.11426369100809097
    ],
    [
      0.5929805636405945,
      0.1333369016647339,
      -0.16962668299674988,
      -0.022849474102258682,
      0.0119417579844594,
      -0.08060988783836365,
      0.007400818169116974,
      0.18699416518211365,
      0.1911144256591797,
      0.03294024616479874,
      0.03412362560629845,
      0.20610831677913666,
      0.19282811880111694,
      0.10406938940286636,
      0.1530192643404007,
      0.3323419690132141,
      -0.1759275496006012,
      -0.14100120961666107,
      0.2778986394405365
    ],
    [
      0.44289660453796387,
      -0.13290010392665863,
      -0.25505319237709045,
      -0.04282217100262642,
      0.15592128038406372,
      0.11878108233213425,
      0.039040833711624146,
      -0.06800303608179092,
      0.18063734471797943,
      0.08662731200456619,
      -0.05726180970668793,
      0.18385684490203857,
      0.19918155670166016,
      0.38266870379447937,
      0.0449364110827446,
      0.3275270164012909,
      0.015874285250902176,
      0.15435299277305603,
      -0.051097821444272995
    ],
    [
      0.529979407787323,
      0.02675325982272625,
      -0.07922863215208054,
      0.10434908419847488,
      -0.04308485612273216,
      -0.11486360430717468,
      -0.20934714376926422,
      0.016749855130910873,
      -0.09729580581188202,
      0.06774038076400757,
      0.14307160675525665,
      -0.020244965329766273,
      -0.1364951878786087,
      0.0068305185995996,
      -0.1419059783220291,
      0.06665480136871338,
      -0.11954109370708466,
      0.008114682510495186,
      0.05574177950620651
    ],
    [
      0.6145674586296082,
      0.22861841320991516,
      -0.06279861927032471,
      -0.04076257720589638,
      0.07440590858459473,
      0.0561133548617363,
      0.21807843446731567,
      0.17361991107463837,
      0.21781529486179352,
      0.01546739973127842,
      0.07331852614879608,
      0.16577769815921783,
      0.05043775588274002,
      0.37409430742263794,
      0.31709176301956177,
      0.1261283904314041,
      -0.17199142277240753,
      0.10827784240245819,
      0.03435231372714043
    ],
    [
      -0.44167935848236084,
      -0.019948862493038177,
      0.18964093923568726,
      0.19274723529815674,
      0.0027712322771549225,
      0.3368970453739166,
      0.08989018946886063,
      0.3129647970199585,
      0.04611406847834587,
      0.15576179325580597,
      0.04667501524090767,
      -0.08679212629795074,
      0.07690360397100449,
      0.0653771460056305,
      -0.14200282096862793,
      -0.1860666424036026,
      -0.03206484764814377,
      0.10248898714780807,
      0.08448757231235504
    ],
    [
      -0.1491106152534485,
      -0.08860817551612854,
      0.11248112469911575,
      -0.003293086541816592,
      0.1178307980298996,
      0.2670692503452301,
      0.3445165455341339,
      0.38361337780952454,
      0.31435009837150574,
      0.29510003328323364,
      0.0634971484541893,
      0.28206148743629456,
      -0.0659269168972969,
      0.024887826293706894,
      0.0785084143280983,
      0.21440047025680542,
      -0.040670495480298996,
      0.30230236053466797,
      0.2063230723142624
    ],
    [
      0.6421383023262024,
      0.004690088797360659,
      0.05051448941230774,
      -0.1115700751543045,
      0.1505662202835083,
      0.2895951271057129,
      -0.09295075386762619,
      0.022503076121211052,
      0.2917269766330719,
      -0.06155369430780411,
      0.24383112788200378,
      0.32450249791145325,
      0.0007067801780067384,
      0.25354716181755066,
      0.11934934556484222,
      0.3255768418312073,
      -0.04751992225646973,
      0.06455941498279572,
      0.16501465439796448
    ],
    [
      -0.03645185008645058,
      -0.06130416318774223,
      0.12927567958831787,
      -0.18312378227710724,
      0.08203993737697601,
      -0.20691198110580444,
      0.06466080248355865,
      -0.10952256619930267,
      0.053615640848875046,
      -0.11270762979984283,
      0.07494562119245529,
      0.025140482932329178,
      -0.13046550750732422,
      -0.13481110334396362,
      -0.09426575154066086,
      -0.09962951391935349,
      -0.11335629969835281,
      -0.19769488275051117,
      -0.04495587572455406
    ]
  ],
  "player_model.layers.0.bias": [
    -0.1483214795589447,
    0.2574416399002075,
    -0.008395020850002766,
    -0.018291011452674866,
    -0.1434815376996994,
    -0.16267059743404388,
    -0.15961778163909912,
    -0.12461301684379578,
    0.14012253284454346,
    -0.03565593063831329,
    -0.06059408187866211,
    -0.211972177028656
  ],
  "player_model.layers.2.weight": [
    [
      0.19240276515483856,
      -0.025136401876807213,
      0.34333938360214233,
      -0.026110732927918434,
      0.3199186623096466,
      0.49742740392684937,
      0.32467931509017944,
      0.638796329498291,
      0.13379661738872528,
      -0.15903858840465546,
      0.3971422612667084,
      0.1289799064397812
    ],
    [
      0.21443447470664978,
      -0.09103972464799881,
      0.2326427549123764,
      0.42448660731315613,
      0.15624040365219116,
      0.4010423421859741,
      0.04637167230248451,
      0.24649623036384583,
      0.060435205698013306,
      0.4814510643482208,
      0.31520089507102966,
      0.26493436098098755
    ],
    [
      0.04284277558326721,
      -0.020981652662158012,
      0.06242101639509201,
      0.03320515155792236,
      -0.16223257780075073,
      0.01572882942855358,
      -0.2267921417951584,
      -0.10215257853269577,
      0.19825924932956696,
      0.25511401891708374,
      -0.15620604157447815,
      0.2748304307460785
    ],
    [
      0.03696512058377266,
      0.29018068313598633,
      -0.07813890278339386,
      0.3727410137653351,
      -0.3157181739807129,
      -0.46151506900787354,
      -0.32762855291366577,
      -0.3876906633377075,
      0.4061606824398041,
      0.02424047701060772,
      -0.14135663211345673,
      -0.08430524170398712
    ],
    [
      -0.13323785364627838,
      0.44681254029273987,
      0.16897213459014893,
      0.35187822580337524,
      -0.187385693192482,
      0.06364821642637253,
      -0.42237597703933716,
      -0.2919965982437134,
      0.27005958557128906,
      0.0983150377869606,
      -0.15930913388729095,
      0.025970401242375374
    ],
    [
      -0.0019191751489415765,
      0.2187628298997879,
      -0.11908603459596634,
      -0.25779256224632263,
      -0.14697274565696716,
      -0.0212278813123703,
      0.07927173376083374,
      -0.25372233986854553,
      0.021056076511740685,
      0.26595941185951233,
      -0.21618172526359558,
      -0.18214638531208038
    ]
  ],
  "player_model.layers.2.bias": [
    0.01920280046761036,
    0.38837990164756775,
    0.4766004979610443,
    0.23973645269870758,
    0.3465231955051422,
    -0.2577499747276306
  ],
  "player_model.layers.4.weight": [
    [
      0.3385750353336334,
      0.43861618638038635,
      0.23192252218723297,
      0.3530777096748352,
      0.31279534101486206,
      -0.15168780088424683
    ],
    [
      0.32238632440567017,
      0.47566819190979004,
      0.002808842109516263,
      -0.5419803261756897,
      -0.10122941434383392,
      0.24362576007843018
    ],
    [
      -0.4027124047279358,
      0.11335182934999466,
      0.5686997771263123,
      0.6702728271484375,
      0.5909665822982788,
      -0.3520267903804779
    ],
    [
      -0.018561076372861862,
      0.3551883101463318,
      0.43304046988487244,
      -0.3489759564399719,
      -0.3037883937358856,
      -0.2162131369113922
    ]
  ],
  "player_model.layers.4.bias": [
    0.0152351139113307,
    0.1938028335571289,
    0.13603712618350983,
    0.3820042908191681
  ],
  "layers.0.weight": [
    [
      0.21884842216968536,
      0.36858049035072327,
      -0.37895023822784424,
      -0.14499764144420624,
      -0.08009707927703857,
      -0.09498423337936401,
      0.2045792043209076,
      -0.005986189004033804
    ],
    [
      0.08946197479963303,
      -0.3523797392845154,
      0.41908758878707886,
      0.20442171394824982,
      0.21654130518436432,
      0.24654607474803925,
      -0.0009667149279266596,
      0.13019676506519318
    ],
    [
      -0.30060985684394836,
      0.22084566950798035,
      -0.15766240656375885,
      0.324943482875824,
      0.03375446796417236,
      0.19796152412891388,
      0.4296437203884125,
      -0.09658877551555634
    ],
    [
      -0.2635239064693451,
      0.09838994592428207,
      0.012463473714888096,
      0.16583910584449768,
      0.04854041710495949,
      0.1074543371796608,
      0.009522785432636738,
      -0.1328616440296173
    ],
    [
      -0.054993920028209686,
      0.3089374899864197,
      -0.27654170989990234,
      0.11680229753255844,
      -0.05439239740371704,
      -0.0188441164791584,
      0.4595397412776947,
      0.15015192329883575
    ],
    [
      0.264489084482193,
      0.17092198133468628,
      0.1988161951303482,
      -0.15460357069969177,
      0.2312801629304886,
      0.4597480595111847,
      -0.4190692603588104,
      0.18682314455509186
    ],
    [
      -0.10446074604988098,
      -0.29918643832206726,
      0.040834005922079086,
      -0.3202129006385803,
      0.08215594291687012,
      -0.31386128067970276,
      0.18133890628814697,
      -0.10663749277591705
    ],
    [
      0.31497493386268616,
      0.05999075248837471,
      -0.4861506223678589,
      0.039358314126729965,
      0.02172842249274254,
      -0.13078573346138,
      0.3401065766811371,
      -0.19596435129642487
    ],
    [
      0.005781617481261492,
      0.17916378378868103,
      -0.3850318491458893,
      0.15734460949897766,
      -0.2549991309642792,
      0.28631365299224854,
      0.36155083775520325,
      0.25218626856803894
    ],
    [
      0.05476882681250572,
      -0.0362812839448452,
      0.2608928680419922,
      -0.23433351516723633,
      -0.10147326439619064,
      -0.025612909346818924,
      0.30743926763534546,
      -0.02345704287290573
    ],
    [
      -0.2218686044216156,
      0.1540674865245819,
      -0.29701924324035645,
      -0.20538237690925598,
      -0.10733215510845184,
      -0.29134365916252136,
      0.18715186417102814,
      -0.20565733313560486
    ],
    [
      0.27916157245635986,
      0.23077067732810974,
      -0.20131738483905792,
      0.000576850725337863,
      0.0682213231921196,
      -0.1695527732372284,
      -0.037427082657814026,
      -0.395173043012619
    ],
    [
      -0.04200642928481102,
      -0.1878829002380371,
      -0.2162759155035019,
      -0.22245562076568604,
      -0.15819035470485687,
      -0.2534254789352417,
      -0.20659835636615753,
      -0.18053807318210602
    ],
    [
      -0.09914442896842957,
      0.010263249278068542,
      -0.04167231544852257,
      -0.3630301058292389,
      0.07406389713287354,
      -0.07788192480802536,
      -0.13224312663078308,
      -0.12920142710208893
    ],
    [
      -0.19502395391464233,
      0.15790300071239471,
      -0.34209901094436646,
      0.19241414964199066,
      -0.05759342014789581,
      0.022853409871459007,
      0.03672216460108757,
      -0.0182417631149292
    ],
    [
      0.13157187402248383,
      -0.15117473900318146,
      0.021456973627209663,
      0.30035296082496643,
      -0.27928000688552856,
      -0.3175276815891266,
      -0.0598943792283535,
      -0.04182082787156105
    ]
  ],
  "layers.0.bias": [
    -0.36235642433166504,
    -0.2759006917476654,
    -0.04664972051978111,
    -0.05686993896961212,
    -0.2521326243877411,
    0.30783918499946594,
    0.09645074605941772,
    -0.18441073596477509,
    -0.08289752900600433,
    -0.020700976252555847,
    -0.014140041545033455,
    -0.3802127540111542,
    -0.30573293566703796,
    0.2318580001592636,
    -0.10772298276424408,
    -0.32765236496925354
  ],
  "layers.2.weight": [
    [
      0.2648736238479614,
      -0.23575863242149353,
      0.3365488350391388,
      0.06694819033145905,
      0.3266679644584656,
      -0.32241734862327576,
      0.09720227122306824,
      0.21722397208213806,
      0.18604837357997894,
      0.0660245418548584,
      0.09172950685024261,
      0.08611509203910828,
      -0.11571922898292542,
      0.05227218568325043,
      -0.13675066828727722,
      0.025149136781692505
    ]
  ],
  "layers.2.bias": [
    -0.19171854853630066
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
                        c_player_data = copy.deepcopy(self.player_data[pid][-40:])

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
                        c_player_data = copy.deepcopy(self.player_data[pid][-40:])

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
        season = current['Season']
        home_id = current['HID']
        away_id = current['AID']
        home_score = current['HSC']
        away_score = current['ASC']
        date = current['Date']

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

    def get_input_data(self, home_id, away_id, season, date):
        game_frame = self._get_game_frame(season, date, home_id, away_id)

        if game_frame is None:
            return None

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

        return [
            prediction.item()
        ]

import math
from collections import defaultdict

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
import math
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from collections import defaultdict

def inverse_sigmoid(x):
    return math.log(x / (1 - x))

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def x1(delta, game_sigma):
    return 1 / (game_sigma * 2.50662828) * np.exp(-0.5 * (delta / game_sigma) ** 2)

def x2(delta, game_sigma):
    return delta / game_sigma ** 2

def x3(delta, game_sigma):
    return -delta / game_sigma ** 2

def x4(delta, game_sigma):
    return -(game_sigma ** 2 - delta ** 2) / game_sigma ** 3

class GradientDescent:
    def __init__(self, num_teams=30, learning_rate=0.03, monthly_decay=0.75, season_reset_mult=0.7):
        self.learning_rate = learning_rate
        self.monthly_decay = monthly_decay
        self.season_reset_mult = season_reset_mult
        self.games = np.empty((0, 5), int)
        self.home_advantage = 5
        self.sigma = 12
        self.team_mus = np.zeros(num_teams)
        self.team_sigmas = np.ones(num_teams) * 42
        self.my_team_id = {}
        self.last_season = -1
        self.fit_date = None

    def _check_season(self, season):
        if self.last_season != season:
            self.last_season = season

            self.team_sigmas = np.ones_like(self.team_sigmas) * (42 * self.season_reset_mult)

    def _get_time_weights(self):
        last_ts = self.games[-1, 0]

        return self.monthly_decay ** (np.abs(self.games[:, 0] - last_ts) / 30 / 24 / 60 / 60 / 1000)

    def _gradients(self):
        weights = self._get_time_weights()

        home_ratings = self.team_mus[self.games[:, 1]]
        away_ratings = self.team_mus[self.games[:, 2]]

        expectations_home = self.home_advantage + home_ratings - away_ratings

        realities_home = self.games[:, 3] - self.games[:, 4]
        realities_home = np.sign(realities_home) * (np.abs(realities_home) + 3.0) ** 0.7
        realities_away = self.games[:, 4] - self.games[:, 3]
        realities_away = np.sign(realities_away) * (np.abs(realities_away) + 3.0) ** 0.7

        game_sigmas = np.sqrt(self.team_sigmas[self.games[:, 1]] ** 2 + self.team_sigmas[self.games[:, 2]] ** 2 + self.sigma ** 2)

        mus_adjustments_home = x2(realities_home + away_ratings - self.home_advantage - home_ratings, game_sigmas) * weights
        mus_adjustments_away = x2(realities_away + home_ratings + self.home_advantage - away_ratings, game_sigmas) * weights
        grad_sigma = x4(realities_home - expectations_home, game_sigmas) * weights
        grad_home_advantage = x3(realities_home - expectations_home, game_sigmas) * weights

        grad_team_mus = np.zeros_like(self.team_mus)

        np.add.at(grad_team_mus, self.games[:, 1], mus_adjustments_home)
        np.add.at(grad_team_mus, self.games[:, 2], mus_adjustments_away)

        return grad_team_mus, np.mean(grad_sigma), np.mean(grad_home_advantage)

    def _calculate_objective(self):
        weights = self._get_time_weights()

        home_ratings = self.team_mus[self.games[:, 1]]
        away_ratings = self.team_mus[self.games[:, 2]]

        expectations_home = self.home_advantage + home_ratings - away_ratings

        realities_home = self.games[:, 3] - self.games[:, 4]
        realities_away = self.games[:, 4] - self.games[:, 3]

        game_sigmas = np.sqrt(self.team_sigmas[self.games[:, 1]] ** 2 + self.team_sigmas[self.games[:, 2]] ** 2 + self.sigma ** 2)

        home_objective = np.sum(x1(realities_home + away_ratings - self.home_advantage - home_ratings, game_sigmas) * weights)
        away_objective = np.sum(x1(realities_away + home_ratings + self.home_advantage - away_ratings, game_sigmas) * weights)
        sigma_objective = np.sum(x1(realities_home - expectations_home, game_sigmas) * weights) * 0.1

        return home_objective + away_objective + sigma_objective

    def _map_team_id(self, team_id):
        if team_id not in self.my_team_id:
            self.my_team_id[team_id] = len(self.my_team_id)

        return self.my_team_id[team_id]

    def _fit(self):
        games_count = len(self.games)
        best_objective = self._calculate_objective() / games_count
        best_state = [self.team_mus, self.sigma, self.home_advantage]
        countdown = 30
        while countdown > 0:
            countdown -= 1

            grad_team_mus, grad_sigma, grad_home_advantage = self._gradients()

            self.team_mus += self.learning_rate * grad_team_mus
            self.sigma += self.learning_rate * grad_sigma
            self.home_advantage -= self.learning_rate * grad_home_advantage

            new_objective = self._calculate_objective() / games_count

            if new_objective > best_objective + 0.00001:
                best_objective = new_objective
                best_state = [self.team_mus, self.sigma, self.home_advantage]
                countdown = 30

        self.team_mus, self.sigma, self.home_advantage = best_state

    def pre_add_game(self, current, current_players):
        pass

    def add_game(self, current, current_players):
        self._check_season(current['Season'])

        timestamp = int(current['Date'].timestamp() * 1000)
        team_home = self._map_team_id(current['HID'])
        team_away = self._map_team_id(current['AID'])
        score_home = current['HSC']
        score_away = current['ASC']

        self.games = np.vstack([self.games, np.array([timestamp, team_home, team_away, score_home, score_away])])
        self.games = self.games[-6000:]

        # Reduce sigma
        self.team_sigmas[team_home] = 1 / math.sqrt(1 / self.team_sigmas[team_home] ** 2 + 1 / self.sigma ** 2)
        self.team_sigmas[team_away] = 1 / math.sqrt(1 / self.team_sigmas[team_away] ** 2 + 1 / self.sigma ** 2)

        self.fit_date = None

    def get_input_data(self, home_id, away_id, season, date):
        self._check_season(season)

        if self.fit_date is None or self.fit_date != date:
            self.fit_date = date

            self._fit()

        team_home = self._map_team_id(home_id)
        team_away = self._map_team_id(away_id)

        game_exp = self.home_advantage + self.team_mus[team_home] - self.team_mus[team_away]
        game_sigma = math.sqrt(self.team_sigmas[team_home] ** 2 + self.team_sigmas[team_away] ** 2 + self.sigma ** 2)

        return [
            game_exp / game_sigma
        ]
import math
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression

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

class Model:
    def __init__(self):
        # Hyperparameters
        self.ensamble_required_n = 3000
        nate_silver_elo = NateSilverElo()
        self.model_list = [
            Pythagorean(),                  # 0.022865
            FourFactor(),                   # 0.001608
            GradientDescent(),              # 0.031539
            Exhaustion(),                   # -0.000207
            nate_silver_elo,                # 0.762238
            NeuralNetwork(nate_silver_elo)  # -0.004663
        ]
        # End

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

        if year >= 1994:
            input_arr = self._get_input_features(home_id, away_id, season, date)

            if input_arr is not None:
                self.past_pred.append([*input_arr, home_win])
                self.ensamble_retrain -= 1

        self._handle_metrics(idx, current)

        if year >= 1986:
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
        my_bet = max(min_bet, min(max_bet * 0.3, summary.iloc[0]['Bankroll'] * 0.03))

        bets = pd.DataFrame(data=np.zeros((len(opps), 2)), columns=['BetH', 'BetA'], index=opps.index)

        for i in opps.index:
            current = opps.loc[i]

            season = current['Season']
            date = current['Date']
            home_id = current['HID']
            away_id = current['AID']

            if len(self.past_pred) >= self.ensamble_required_n:
                input_arr = self._get_input_features(home_id, away_id, season, date)

                if input_arr is not None:
                    if self.ensamble_retrain <= 0:
                        self.ensamble_retrain = 300
                        np_array = np.array(self.past_pred)
                        sample_weights = np.exp(-0.0003 * np.arange(len(self.past_pred)))
                        self.ensamble = LogisticRegression(max_iter=10000)
                        self.ensamble.fit(np_array[:, :-1], np_array[:, -1], sample_weight=sample_weights[::-1])

                    self.bet_metrics['opps'] += 1

                    pred = self.ensamble.predict_proba(np.array([input_arr]))[0, 1]

                    self.prediction_map[i] = pred
                    self.input_map[i] = input_arr
                    self.coef_map[i] = [self.ensamble.intercept_.tolist(), *self.ensamble.coef_.tolist()]

                    odds_home = current['OddsH']
                    odds_away = current['OddsA']

                    min_home_odds = (1 / pred - 1) * 1.0 + 1 + 0.02
                    min_away_odds = (1 / (1 - pred) - 1) * 1.0 + 1 + 0.02

                    if odds_home >= min_home_odds:
                        bets.at[i, 'BetH'] = my_bet

                        self.bet_metrics['exp_pnl'] += pred * odds_home - 1
                        self.bet_metrics['volume'] += my_bet
                        self.bet_metrics['count'] += 1
                        self.bet_metrics['sum_odds'] += odds_home

                    if odds_away >= min_away_odds:
                        bets.at[i, 'BetA'] = my_bet

                        self.bet_metrics['exp_pnl'] += (1 - pred) * odds_away - 1
                        self.bet_metrics['volume'] += my_bet
                        self.bet_metrics['count'] += 1
                        self.bet_metrics['sum_odds'] += odds_away

        return bets
