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
  "home_field_advantage": 4.165315628051758,
  "player_model.layers.0.weight": [
    [
      0.27483654022216797,
      -0.11663494259119034,
      0.14240792393684387,
      0.10027884691953659,
      0.10766737163066864,
      0.04202750697731972,
      0.07376166433095932,
      -0.042419545352458954,
      -0.08649297803640366,
      0.26349613070487976,
      0.041232191026210785,
      0.16727621853351593,
      0.296327143907547,
      0.3920525014400482,
      0.21112287044525146,
      0.27708902955055237,
      -0.035016220062971115,
      -0.11143374443054199,
      0.11955160647630692,
      0.17898039519786835
    ],
    [
      -0.3923317790031433,
      0.02233298122882843,
      0.2574632465839386,
      0.11358243227005005,
      -0.07562223076820374,
      0.2910689115524292,
      -0.0657278299331665,
      -0.12604492902755737,
      -0.09522169828414917,
      0.051745522767305374,
      0.11758214980363846,
      -0.15181085467338562,
      0.07409369200468063,
      -0.22885198891162872,
      -0.22050686180591583,
      -0.3501349091529846,
      0.08825986832380295,
      0.12792277336120605,
      -0.07076597213745117,
      -0.17390698194503784
    ],
    [
      -0.26161298155784607,
      0.05258774384856224,
      0.10392651706933975,
      0.16174069046974182,
      -0.05836845561861992,
      0.12159866839647293,
      -0.09663219004869461,
      0.13489903509616852,
      0.009256603196263313,
      0.1090124100446701,
      -0.10235778987407684,
      -0.21123337745666504,
      -0.0341445691883564,
      -0.10570918023586273,
      -0.14394885301589966,
      -0.24377138912677765,
      0.07730051130056381,
      0.061643075197935104,
      -0.05308728665113449,
      -0.1397508680820465
    ],
    [
      -0.3865657448768616,
      -0.10646329820156097,
      0.26902034878730774,
      -0.018886258825659752,
      -0.1379159837961197,
      0.313301146030426,
      0.15782377123832703,
      0.13926537334918976,
      0.1301860809326172,
      -0.019811324775218964,
      -0.11722590774297714,
      -0.10548106580972672,
      0.10185559093952179,
      -0.2849684953689575,
      -0.008150982670485973,
      -0.028536779806017876,
      0.20505844056606293,
      0.06319242715835571,
      -0.14368999004364014,
      0.15403813123703003
    ],
    [
      0.019205274060368538,
      -0.04697073996067047,
      0.01016977522522211,
      -0.15750201046466827,
      0.13687032461166382,
      -0.15986673533916473,
      -0.0499456413090229,
      -0.16903473436832428,
      -0.053070977330207825,
      -0.10312438756227493,
      0.05814743787050247,
      -0.05073496699333191,
      -0.02543194219470024,
      0.03007025271654129,
      0.14854025840759277,
      -0.21270082890987396,
      -0.19262301921844482,
      0.02007031999528408,
      0.1115419939160347,
      -0.18260550498962402
    ],
    [
      0.39532867074012756,
      0.22087566554546356,
      -0.21566371619701385,
      -0.050110824406147,
      0.3444591462612152,
      -0.01655905693769455,
      0.31504854559898376,
      0.16447293758392334,
      -0.00597230764105916,
      0.12088286131620407,
      -0.030970828607678413,
      0.12232412397861481,
      0.2597135007381439,
      0.13906329870224,
      0.1796596497297287,
      0.3392762541770935,
      -0.1333225965499878,
      0.29441240429878235,
      -0.0010092267766594887,
      0.32050955295562744
    ],
    [
      -0.2523639500141144,
      -0.17913426458835602,
      0.17516915500164032,
      0.23889297246932983,
      0.048213303089141846,
      -0.024019530043005943,
      -0.23150742053985596,
      0.04679229483008385,
      -0.11183415353298187,
      0.08894336223602295,
      -0.04428943246603012,
      -0.32621991634368896,
      0.12234974652528763,
      -0.3178141713142395,
      -0.06743152439594269,
      -0.27329832315444946,
      0.3335261642932892,
      0.17129497230052948,
      0.03110002540051937,
      -0.027323056012392044
    ],
    [
      -0.3235953152179718,
      -0.17015089094638824,
      0.15535496175289154,
      0.09226729720830917,
      0.03971102833747864,
      0.2017490416765213,
      0.07911093533039093,
      0.035113703459501266,
      0.1310320794582367,
      -0.21533329784870148,
      0.103084497153759,
      -0.13514548540115356,
      0.06705748289823532,
      -0.03991509601473808,
      -0.06954030692577362,
      -0.2518046498298645,
      0.05385119840502739,
      -0.01898231916129589,
      -0.09471455216407776,
      -0.1547333151102066
    ],
    [
      0.07143045961856842,
      -0.19366273283958435,
      -0.03327907249331474,
      0.19345618784427643,
      0.12651024758815765,
      -0.1411834955215454,
      0.04707539826631546,
      0.2594698369503021,
      -0.014088377356529236,
      0.14554402232170105,
      -0.07875525206327438,
      0.1082228496670723,
      0.11219648271799088,
      0.12490912526845932,
      0.3245250880718231,
      0.06039808318018913,
      0.1668875813484192,
      0.19445990025997162,
      0.11736306548118591,
      -0.036860231310129166
    ],
    [
      0.2469906210899353,
      -0.016935572028160095,
      -0.08356862515211105,
      0.2367224395275116,
      0.044693224132061005,
      0.23965391516685486,
      0.1536402404308319,
      0.33566033840179443,
      0.20119546353816986,
      0.21070139110088348,
      0.16848687827587128,
      0.08708243072032928,
      0.04505540058016777,
      0.3571426570415497,
      0.3976157605648041,
      0.2693118155002594,
      -0.08308805525302887,
      0.08768615126609802,
      0.3161165416240692,
      0.2728072702884674
    ],
    [
      0.1348535269498825,
      0.1747046411037445,
      -0.09330521523952484,
      0.05934368818998337,
      -0.06802845746278763,
      0.2161739468574524,
      0.19326935708522797,
      0.20836763083934784,
      0.25553449988365173,
      0.051726892590522766,
      0.10701126605272293,
      0.1725313514471054,
      0.0013324252795428038,
      0.15438221395015717,
      0.32080501317977905,
      0.44477716088294983,
      -0.20513227581977844,
      0.1283305585384369,
      0.09818774461746216,
      -0.020130859687924385
    ],
    [
      -0.27101385593414307,
      -0.06113317608833313,
      0.3131949007511139,
      0.060331448912620544,
      -0.09593584388494492,
      0.13569512963294983,
      0.12142514437437057,
      -0.07726804167032242,
      -0.017325760796666145,
      -0.036930978298187256,
      -0.052327800542116165,
      -0.20424552261829376,
      0.08385436236858368,
      -0.2792643904685974,
      -0.25252124667167664,
      -0.06196605786681175,
      0.08569230139255524,
      0.06397843360900879,
      0.017502354457974434,
      -0.10592666268348694
    ],
    [
      -0.5320523381233215,
      0.16291025280952454,
      0.517160177230835,
      0.015825552865862846,
      -0.09249740839004517,
      0.30085498094558716,
      -0.012434116564691067,
      0.24261507391929626,
      -0.15821243822574615,
      -0.035883303731679916,
      0.09262204170227051,
      0.08324047923088074,
      -0.10389310121536255,
      0.01450758520513773,
      -0.21571843326091766,
      -0.11012691259384155,
      0.36567917466163635,
      -0.05336492136120796,
      0.018144838511943817,
      -0.03363567590713501
    ],
    [
      -0.5280119776725769,
      -0.22962425649166107,
      0.3265664279460907,
      -0.008275237865746021,
      0.11404497176408768,
      0.029113123193383217,
      -0.058531951159238815,
      0.006207883358001709,
      0.03131035715341568,
      -0.003047836711630225,
      0.10951424390077591,
      0.09244205057621002,
      -0.1107049435377121,
      -0.11360927671194077,
      0.08171069622039795,
      0.021922556683421135,
      0.03353795036673546,
      0.16014353930950165,
      -0.09201155602931976,
      -0.02976619452238083
    ],
    [
      0.5579452514648438,
      0.02732166275382042,
      -0.17723824083805084,
      0.05786415934562683,
      0.14990384876728058,
      0.0414896234869957,
      -0.11502785980701447,
      0.19744136929512024,
      -0.02788306586444378,
      0.0917033776640892,
      0.07657420635223389,
      -0.02560177631676197,
      -0.021768758073449135,
      0.05378146842122078,
      0.20470775663852692,
      0.3189193606376648,
      -0.2568061053752899,
      -0.13269449770450592,
      0.05345243588089943,
      0.19907261431217194
    ],
    [
      0.17041808366775513,
      -0.2542596161365509,
      0.2505456507205963,
      -0.0787682756781578,
      0.16627290844917297,
      0.19995544850826263,
      0.30564403533935547,
      0.10321100801229477,
      -0.031788215041160583,
      -0.022108545526862144,
      -0.09596249461174011,
      0.21427462995052338,
      0.040483057498931885,
      0.050102636218070984,
      0.4896760880947113,
      0.21631722152233124,
      0.1672673523426056,
      -0.037433769553899765,
      0.2365647703409195,
      -0.0038152760826051235
    ]
  ],
  "player_model.layers.0.bias": [
    0.09144393354654312,
    -0.12414763122797012,
    -0.4180801808834076,
    -0.1800328940153122,
    -0.23473480343818665,
    0.5175901651382446,
    -0.1908530741930008,
    -0.2519385814666748,
    0.31473901867866516,
    0.14472757279872894,
    0.2586033046245575,
    -0.2814367711544037,
    -0.0764300599694252,
    0.17299072444438934,
    0.11340539902448654,
    0.4335203468799591
  ],
  "player_model.layers.2.weight": [
    [
      0.23462079465389252,
      0.029228975996375084,
      0.21307337284088135,
      -0.1228189691901207,
      0.13543586432933807,
      0.18603089451789856,
      0.30874955654144287,
      0.06782574206590652,
      0.22754642367362976,
      0.06637444347143173,
      0.10855647176504135,
      -0.2974257469177246,
      0.23902660608291626,
      -0.14839206635951996,
      -0.023193208500742912,
      0.11025428026914597
    ],
    [
      -0.022726519033312798,
      -0.17144978046417236,
      -0.24815437197685242,
      0.11156486719846725,
      -0.04940429702401161,
      0.3653971552848816,
      -0.24523518979549408,
      -0.2328503578901291,
      -0.09033306688070297,
      0.09368779510259628,
      0.3256649971008301,
      -0.12229049205780029,
      -0.18235711753368378,
      -0.09788567572832108,
      0.14397725462913513,
      0.11194838583469391
    ],
    [
      -0.022139089182019234,
      -0.31995537877082825,
      -0.3583025634288788,
      -0.3521984815597534,
      0.2456395924091339,
      0.30479833483695984,
      0.010316158644855022,
      -0.2630949020385742,
      0.14622551202774048,
      0.24240528047084808,
      0.10585170984268188,
      -0.5335973501205444,
      -0.4610822796821594,
      -0.41539445519447327,
      0.40601375699043274,
      0.26657548546791077
    ],
    [
      0.0901898741722107,
      0.09327628463506699,
      -0.0812516137957573,
      0.053125012665987015,
      0.20684920251369476,
      -0.0996464341878891,
      -0.03255850821733475,
      -0.1940290331840515,
      -0.023275073617696762,
      -0.24820385873317719,
      0.005242537707090378,
      -0.10932974517345428,
      0.0708639919757843,
      -0.23286719620227814,
      0.09937562048435211,
      0.02297169342637062
    ],
    [
      -0.032547175884246826,
      -0.1138816699385643,
      -0.02116870880126953,
      -0.21675142645835876,
      -0.23867233097553253,
      -0.15030315518379211,
      -0.21387508511543274,
      0.08042111992835999,
      0.1026369035243988,
      -0.05267069861292839,
      -0.020976562052965164,
      0.1489177793264389,
      -0.1428367793560028,
      -0.17059534788131714,
      -0.16884177923202515,
      -0.12083801627159119
    ],
    [
      0.28204116225242615,
      -0.33336159586906433,
      -0.39894986152648926,
      -0.4072822034358978,
      -0.18084262311458588,
      0.4229351282119751,
      -0.12953989207744598,
      -0.11883924156427383,
      0.2324020266532898,
      0.15657497942447662,
      -0.10114438831806183,
      -0.03718860074877739,
      0.04826894402503967,
      -0.3887978196144104,
      -0.08952491730451584,
      0.1693546175956726
    ],
    [
      0.26367467641830444,
      -0.2284558117389679,
      -0.29414448142051697,
      -0.03195197507739067,
      -0.18849198520183563,
      0.29420050978660583,
      -0.11364299058914185,
      -0.2628895342350006,
      0.17802278697490692,
      0.2121523767709732,
      0.2433895617723465,
      -0.4424102008342743,
      -0.2195013016462326,
      0.04435839131474495,
      -0.04805436730384827,
      0.14848336577415466
    ],
    [
      0.2902897000312805,
      -0.0607617013156414,
      -0.11640702188014984,
      0.010535872541368008,
      -0.1945512890815735,
      -0.11106765270233154,
      0.01475563645362854,
      0.09729335457086563,
      0.23581719398498535,
      0.27614539861679077,
      0.1600952297449112,
      -0.08293437957763672,
      -0.07894296944141388,
      0.07535793632268906,
      -0.11818225681781769,
      0.2591196894645691
    ],
    [
      -0.10163122415542603,
      0.47209981083869934,
      0.1367838978767395,
      0.38373705744743347,
      -0.1766863316297531,
      -0.07348237931728363,
      0.25133997201919556,
      0.007951400242745876,
      0.06281981617212296,
      -0.061771344393491745,
      0.054842568933963776,
      0.3214791715145111,
      0.42687609791755676,
      0.3129596412181854,
      -0.1573939025402069,
      0.11978723853826523
    ],
    [
      0.38089719414711,
      -0.3406607210636139,
      -0.19792334735393524,
      -0.41183367371559143,
      -0.07551927864551544,
      0.5059463381767273,
      -0.04439328610897064,
      -0.21325309574604034,
      0.11717896908521652,
      -0.004392738454043865,
      0.04687713086605072,
      -0.36426231265068054,
      -0.49124178290367126,
      -0.544108510017395,
      0.18443453311920166,
      0.11743009090423584
    ],
    [
      0.003256102791056037,
      0.02388499490916729,
      -0.027766862884163857,
      -0.06143830344080925,
      0.044350311160087585,
      -0.03106800466775894,
      0.033068232238292694,
      -0.17726141214370728,
      -0.1628926396369934,
      -0.18202710151672363,
      0.1347830891609192,
      0.031557053327560425,
      -0.1287650167942047,
      -0.027187630534172058,
      -0.11046192049980164,
      0.1873260736465454
    ],
    [
      -0.0077642700634896755,
      0.07403316348791122,
      -0.0733780488371849,
      0.09038887172937393,
      -0.15682165324687958,
      0.2743035554885864,
      -0.04486466944217682,
      -0.13361038267612457,
      0.22189585864543915,
      0.19524507224559784,
      0.1305183619260788,
      -0.29945048689842224,
      0.18782605230808258,
      0.012386183254420757,
      0.005023885052651167,
      0.17597445845603943
    ]
  ],
  "player_model.layers.2.bias": [
    0.0014843581011518836,
    0.5298693180084229,
    0.31879401206970215,
    0.11523132026195526,
    -0.07732415199279785,
    0.3262571394443512,
    0.10575331747531891,
    0.004500285722315311,
    -0.29612040519714355,
    0.3695133328437805,
    0.0382782444357872,
    0.4307365417480469
  ],
  "player_model.layers.4.weight": [
    [
      0.23491419851779938,
      0.24145357310771942,
      0.08721575140953064,
      -0.038715362548828125,
      -0.1509448140859604,
      0.1479366421699524,
      0.3442608416080475,
      0.2505197525024414,
      -0.2504662573337555,
      0.0007792213000357151,
      0.18215946853160858,
      0.13364464044570923
    ],
    [
      -0.08838813006877899,
      -0.21319706737995148,
      0.1672496497631073,
      0.1519836038351059,
      -0.10966846346855164,
      0.027446666732430458,
      -0.21280768513679504,
      -0.07936806231737137,
      -0.13203215599060059,
      0.14293812215328217,
      0.17513103783130646,
      -0.009325327351689339
    ],
    [
      0.017591923475265503,
      0.09578417986631393,
      -0.11741982400417328,
      0.19243280589580536,
      -0.2590963840484619,
      0.16243009269237518,
      -0.23049858212471008,
      -0.05455685406923294,
      0.050067514181137085,
      0.0851166844367981,
      0.10773833096027374,
      -0.34104636311531067
    ],
    [
      0.013829938136041164,
      0.4589093029499054,
      0.3980034589767456,
      0.17996644973754883,
      0.05827357992529869,
      0.23769085109233856,
      0.3495645523071289,
      -0.06327381730079651,
      -0.6100690364837646,
      0.5106929540634155,
      0.21619319915771484,
      0.05109670013189316
    ]
  ],
  "player_model.layers.4.bias": [
    0.302874892950058,
    -0.24117659032344818,
    0.1331576704978943,
    0.13448311388492584
  ],
  "layers.0.weight": [
    [
      -0.01647338643670082,
      0.21277152001857758,
      0.20660613477230072,
      -0.4271829426288605,
      0.2585500180721283,
      -0.01276028249412775,
      -0.0034119202755391598,
      0.4401811957359314
    ],
    [
      0.043419625610113144,
      0.23579704761505127,
      0.04172496125102043,
      -0.28964897990226746,
      0.02979009784758091,
      -0.13172392547130585,
      -0.22835062444210052,
      -0.3255830705165863
    ],
    [
      -0.03899431228637695,
      0.17921678721904755,
      0.07795165479183197,
      -0.2564895451068878,
      -0.03745942935347557,
      -0.21864859759807587,
      -0.12021438777446747,
      0.45334506034851074
    ],
    [
      -0.3015911877155304,
      -0.08029738813638687,
      -0.10717935860157013,
      0.3749137818813324,
      0.0945703536272049,
      -0.2326023429632187,
      0.31824877858161926,
      0.07913509011268616
    ],
    [
      -0.08670865744352341,
      -0.06294619292020798,
      0.0942768082022667,
      0.25854426622390747,
      0.09654583036899567,
      -0.11464706808328629,
      0.18147459626197815,
      -0.12731249630451202
    ],
    [
      0.23507489264011383,
      0.16188384592533112,
      -0.10779140889644623,
      0.015943320468068123,
      0.024942966178059578,
      -0.23015940189361572,
      -0.3849915564060211,
      0.10592325031757355
    ],
    [
      -0.042079079896211624,
      -0.1880163699388504,
      -0.2227848917245865,
      0.034099698066711426,
      -0.3283208906650543,
      0.32967352867126465,
      -0.3030155301094055,
      0.20001071691513062
    ],
    [
      0.1414898931980133,
      -0.2512173056602478,
      -0.05179664120078087,
      0.4197676181793213,
      0.20509351789951324,
      -0.17994587123394012,
      0.3388320207595825,
      -0.12726187705993652
    ],
    [
      0.2675618827342987,
      -0.13345827162265778,
      -0.20456403493881226,
      -0.2583967447280884,
      0.4620954096317291,
      -0.24092480540275574,
      -0.1925870180130005,
      0.03375738486647606
    ],
    [
      0.3282740116119385,
      -0.04914892464876175,
      0.12812991440296173,
      -0.004814974032342434,
      0.07797832787036896,
      0.25255733728408813,
      0.10759948194026947,
      -0.3629801571369171
    ],
    [
      0.17659056186676025,
      0.15071150660514832,
      0.07270646095275879,
      0.3352404534816742,
      0.020076578482985497,
      0.05799933895468712,
      -0.03829854726791382,
      -0.35343676805496216
    ],
    [
      -0.24639557301998138,
      -0.03816771134734154,
      0.3142617642879486,
      0.07929529994726181,
      -0.1925474852323532,
      0.3150944709777832,
      -0.17645899951457977,
      0.0037926211953163147
    ],
    [
      0.3683733642101288,
      0.07504305243492126,
      0.31636810302734375,
      -0.031941015273332596,
      -0.2975699007511139,
      -0.2827877402305603,
      -0.043233722448349,
      0.12318459898233414
    ],
    [
      -0.008739140816032887,
      0.10947830229997635,
      0.19282400608062744,
      -0.22382700443267822,
      -0.08746837079524994,
      0.026634706184267998,
      -0.2779005765914917,
      0.13391801714897156
    ],
    [
      0.3262586295604706,
      -0.14721496403217316,
      0.17534852027893066,
      0.26736775040626526,
      -0.21351759135723114,
      -0.11720752716064453,
      0.07503647357225418,
      -0.06420503556728363
    ],
    [
      0.19699148833751678,
      0.3064175546169281,
      -0.28907325863838196,
      -0.1460060477256775,
      -0.02603035978972912,
      0.13322024047374725,
      0.04204396903514862,
      0.47651833295822144
    ]
  ],
  "layers.0.bias": [
    0.28081417083740234,
    0.2500483989715576,
    0.5097716450691223,
    0.11968014389276505,
    0.07962538301944733,
    0.02522069774568081,
    0.13643327355384827,
    -0.4874662160873413,
    0.49100425839424133,
    -0.3829200267791748,
    0.19866210222244263,
    -0.04139505699276924,
    -0.1385745108127594,
    -0.007303498685359955,
    0.0027778425719588995,
    0.22869141399860382
  ],
  "layers.2.weight": [
    [
      -0.33102649450302124,
      0.06933033466339111,
      -0.3668561577796936,
      0.26433059573173523,
      0.14646494388580322,
      -0.26223820447921753,
      0.16986757516860962,
      0.1157240942120552,
      -0.35376304388046265,
      0.03447151929140091,
      0.20022684335708618,
      -0.06835812330245972,
      0.28566721081733704,
      0.14498506486415863,
      0.34896478056907654,
      -0.13079220056533813
    ]
  ],
  "layers.2.bias": [
    -0.2671699821949005
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

