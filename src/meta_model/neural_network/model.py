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
  "home_field_advantage": 4.275066375732422,
  "player_model.layers.0.weight": [
    [
      0.33716967701911926,
      -0.08335791528224945,
      -0.605610728263855,
      -0.07760924100875854,
      0.23251213133335114,
      0.1287582814693451,
      0.19165240228176117,
      -0.07121167331933975,
      0.15302135050296783,
      0.15762318670749664,
      0.06737647950649261,
      0.10561402887105942,
      0.3571943938732147,
      0.22780375182628632,
      0.1744825392961502,
      0.3802047371864319,
      -0.0026645020116120577,
      0.12478184700012207,
      0.23947149515151978,
      0.1872575283050537
    ],
    [
      -0.4321553409099579,
      -0.14374017715454102,
      0.31125715374946594,
      0.13580574095249176,
      0.1846451759338379,
      0.043880265206098557,
      -0.1017126739025116,
      0.040191859006881714,
      -0.15443557500839233,
      -0.06081175431609154,
      0.14917884767055511,
      -0.21805980801582336,
      0.07106807827949524,
      -0.28771600127220154,
      0.0870630145072937,
      -0.27984818816185,
      0.3280668258666992,
      -0.034671612083911896,
      0.04329955577850342,
      -0.09686330705881119
    ],
    [
      -0.1563631296157837,
      0.19638003408908844,
      0.5374997854232788,
      0.2098410576581955,
      -0.12055864185094833,
      0.07522831857204437,
      0.007869989611208439,
      0.23356586694717407,
      0.06744299829006195,
      0.2046714872121811,
      -0.018705591559410095,
      -0.004245663993060589,
      0.1260949671268463,
      -0.2046232521533966,
      -0.12074372172355652,
      -0.15549950301647186,
      0.2661765217781067,
      -0.10817979276180267,
      0.15849940478801727,
      0.14653484523296356
    ],
    [
      0.44798544049263,
      0.1650163233280182,
      -0.46112489700317383,
      0.22167955338954926,
      0.2416202425956726,
      0.1430758535861969,
      0.36933571100234985,
      0.3458596467971802,
      0.28898531198501587,
      0.2581123411655426,
      0.04013100266456604,
      0.20868797600269318,
      0.13273358345031738,
      0.4332369267940521,
      0.13781937956809998,
      0.2465444803237915,
      -0.0034340345300734043,
      0.18960945308208466,
      0.3075653612613678,
      0.36844322085380554
    ],
    [
      -0.30327481031417847,
      -0.20976252853870392,
      0.5429397821426392,
      0.21050386130809784,
      0.13561449944972992,
      -0.06173057109117508,
      0.15095971524715424,
      0.0008835455519147217,
      0.020203707739710808,
      -0.036481913179159164,
      -0.19057859480381012,
      0.1845466047525406,
      0.019874919205904007,
      0.009621075354516506,
      -0.19400058686733246,
      -0.23991075158119202,
      0.125342458486557,
      0.057649876922369,
      -0.0975998267531395,
      -0.09229779243469238
    ],
    [
      0.43792349100112915,
      0.11199288070201874,
      -0.1673564910888672,
      0.29321205615997314,
      0.3167569041252136,
      0.2511938214302063,
      0.19074928760528564,
      0.2642432749271393,
      0.42216941714286804,
      0.10109241306781769,
      -3.332440974190831e-05,
      0.15690988302230835,
      0.10298343747854233,
      0.36732974648475647,
      0.2927228510379791,
      0.22536803781986237,
      -0.1476352959871292,
      0.1257053017616272,
      0.1269182562828064,
      0.3403375446796417
    ],
    [
      0.4802103638648987,
      0.045178718864917755,
      -0.5166146159172058,
      -0.07737713307142258,
      0.21732467412948608,
      -0.009874378331005573,
      0.12799745798110962,
      -0.08483780175447464,
      0.07357116788625717,
      0.1804269254207611,
      0.061800092458724976,
      0.24841229617595673,
      0.07347967475652695,
      0.17086035013198853,
      0.22122527658939362,
      0.13112381100654602,
      0.07154550403356552,
      -0.08498431742191315,
      -0.1086508184671402,
      0.10349929332733154
    ],
    [
      -0.5031328201293945,
      -0.08055971562862396,
      0.4461577832698822,
      0.10406234860420227,
      -0.015966659411787987,
      0.1853797882795334,
      -0.2625473737716675,
      0.002743659308180213,
      -0.083912193775177,
      0.12504038214683533,
      0.13389478623867035,
      -0.09793229401111603,
      -0.04561302438378334,
      -0.2505592405796051,
      -0.1876792013645172,
      0.040785420686006546,
      0.2565971612930298,
      -0.050851576030254364,
      -0.0026426303666085005,
      0.09719252586364746
    ],
    [
      -0.34971725940704346,
      -0.05289565399289131,
      0.394408255815506,
      0.056137263774871826,
      -0.18405146896839142,
      0.22330231964588165,
      -0.003274791408330202,
      -0.16772763431072235,
      -0.04448433592915535,
      0.10779266059398651,
      0.06916803866624832,
      -0.23772567510604858,
      -0.020631959661841393,
      -0.17507779598236084,
      -0.24191230535507202,
      -0.4882057309150696,
      0.10827741771936417,
      0.11120304465293884,
      -0.054764505475759506,
      0.1638510376214981
    ],
    [
      -0.31037402153015137,
      -0.19498959183692932,
      0.49378153681755066,
      0.12322954833507538,
      0.05984479561448097,
      0.12866173684597015,
      -0.040452782064676285,
      0.014362459070980549,
      -0.08675078302621841,
      -0.04753026366233826,
      -0.16200216114521027,
      -0.18170684576034546,
      0.09125450253486633,
      -0.16304226219654083,
      -0.0028690113686025143,
      -0.30084753036499023,
      0.08188307285308838,
      0.10362451523542404,
      0.027202541008591652,
      -0.10817208886146545
    ],
    [
      0.4408489763736725,
      -0.054870929569005966,
      -0.2918090522289276,
      -0.12168312072753906,
      0.13197752833366394,
      0.21845406293869019,
      0.3251284956932068,
      0.014363757334649563,
      0.12790584564208984,
      0.22404660284519196,
      0.1854974925518036,
      0.17864036560058594,
      0.08733687549829483,
      0.43238455057144165,
      0.2103210836648941,
      0.28398382663726807,
      0.06484963744878769,
      0.1812545210123062,
      0.0823042243719101,
      0.2178899049758911
    ],
    [
      -0.3246605396270752,
      0.09077856689691544,
      0.4426238536834717,
      0.017816212028265,
      0.10268823802471161,
      0.11413723230361938,
      -0.03905889764428139,
      -0.07359962910413742,
      0.03310370445251465,
      0.008570934645831585,
      0.1785728484392166,
      -0.20325472950935364,
      -0.03853495791554451,
      -0.03512810543179512,
      -0.13297900557518005,
      -0.2141052931547165,
      0.038022156804800034,
      0.08570031821727753,
      -0.0957588404417038,
      0.19691145420074463
    ],
    [
      -0.11983684450387955,
      -0.16394676268100739,
      0.4539038836956024,
      0.19301076233386993,
      0.2716118097305298,
      0.030826477333903313,
      0.1760472059249878,
      0.4412926435470581,
      -0.017647769302129745,
      0.046891964972019196,
      0.1967693418264389,
      0.1249181479215622,
      0.14643149077892303,
      0.20017415285110474,
      0.028337085619568825,
      0.237754687666893,
      0.10773809254169464,
      0.07708846032619476,
      -0.07411788403987885,
      0.13077327609062195
    ],
    [
      0.5942040681838989,
      -0.018073441460728645,
      -0.31034642457962036,
      0.07071219384670258,
      0.07027594745159149,
      -0.06354647129774094,
      0.25163519382476807,
      0.025958720594644547,
      0.13755455613136292,
      0.20662368834018707,
      0.08591572940349579,
      0.06875640898942947,
      0.07100965082645416,
      0.18347591161727905,
      0.3690035343170166,
      0.16093549132347107,
      -0.1659395843744278,
      -0.17761367559432983,
      0.0151915792375803,
      0.28197425603866577
    ],
    [
      -0.42509976029396057,
      -0.13498294353485107,
      0.47714337706565857,
      0.18688583374023438,
      0.21633727848529816,
      0.02358892746269703,
      0.05335703864693642,
      0.1526256948709488,
      -0.04679447412490845,
      -0.08786964416503906,
      0.09404000639915466,
      0.14858442544937134,
      0.01876136101782322,
      -0.29118409752845764,
      -0.2150459587574005,
      -0.06718487292528152,
      0.27490168809890747,
      0.04611077904701233,
      0.021399149671196938,
      -0.07895532995462418
    ],
    [
      -0.4803536534309387,
      -0.025846445932984352,
      0.47042176127433777,
      -0.005742198321968317,
      -0.14827264845371246,
      0.24345369637012482,
      -0.11148562282323837,
      0.24683430790901184,
      0.10734832286834717,
      0.0762156993150711,
      -0.09573104977607727,
      -0.06961485743522644,
      0.07467178255319595,
      -0.1787157952785492,
      -0.11479397863149643,
      0.050605665892362595,
      0.21023273468017578,
      0.1285254806280136,
      -0.07195904850959778,
      0.0528254471719265
    ]
  ],
  "player_model.layers.0.bias": [
    -0.5430589318275452,
    0.2736566960811615,
    0.5228562951087952,
    -0.18951627612113953,
    0.24992945790290833,
    -0.27591386437416077,
    -0.09510115534067154,
    0.3524607717990875,
    0.2857624590396881,
    0.12325622886419296,
    -0.3459973931312561,
    0.37863972783088684,
    0.2208547592163086,
    -0.6237604022026062,
    0.5192723274230957,
    0.4525803029537201
  ],
  "player_model.layers.2.weight": [
    [
      -0.08546287566423416,
      0.11194553226232529,
      -0.10065071284770966,
      -0.09234214574098587,
      0.3242262601852417,
      -0.03401041775941849,
      -0.3812221586704254,
      0.18086005747318268,
      0.4045843780040741,
      0.3148905336856842,
      0.03608741611242294,
      0.2329384982585907,
      0.07130797952413559,
      -0.03194832429289818,
      -0.03140336275100708,
      0.3896021544933319
    ],
    [
      -0.16161365807056427,
      -0.07576719671487808,
      -0.21159875392913818,
      -0.1590946912765503,
      -0.18397435545921326,
      -0.14959290623664856,
      -0.09643447399139404,
      -0.2294420301914215,
      0.04330110177397728,
      0.18047964572906494,
      -0.17657575011253357,
      0.0389273464679718,
      0.02563384547829628,
      -0.08438234031200409,
      -0.004931362345814705,
      -0.18951213359832764
    ],
    [
      -0.024216758087277412,
      0.08944062143564224,
      0.08114015311002731,
      -0.03366118669509888,
      0.3774569034576416,
      0.04526684060692787,
      -0.5086910724639893,
      0.19124524295330048,
      0.49885985255241394,
      0.058290157467126846,
      -0.31096431612968445,
      0.43762099742889404,
      0.12997092306613922,
      -0.4545593559741974,
      0.0772978886961937,
      0.417728453874588
    ],
    [
      -0.10963820666074753,
      -0.1459769457578659,
      0.09604083746671677,
      -0.18648631870746613,
      -0.2597407400608063,
      -0.05561615899205208,
      0.07121582329273224,
      -0.042753804475069046,
      -0.03871463984251022,
      0.20906023681163788,
      -0.05121006444096565,
      0.03252919018268585,
      0.14022742211818695,
      -0.007667973171919584,
      -0.10573845356702805,
      -0.2201962023973465
    ],
    [
      0.1919001340866089,
      0.11300860345363617,
      -0.15471531450748444,
      0.1658826321363449,
      -0.030030526220798492,
      0.08775990456342697,
      0.29371580481529236,
      -0.1666172742843628,
      0.05318206921219826,
      -0.13380219042301178,
      0.36993810534477234,
      -0.26715412735939026,
      -0.006684792693704367,
      0.1379816085100174,
      -0.013254296034574509,
      0.00627771345898509
    ],
    [
      0.1408521831035614,
      -0.21955078840255737,
      -0.18707811832427979,
      0.021022332832217216,
      0.07648585736751556,
      0.2656473219394684,
      0.3065103590488434,
      0.14760559797286987,
      -0.2623944878578186,
      -0.057496532797813416,
      0.19580277800559998,
      -0.31009653210639954,
      0.15273050963878632,
      0.17941102385520935,
      0.19137157499790192,
      -0.11271077394485474
    ],
    [
      -0.29781171679496765,
      0.28255510330200195,
      -0.02722911164164543,
      0.2890074551105499,
      0.2686646282672882,
      0.2348177582025528,
      0.10653307288885117,
      0.10088290274143219,
      0.03787131607532501,
      0.11583121865987778,
      -0.11544644087553024,
      0.3018554151058197,
      0.22797268629074097,
      -0.29275578260421753,
      0.20672470331192017,
      0.09188414365053177
    ],
    [
      0.2696487009525299,
      -0.062502920627594,
      0.08225347846746445,
      0.36012569069862366,
      0.24360120296478271,
      0.24642786383628845,
      0.09067936986684799,
      -0.11327574402093887,
      -0.22426916658878326,
      0.04566711187362671,
      0.30697065591812134,
      -0.15584667026996613,
      0.2628919184207916,
      0.25689494609832764,
      0.004475443158298731,
      -0.019894352182745934
    ],
    [
      -0.1286524385213852,
      0.3756074607372284,
      0.3283448815345764,
      -0.14045901596546173,
      0.1716122180223465,
      -0.14798349142074585,
      -0.40624576807022095,
      0.28210046887397766,
      0.20153795182704926,
      0.0017047368455678225,
      -0.13985183835029602,
      0.24318012595176697,
      0.011913014575839043,
      -0.2251776158809662,
      0.2943289577960968,
      0.32903167605400085
    ],
    [
      -0.11710882931947708,
      0.07275696098804474,
      0.09356892108917236,
      0.034914419054985046,
      -0.13292355835437775,
      0.002101920312270522,
      -0.2620536983013153,
      -0.11537367105484009,
      0.009087365120649338,
      -0.16552278399467468,
      0.04144664853811264,
      -0.09778168797492981,
      -0.10017754882574081,
      0.07477585226297379,
      -0.13592585921287537,
      -0.25839781761169434
    ],
    [
      -0.028186604380607605,
      -0.003717937506735325,
      -0.29887500405311584,
      -0.2891698181629181,
      0.09879128634929657,
      -0.20739413797855377,
      -0.22700044512748718,
      0.04614005982875824,
      0.014165087603032589,
      -0.031005600467324257,
      0.10214091837406158,
      0.11891835182905197,
      -0.18830494582653046,
      -0.045796144753694534,
      0.08609893918037415,
      0.048444703221321106
    ],
    [
      0.06313416361808777,
      0.13430888950824738,
      0.09228254854679108,
      -0.053757306188344955,
      0.14091821014881134,
      0.3187471628189087,
      -0.07018858939409256,
      -0.0808287188410759,
      0.25605154037475586,
      -0.06052357330918312,
      -0.02541757933795452,
      0.30005738139152527,
      -0.057798225432634354,
      -0.03576246276497841,
      0.1528255045413971,
      0.0753026232123375
    ]
  ],
  "player_model.layers.2.bias": [
    0.18887552618980408,
    0.07646015286445618,
    0.6320331692695618,
    -0.03216896206140518,
    -0.11482775211334229,
    -0.3138737082481384,
    0.506674587726593,
    0.1784706711769104,
    0.4204448163509369,
    -0.10094323009252548,
    -0.05525340139865875,
    0.4438953697681427
  ],
  "player_model.layers.4.weight": [
    [
      0.01908872276544571,
      -0.0910029411315918,
      -0.29315173625946045,
      -0.17293651401996613,
      0.03492208942770958,
      0.15978439152240753,
      0.11128585040569305,
      0.363555908203125,
      -0.03019803948700428,
      -0.1409897357225418,
      -0.24867218732833862,
      0.14878305792808533
    ],
    [
      0.337243914604187,
      -0.1968746781349182,
      0.3608132302761078,
      -0.10373634845018387,
      -0.32585227489471436,
      -0.2881564497947693,
      0.12990865111351013,
      0.04920830577611923,
      0.35795125365257263,
      0.30586642026901245,
      -0.12881988286972046,
      0.09953689575195312
    ],
    [
      0.43755000829696655,
      -0.19349056482315063,
      0.1884087473154068,
      -0.16213688254356384,
      -0.4779568016529083,
      -0.23798097670078278,
      0.31310567259788513,
      -0.041906531900167465,
      0.47506386041641235,
      0.11587857455015182,
      -0.18304800987243652,
      0.3502225875854492
    ],
    [
      0.2173069715499878,
      0.15546339750289917,
      -0.12435170263051987,
      0.09787209331989288,
      0.11475265771150589,
      0.15552569925785065,
      -0.1741158366203308,
      -0.201650932431221,
      -0.1805732250213623,
      0.18549901247024536,
      0.1455291211605072,
      0.19728490710258484
    ]
  ],
  "player_model.layers.4.bias": [
    0.055953703820705414,
    0.4177946448326111,
    0.4725373685359955,
    -0.016248231753706932
  ],
  "layers.0.weight": [
    [
      0.08502505719661713,
      -0.15641441941261292,
      -0.0956944152712822,
      -0.06428545713424683,
      -0.2942630648612976,
      -0.06283058226108551,
      0.08960908651351929,
      -0.04545668512582779
    ],
    [
      0.14156918227672577,
      -0.3433835506439209,
      0.1799863874912262,
      -0.05935509875416756,
      -0.07721060514450073,
      -0.2624180316925049,
      -0.36295637488365173,
      -0.23012323677539825
    ],
    [
      -0.1966954469680786,
      -0.16357018053531647,
      -0.07932806760072708,
      -0.2380443513393402,
      -0.2678462862968445,
      0.21005703508853912,
      -0.2222585529088974,
      0.07805059105157852
    ],
    [
      0.007670826744288206,
      -0.21256475150585175,
      -0.1651887595653534,
      0.22302870452404022,
      0.021419480443000793,
      -0.11251213401556015,
      -0.24683432281017303,
      0.03552419692277908
    ],
    [
      -0.18850138783454895,
      0.25981301069259644,
      0.2045484185218811,
      -0.30689114332199097,
      0.40062981843948364,
      -0.016380993649363518,
      -0.19095677137374878,
      -0.082966648042202
    ],
    [
      0.04011234641075134,
      0.3075779974460602,
      0.11264742165803909,
      -0.11409718543291092,
      0.3192434012889862,
      -0.09922121465206146,
      -0.4288073182106018,
      -0.06557448953390121
    ],
    [
      0.009436924010515213,
      0.0815654844045639,
      -0.09817706793546677,
      -0.22996477782726288,
      0.3102097511291504,
      0.13413281738758087,
      -0.32714787125587463,
      0.21544955670833588
    ],
    [
      -0.18599022924900055,
      -0.03604401648044586,
      0.22264671325683594,
      0.05061063915491104,
      -0.3070117235183716,
      -0.0594988577067852,
      -0.20782580971717834,
      0.16048742830753326
    ],
    [
      0.18584008514881134,
      -0.014070925302803516,
      -0.2749707102775574,
      -0.1908201426267624,
      -0.18628649413585663,
      -0.20958511531352997,
      0.05899692699313164,
      0.137327179312706
    ],
    [
      -0.04390997812151909,
      0.3232097029685974,
      -0.21275769174098969,
      0.2840330898761749,
      -0.25518566370010376,
      -0.03675670549273491,
      -0.32903069257736206,
      -0.2335464358329773
    ],
    [
      0.22693400084972382,
      0.43360286951065063,
      0.21512636542320251,
      -0.31332725286483765,
      0.22559425234794617,
      0.013349916785955429,
      -0.07731770724058151,
      0.24215848743915558
    ],
    [
      -0.04302341118454933,
      0.2699640691280365,
      0.20280365645885468,
      -0.2351461797952652,
      0.014747949317097664,
      -0.2682361304759979,
      -0.03909872844815254,
      -0.20445501804351807
    ],
    [
      -0.16130411624908447,
      0.3193855285644531,
      0.10088461637496948,
      0.3280896842479706,
      0.3546585440635681,
      -0.41016581654548645,
      -0.045590128749608994,
      0.037729863077402115
    ],
    [
      0.291481077671051,
      -0.21082136034965515,
      -0.02910003997385502,
      -0.10804451256990433,
      -0.00742441276088357,
      0.13307316601276398,
      0.31116899847984314,
      -0.04634278267621994
    ],
    [
      -0.07377061992883682,
      -0.2637559771537781,
      -0.10985396057367325,
      -0.020510930567979813,
      -0.2061525285243988,
      -0.11659234762191772,
      0.01864977553486824,
      0.2829062044620514
    ],
    [
      -0.05670225992798805,
      0.38618314266204834,
      0.2864970266819,
      0.028587913140654564,
      0.028539828956127167,
      -0.20691615343093872,
      0.17705649137496948,
      -0.18291372060775757
    ]
  ],
  "layers.0.bias": [
    -0.12485630065202713,
    -0.20138153433799744,
    0.15278729796409607,
    -0.023763934150338173,
    -0.17349450290203094,
    -0.14805331826210022,
    -0.015202183276414871,
    -0.1315658688545227,
    -0.023175982758402824,
    0.31529948115348816,
    0.3129565119743347,
    0.17716176807880402,
    0.06404713541269302,
    0.013140086084604263,
    -0.007095624227076769,
    0.2869715392589569
  ],
  "layers.2.weight": [
    [
      -0.13353951275348663,
      0.05969124659895897,
      0.1359640508890152,
      -0.13434554636478424,
      -0.40944933891296387,
      -0.1684311330318451,
      -0.09832209348678589,
      0.06608064472675323,
      -0.028434982523322105,
      0.06449000537395477,
      -0.14440007507801056,
      -0.34618112444877625,
      -0.344585657119751,
      0.54095458984375,
      0.18779191374778748,
      -0.20417067408561707
    ]
  ],
  "layers.2.bias": [
    0.13815227150917053
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

        if season <= 28:
            return [
                prediction.item() + random.random() - 0.5
            ]
        else:
            return [
                prediction.item()
            ]

