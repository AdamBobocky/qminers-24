import json
import pandas as pd

games_df = pd.read_csv('data/games-merged.csv')
odds_df = pd.read_csv('data/odds/concat.csv')

print(len(games_df))
print(len(odds_df))

games_df['RealDate'] = pd.to_datetime(games_df['Date']) + pd.Timedelta(days=4011)

# print(odds_df)
#        Date  Rot VH         Team  1st  2nd  3rd  4th  Final   Open  Close   ML     2H Season
# 0      1017  501  V       Boston   19   19   33   28     99  214.5    216  166    1.5  17-18
# 1      1017  502  H    Cleveland   29   25   18   30    102      4    4.5 -195    110  17-18
# 2      1017  503  V      Houston   34   28   26   34    122    232  231.5  364  115.5  17-18
# 3      1017  504  H  GoldenState   35   36   30   20    121      9      9 -470    2.5  17-18
# 4      1018  701  V     Brooklyn   30   33   35   33    131    212  216.5  135  112.5  17-18
# ...     ...  ... ..          ...  ...  ...  ...  ...    ...    ...    ...  ...    ...    ...

# Your job is to now convert the odds_df 'Date' and Season into a pd datetime, use the Date and Season columns

# Function to convert Date and Season into a datetime
def convert_to_datetime(row):
    # Extract Date and Season
    mmdd = str(row['Date']).zfill(4)  # Ensure Date has 4 digits
    season_start, season_end = map(int, row['Season'].split('-'))

    # Extract month and day from Date
    month = int(mmdd[:2])
    day = int(mmdd[2:])

    # Determine the year based on the game date
    year = season_start + 2000 if month >= 7 else season_end + 2000

    # Create and return the datetime
    return pd.Timestamp(year=year, month=month, day=day)

# Apply the function to create a new datetime column
odds_df['RealDate'] = odds_df.apply(convert_to_datetime, axis=1)

name_id_map = {
    0: 'GoldenState',
    6: 'Chicago',
    3: 'Portland',
    21: 'NewOrleans',
    26: 'Atlanta',
    19: 'Boston',
    42: 'Denver',
    43: 'Phoenix',
    30: 'Sacramento',
    25: 'Memphis',
    13: 'Minnesota',
    15: 'Cleveland',
    41: 'Charlotte',
    34: 'Brooklyn',
    36: 'Utah',
    23: 'Washington',
    14: 'SanAntonio',
    16: 'OklahomaCity',
    33: 'Toronto',
    24: 'Orlando',
    35: 'NewYork',
    37: 'Indiana',
    31: 'Miami',
    29: 'LAClippers',
    39: 'Philadelphia',
    11: 'Dallas',
    2: 'Milwaukee',
    5: 'Detroit',
    17: 'LALakers',
    32: 'Houston'
}

mathces = 0
total_potential = 0

odds_map = {}

for i in games_df.index:
    current = games_df.loc[i]

    if current['POFF'] == 1:
        continue

    matching_date = odds_df[odds_df['RealDate'] == current['RealDate']]

    if len(matching_date):
        HID = current['HID']
        AID = current['AID']

        total_potential += 1

        some_new = HID not in name_id_map or AID not in name_id_map

        if HID in name_id_map and AID in name_id_map:
            home_match = matching_date[matching_date['Team'] == name_id_map[HID]]
            away_match = matching_date[matching_date['Team'] == name_id_map[AID]]

            if len(home_match) > 1 or len(away_match) > 1:
                print('???', len(home_match), len(away_match))
            elif len(home_match) == 1 or len(away_match) == 1:
                try:
                    if home_match['Final'].iloc[0] == current['HSC'] and away_match['Final'].iloc[0] == current['ASC']:
                        mathces += 1
                        print('Match it', mathces, total_potential)
                        print(current['RealDate'])
                        odds_map[i] = [
                            str(home_match['Open'].iloc[0]), str(home_match['Close'].iloc[0]), str(home_match['ML'].iloc[0]), str(home_match['Final'].iloc[0]),
                            str(away_match['Open'].iloc[0]), str(away_match['Close'].iloc[0]), str(away_match['ML'].iloc[0]), str(away_match['Final'].iloc[0])
                        ]
                    else:
                        print('???')
                except:
                    pass

        if some_new:
            print('-------------------')
            print(current[['HID', 'AID', 'HSC', 'ASC']])
            print(matching_date)

with open('data/odds/odds_map.json', 'w') as json_file:
    json.dump(odds_map, json_file, indent=2)

# 2015-10-27
