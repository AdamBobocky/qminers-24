import pandas as pd

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('data/games.csv')

# Extract the HID and AID columns and merge them into one series
merged_ids = pd.concat([df['HID'], df['AID']])

# Get the unique values from the merged HID and AID columns
unique_ids = merged_ids.unique()

# Print the unique IDs
print(f'Unique teams({len(unique_ids)}):', unique_ids.tolist())
