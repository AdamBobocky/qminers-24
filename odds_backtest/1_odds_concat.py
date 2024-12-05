import os
import pandas as pd

def load_and_concat_csvs(input_dir, output_file):
    # List to store individual DataFrames
    dataframes = []

    # Iterate through all CSV files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(input_dir, filename)

            # Load the CSV file into a DataFrame
            df = pd.read_csv(filepath)

            # Add a column "Season" with the filename (without .csv)
            df['Season'] = os.path.splitext(filename)[0]

            # Append the DataFrame to the list
            dataframes.append(df)

    # Concatenate all DataFrames
    concatenated_df = pd.concat(dataframes, ignore_index=True)

    # Save the concatenated DataFrame to the output file
    concatenated_df.to_csv(output_file, index=False)
    print(f"Concatenated CSV saved to {output_file}")

# Define the input directory and output file
input_directory = "data/odds"
output_csv = "data/odds/concat.csv"

# Execute the function
load_and_concat_csvs(input_directory, output_csv)
