import pandas as pd

df = pd.read_csv('data/games.csv')

# Assuming df is your DataFrame
df['Difference'] = df['HSC'] - df['ASC']

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(df['Difference'], bins=80, kde=True)
plt.xlabel('Difference (HSC - ASC)')
plt.title('Distribution of Differences between HSC and ASC')
plt.show()

from scipy import stats
import numpy as np

# List of distributions to check
distributions = [stats.norm, stats.expon, stats.gamma, stats.t]

# Array to store results
results = []

# Calculate difference values
data = df['Difference'].dropna()

# Fit and test each distribution
for distribution in distributions:
    # Fit the distribution to the data
    params = distribution.fit(data)

    # Perform a KS test
    ks_stat, p_value = stats.kstest(data, distribution.name, args=params)

    # Save the results
    results.append((distribution.name, ks_stat, p_value))

# Display results
results_df = pd.DataFrame(results, columns=['Distribution', 'KS Statistic', 'p-value'])
print(results_df.sort_values(by='KS Statistic'))
