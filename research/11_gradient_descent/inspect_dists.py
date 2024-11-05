# research/11_gradient_descent/mu_dist.json
import json
import numpy as np
import matplotlib.pyplot as plt
from distfit import distfit

# Load the JSON file
with open('research/11_gradient_descent/mu_dist.json', 'r') as f:
    values = np.array(json.load(f))

# Initialize the distfit model
dist = distfit(distr=['norm'])

# Fit distributions to your data
dist.fit_transform(values)

dist.plot()
plt.show()
