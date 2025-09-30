import numpy as np
import pandas as pd
import os

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Generate time-series data
t = np.linspace(0, 50, 1000)   # 1000 time steps
data = np.sin(t) + 0.1 * np.random.randn(len(t))  # sine wave + noise

# Save to CSV
df = pd.DataFrame({"value": data})
df.to_csv("data/sample_large.csv", index=False)

print("Dataset saved to data/sample_large.csv with 1000 rows")
