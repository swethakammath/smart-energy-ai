
import pandas as pd
import numpy as np
import os

# Create data folder
os.makedirs("data", exist_ok=True)

# Set a fixed seed so results can be reproduced
np.random.seed(42)

# Number of days
number_of_days = 60

# Create dates
dates = pd.date_range(
    start="2026-07-01",
    periods=number_of_days,
    freq="D"
)

# Generate simulated daily energy consumption
energy_values = []

for day in range(number_of_days):

    # Normal daily consumption
    base_energy = np.random.uniform(4, 10)

    # Slightly higher usage on some days
    extra_usage = np.random.uniform(0, 3)

    total_energy = base_energy + extra_usage

    energy_values.append(round(total_energy, 2))

# Electricity rate
electricity_rate = 8.0

# Calculate daily cost
cost_values = [
    round(energy * electricity_rate, 2)
    for energy in energy_values
]

# Create dataset
data = pd.DataFrame({
    "date": dates,
    "mode": "Home Mode",
    "total_energy_kwh": energy_values,
    "cost_inr": cost_values
})

# Save dataset
file_path = "data/historical_energy.csv"

data.to_csv(
    file_path,
    index=False
)

print("✅ Historical dataset created successfully!")

print(f"📁 Saved to: {file_path}")

print(f"📊 Total records: {len(data)}")

print("\nFirst five records:")

print(data.head())