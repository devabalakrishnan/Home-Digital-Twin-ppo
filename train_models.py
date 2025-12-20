import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import timedelta
import os

# 1. Setup Directories
if not os.path.exists('data'):
    os.makedirs('data')

# 2. Load and Prepare Data
print("Loading PPO_SYNTHETIC_DATASET.csv...")
df = pd.read_csv('data/PPO_SYNTHETIC_DATASET.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

# Feature Engineering
df['hour'] = df['datetime'].dt.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Defining appliances exactly as they appear in your CSV headers
appliances = ['Fridge', 'Heater', 'Fans', 'Lights', 'TV', 'Microwave', 'Washing Machine']
features = ['hour_sin', 'hour_cos', 'occupancy', 'electricity_price']

print("Training XGBoost Forecasting Engine...")

# 3. Training Loop
models = {}
for app in appliances:
    X = df[features]
    y = df[app]
    
    # We use a Regressor for the Digital Twin forecast
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5
    )
    model.fit(X, y)
    models[app] = model

# 4. Generate Future 24-Hour Forecast
print("Generating next-day predictions...")
last_timestamp = df['datetime'].max()
future_dates = [last_timestamp + timedelta(hours=i) for i in range(1, 25)]

future_df = pd.DataFrame({'datetime': future_dates})
future_df['hour'] = future_df['datetime'].dt.hour
future_df['hour_sin'] = np.sin(2 * np.pi * future_df['hour'] / 24)
future_df['hour_cos'] = np.cos(2 * np.pi * future_df['hour'] / 24)

# Simulate Environment for the Digital Twin
# Occupancy: Higher in morning/evening
future_df['occupancy'] = [np.random.randint(3, 6) if (7<=h<=9 or 18<=h<=22) else np.random.randint(1, 3) for h in future_df['hour']]

# Electricity Price: Peak hours are expensive (Correcting the logic for PPO)
future_df['electricity_price'] = [0.65 if (7<=h<=10 or 18<=h<=22) else 0.25 for h in future_df['hour']]

# Apply models to predict future load
for app in appliances:
    X_future = future_df[features]
    future_df[app] = models[app].predict(X_future)
    # Ensure no negative values
    future_df[app] = future_df[app].clip(lower=0)

# 5. Save Output for app.py
output_path = 'data/next_day_prediction.csv'
future_df.to_csv(output_path, index=False)

print(f"Success! Prediction file saved at: {output_path}")
print("You can now run 'streamlit run app.py'")