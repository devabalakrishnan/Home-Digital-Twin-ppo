import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import timedelta

# Load Dataset
df = pd.read_csv('data/PPO_SYNTHETIC_DATASET.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

# Feature Engineering
df['hour'] = df['datetime'].dt.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Define Appliances
appliances = ['Fridge', 'Heater', 'Fans', 'Lights', 'TV', 'Microwave', 'Washing Machine']
features = ['hour_sin', 'hour_cos', 'occupancy', 'electricity_price']

# Train XGBoost models for each appliance
for app in appliances:
    X = df[features]
    y = df[app]
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X, y)
    df[f'{app}_pred'] = model.predict(X)

# Generate Future 24h Forecast
last_date = df['datetime'].max()
future_dates = [last_date + timedelta(hours=i) for i in range(1, 25)]
future_df = pd.DataFrame({'datetime': future_dates})
future_df['hour'] = future_df['datetime'].dt.hour
future_df['hour_sin'] = np.sin(2 * np.pi * future_df['hour'] / 24)
future_df['hour_cos'] = np.cos(2 * np.pi * future_df['hour'] / 24)
# Simulate future occupancy and price
future_df['occupancy'] = np.random.randint(1, 5, 24)
future_df['electricity_price'] = [0.65 if (7<=h<=10 or 18<=h<=22) else 0.25 for h in future_df['hour']]

for app in appliances:
    X_future = future_df[['hour_sin', 'hour_cos', 'occupancy', 'electricity_price']]
    model = xgb.XGBRegressor().fit(df[features], df[app])
    future_df[app] = model.predict(X_future)

future_df.to_csv('data/next_day_prediction.csv', index=False)
print("Forecast generated: data/next_day_prediction.csv")