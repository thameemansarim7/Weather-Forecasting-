import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# === Load dataset ===
file_path = 'data/GlobalWeatherRepository.csv'

if not os.path.exists(file_path):
    print(f"❌ ERROR: File '{file_path}' not found!")
    exit()

data = pd.read_csv(file_path)
print("✅ Dataset loaded successfully!")

# === Print available columns ===
print("🔎 Columns available:", data.columns.tolist())

# === Rename or map required columns ===
data = data.rename(columns={
    'location_name': 'city',
    'temperature_celsius': 'temperature',
    'wind_kph': 'wind_speed',
    'pressure_mb': 'pressure',
    'precip_mm': 'precipitation'
})

# === Required columns ===
required_columns = ['city', 'temperature', 'humidity', 'wind_speed', 'pressure', 'precipitation']
missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    print(f"❌ ERROR: Missing columns in dataset: {missing_columns}")
    exit()

# === Drop missing values ===
data = data.dropna(subset=required_columns)

# === Features and multi-target ===
X = data[['temperature', 'humidity', 'wind_speed', 'pressure']]
y = data[['temperature', 'precipitation']]  # Multi-output target

# === Scale features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Split into train/test ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === Train model ===
model = LinearRegression()
model.fit(X_train, y_train)

# === Evaluate ===
score = model.score(X_test, y_test)
print(f"📊 Model R² score: {score:.4f}")

# === Save model and scaler ===
os.makedirs("backend", exist_ok=True)
with open("backend/model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("backend/scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("✅ Multi-output model & scaler trained and saved to backend/")
