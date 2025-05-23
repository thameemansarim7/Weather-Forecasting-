from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import requests
import os
import uuid
import traceback
import time

# === Config ===
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
STATIC_DIR = os.path.join(BASE_DIR, "../static")
TEMPLATE_DIR = os.path.join(BASE_DIR, "../templates")

OPENWEATHER_API_KEY = "8c306b2f7be12666a5e09b43ae214a18"
use_fake_data = False  # Toggle to True for development/testing

# Ensure static directory exists
os.makedirs(STATIC_DIR, exist_ok=True)

# === Load model & scaler ===
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# === Setup Flask ===
app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATE_DIR)
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        city = request.form.get("city")
        if not city:
            return jsonify({"error": "City name is required."}), 400

        if use_fake_data:
            print("[SIMULATION MODE] Using fake weather data...")
            weather_condition = "rain"
            temperature = 22.3
            humidity = 92
            wind_speed = 3.8
            pressure = 1001
            icon = "09d"
            precipitation = 3.2
        else:
            print(f"[INFO] Fetching real weather data for '{city}'...")
            url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"

            data = None
            for attempt in range(3):
                try:
                    print(f"API request attempt {attempt+1}")
                    response = requests.get(url, timeout=15)

                    if response.status_code == 401:
                        return jsonify({'error': 'API authorization failed'}), 401
                    if response.status_code == 404:
                        return jsonify({'error': f"City '{city}' not found"}), 404

                    response.raise_for_status()
                    data = response.json()
                    print("[INFO] Weather data retrieved successfully")
                    break
                except requests.exceptions.Timeout:
                    if attempt < 2:
                        time.sleep(2)
                    else:
                        return jsonify({'error': 'Timeout after 3 attempts'}), 504
                except requests.exceptions.RequestException:
                    if attempt < 2:
                        time.sleep(2)
                    else:
                        return jsonify({'error': 'Failed to retrieve weather data'}), 500

            if not data:
                return jsonify({'error': 'Failed to retrieve weather data'}), 500

            try:
                temperature = data["main"]["temp"]
                humidity = data["main"]["humidity"]
                wind_speed = data["wind"]["speed"]
                pressure = data["main"]["pressure"]
                weather_condition = data["weather"][0]["main"].lower()
                icon = data["weather"][0].get("icon", "01d")

                rain_data = data.get("rain", {})
                snow_data = data.get("snow", {})

                rain = rain_data.get("1h", 0) if isinstance(rain_data, dict) else 0
                snow = snow_data.get("1h", 0) if isinstance(snow_data, dict) else 0

                precipitation = round(float(rain) + float(snow), 2)
            except KeyError as e:
                return jsonify({'error': f'Unexpected API response format'}), 500

        features = np.array([[temperature, humidity, wind_speed, pressure]])
        scaled = scaler.transform(features)
        prediction = model.predict(scaled)

        if isinstance(prediction[0], (list, np.ndarray)) and len(prediction[0]) == 2:
            predicted_temp = round(float(prediction[0][0]), 2)
            predicted_precip = round(float(prediction[0][1]), 2)
        elif isinstance(prediction[0], (int, float)):
            predicted_temp = round(float(prediction[0]), 2)
            predicted_precip = 0.0
        else:
            predicted_temp = 0.0
            predicted_precip = 0.0

        video_map = {
            "clear": "sunny.mp4",
            "clouds": "cloudy.mp4",
            "rain": "rain.mp4",
            "drizzle": "rain.mp4",
            "snow": "snow.mp4",
            "thunderstorm": "rain.mp4"
        }
        video_file = video_map.get(weather_condition, "default.mp4")

        response_data = {
            "city": city.title(),
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "pressure": pressure,
            "precipitation": precipitation,
            "weather_condition": weather_condition,
            "icon": icon,
            "predicted_temperature": predicted_temp,
            "predicted_precipitation": predicted_precip,
            "video": video_file
        }

        return jsonify(response_data)

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500

@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)

@app.route("/test-connection", methods=["GET"])
def test_connection():
    try:
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?q=London&appid={OPENWEATHER_API_KEY}"
        weather_response = requests.get(weather_url, timeout=5)
        weather_status = weather_response.status_code

        return jsonify({
            "openweather_api": f"Status: {weather_status}",
            "static_directory": "Exists" if os.path.exists(STATIC_DIR) else "Missing"
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
