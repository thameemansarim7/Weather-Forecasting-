from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import requests
import os
import traceback
import time
import logging

# === Setup Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# === Config ===
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
STATIC_DIR = os.path.join(BASE_DIR, "../static")
TEMPLATE_DIR = os.path.join(BASE_DIR, "../templates")

OPENWEATHER_API_KEY = "4831d5073590e581f62a407e05432e08"

use_fake_data = False

os.makedirs(STATIC_DIR, exist_ok=True)

# === Load model & scaler ===
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info("ML model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model or scaler: {str(e)}")
    raise

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
            weather_condition = "rain"
            temperature = 22.3
            humidity = 92
            wind_speed = 3.8
            pressure = 1001
            icon = "09d"
            precipitation = 3.2
        else:
            weather_data = fetch_weather_data(city)
            if "error" in weather_data:
                return jsonify(weather_data), weather_data.get("status_code", 500)

            temperature = weather_data["main"]["temp"]
            humidity = weather_data["main"]["humidity"]
            wind_speed = weather_data["wind"]["speed"]
            pressure = weather_data["main"]["pressure"]
            weather_condition = weather_data["weather"][0]["main"].lower()
            icon = weather_data["weather"][0].get("icon", "01d")
            precipitation = calculate_precipitation(weather_data)

        features = np.array([[temperature, humidity, wind_speed, pressure]])
        scaled = scaler.transform(features)
        prediction = model.predict(scaled)
        predicted_values = process_prediction(prediction)

        response_data = {
            "city": city.title(),
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "pressure": pressure,
            "precipitation": precipitation,
            "weather_condition": weather_condition,
            "icon": icon,
            "predicted_temperature": predicted_values["temperature"],
            "predicted_precipitation": predicted_values["precipitation"],
            "video": choose_weather_video(weather_condition)
        }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Unhandled exception in prediction endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

def fetch_weather_data(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 401:
                return {"error": "Weather API authorization failed", "status_code": 401}
            if response.status_code == 404:
                return {"error": f"City '{city}' not found", "status_code": 404}
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            time.sleep(2 * (attempt + 1))
        except requests.exceptions.RequestException as e:
            logger.error(f"Weather API error: {str(e)}")
            time.sleep(2 * (attempt + 1))
    return {"error": "Failed to retrieve weather data after multiple attempts", "status_code": 503}

def calculate_precipitation(weather_data):
    try:
        rain = float(weather_data.get("rain", {}).get("1h", 0))
        snow = float(weather_data.get("snow", {}).get("1h", 0))
        return round(rain + snow, 2)
    except Exception as e:
        logger.warning(f"Error calculating precipitation: {str(e)}")
        return 0.0

def process_prediction(prediction):
    try:
        if isinstance(prediction[0], (list, np.ndarray)) and len(prediction[0]) >= 2:
            predicted_temp = round(float(prediction[0][0]), 2)
            predicted_precip = round(float(prediction[0][1]), 2)
        elif isinstance(prediction[0], (int, float, np.number)):
            predicted_temp = round(float(prediction[0]), 2)
            predicted_precip = 0.0
        else:
            predicted_temp = 0.0
            predicted_precip = 0.0
        return {"temperature": predicted_temp, "precipitation": predicted_precip}
    except Exception as e:
        logger.error(f"Error processing prediction: {str(e)}")
        return {"temperature": 0.0, "precipitation": 0.0}

def choose_weather_video(weather_condition):
    video_map = {
        "clear": "sunny.mp4",
        "clouds": "cloudy.mp4",
        "rain": "rain.mp4",
        "drizzle": "rain.mp4",
        "snow": "snow.mp4",
        "thunderstorm": "rain.mp4",
        "mist": "cloudy.mp4",
        "fog": "cloudy.mp4",
        "haze": "cloudy.mp4"
    }
    return video_map.get(weather_condition, "default.mp4")

@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)

@app.route("/test-apis", methods=["GET"])
def test_apis():
    results = {
        "openweather_api": "Unknown",
        "static_directory": "Exists" if os.path.exists(STATIC_DIR) else "Missing",
        "model_loaded": "Yes" if model is not None else "No",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    try:
        test_city = "London"
        url = f"https://api.openweathermap.org/data/2.5/weather?q={test_city}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=5)
        results["openweather_api"] = "Working" if response.status_code == 200 else f"Error: Status {response.status_code}"
    except Exception as e:
        results["openweather_api"] = f"Error: {str(e)}"
    return jsonify(results)

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "version": "1.0.0"})

if __name__ == "__main__":
    logger.info("Weather Prediction App starting up...")
    app.run(debug=True)
