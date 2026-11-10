# ==========================================================
# OptiCrop - Crop Recommendation System
# Random Forest Classifier with Top-3 recommendations
# ==========================================================
import json
import os
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "crop_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "model", "label_encoder.pkl")
IMAGE_DIR = os.path.join(BASE_DIR, "static", "images")
DATASET_PATH = os.path.join(BASE_DIR, "Crop_recommendation.csv")
app = Flask(__name__, template_folder="templates", static_folder="static")

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

SOIL_FIELDS = [
    {"name": "N", "label": "Nitrogen (N)", "unit": "kg/ha", "min": 0, "max": 140, "example": 90},
    {"name": "P", "label": "Phosphorus (P)", "unit": "kg/ha", "min": 5, "max": 145, "example": 42},
    {"name": "K", "label": "Potassium (K)", "unit": "kg/ha", "min": 5, "max": 205, "example": 43},
    {"name": "ph", "label": "Soil pH", "unit": "", "min": 3.5, "max": 9.9, "example": 6.5},
]
CLIMATE_FIELDS = [
    {"name": "temperature", "label": "Temperature", "unit": "°C", "min": 5, "max": 50, "example": 25},
    {"name": "humidity", "label": "Relative humidity", "unit": "%", "min": 10, "max": 100, "example": 80},
    {"name": "rainfall", "label": "Rainfall", "unit": "mm", "min": 15, "max": 300, "example": 200},
]
ALL_FIELDS = SOIL_FIELDS + CLIMATE_FIELDS

CROP_TE_MAP = {
    "rice": "వరి", "maize": "మొక్కజొన్న", "jute": "జనపనార", "cotton": "ప్రత్తి",
    "coconut": "కొబ్బరి", "papaya": "బొప్పాయి", "orange": "నారింజ", "apple": "ఆపిల్",
    "muskmelon": "ఖర్బూజా", "watermelon": "పుచ్చకాయ", "grapes": "ద్రాక్ష",
    "mango": "మామిడి", "banana": "అరటి", "pomegranate": "దానిమ్మ",
    "lentil": "మసూర్ పప్పు", "blackgram": "మినుములు", "mungbean": "పెసలు",
    "mothbeans": "బొబ్బర్లు", "pigeonpeas": "కందులు", "kidneybeans": "రాజ్మా",
    "chickpea": "శనగలు", "coffee": "కాఫీ",
}
FEATURE_LABELS = {
    "N": "nitrogen", "P": "phosphorus", "K": "potassium",
    "temperature": "temperature", "humidity": "humidity", "ph": "soil pH", "rainfall": "rainfall"
}
# Crop-wise average feature profiles from the project's own training dataset.
# These are descriptive reference values, not causal explanations or SHAP values.
try:
    _training_data = pd.read_csv(DATASET_PATH)
    CROP_PROFILES = _training_data.groupby("label")[list(FEATURE_LABELS)].mean().to_dict(orient="index")
    FEATURE_RANGES = (_training_data[list(FEATURE_LABELS)].max() - _training_data[list(FEATURE_LABELS)].min()).to_dict()
except Exception:
    CROP_PROFILES, FEATURE_RANGES = {}, {}


def parse_inputs(data):
    values = {}
    for field in ALL_FIELDS:
        raw = data.get(field["name"], "")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f'{field["label"]} must be a number.')
        if not field["min"] <= value <= field["max"]:
            raise ValueError(f'{field["label"]} must be between {field["min"]} and {field["max"]} {field["unit"]}.')
        values[field["name"]] = value
    return values


def crop_reasons(crop_name, values, count=3):
    """Describe closest input matches to the crop's average dataset profile.

    This is a dataset-profile comparison, not a causal explanation or SHAP output.
    """
    profile = CROP_PROFILES.get(crop_name)
    if not profile:
        return [{"en": "The model ranked this crop among the top recommendations for your entered values.",
                 "te": "మీరు నమోదు చేసిన విలువల ఆధారంగా మోడల్ ఈ పంటను ఉత్తమ సూచనల్లో చేర్చింది."}]
    differences = []
    telugu_labels = {
        "N": "నత్రజని", "P": "భాస్వరం", "K": "పొటాషియం",
        "temperature": "ఉష్ణోగ్రత", "humidity": "తేమ", "ph": "నేల pH", "rainfall": "వర్షపాతం"
    }
    for feature, label in FEATURE_LABELS.items():
        span = float(FEATURE_RANGES.get(feature, 0)) or 1.0
        distance = abs(float(values[feature]) - float(profile[feature])) / span
        differences.append((distance, feature, label, float(profile[feature])))
    differences.sort(key=lambda item: item[0])
    return [
        {"en": f"Your {label} ({values[feature]:.1f}) is relatively close to this crop's dataset average ({average:.1f}).",
         "te": f"మీ {telugu_labels[feature]} విలువ ({values[feature]:.1f}) ఈ పంటకు సంబంధించిన డేటాసెట్ సగటు ({average:.1f})కు దగ్గరగా ఉంది."}
        for _, feature, label, average in differences[:count]
    ]


def top_crops(values, n=3):
    columns = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    sample = pd.DataFrame([values])[columns]
    probabilities = model.predict_proba(sample)[0]
    class_names = label_encoder.inverse_transform(model.classes_)
    ranked = sorted(zip(class_names, probabilities), key=lambda item: item[1], reverse=True)[:n]
    results = []
    for crop_name, probability in ranked:
        key = crop_name.lower().strip().replace(" ", "")
        has_image = os.path.exists(os.path.join(IMAGE_DIR, f"{key}.jpg"))
        results.append({
            "crop_en": crop_name.capitalize(), "crop_te": CROP_TE_MAP.get(key, ""),
            "confidence": round(float(probability) * 100, 1),
            "image": f"/static/images/{key}.jpg" if has_image else None,
            "reasons": crop_reasons(crop_name, values),
        })
    return results


@app.route("/")
def home():
    return render_template("index.html", soil_fields=SOIL_FIELDS, climate_fields=CLIMATE_FIELDS)


@app.route("/weather")
def weather():
    """Get current temperature/humidity and 7-day precipitation for a city."""
    city = (request.args.get("city") or "").strip()
    if not city or len(city) > 100:
        return jsonify({"success": False, "error": "Enter a valid city name."}), 400
    try:
        geo_url = "https://geocoding-api.open-meteo.com/v1/search?" + urlencode({"name": city, "count": 1, "language": "en", "format": "json"})
        geo_req = Request(geo_url, headers={"User-Agent": "OptiCrop/1.0"})
        with urlopen(geo_req, timeout=10) as response:
            geo = json.loads(response.read().decode("utf-8"))
        places = geo.get("results") or []
        if not places:
            return jsonify({"success": False, "error": "City not found. Check the spelling and try again."}), 404
        place = places[0]
        forecast_params = urlencode({
            "latitude": place["latitude"], "longitude": place["longitude"],
            "current": "temperature_2m,relative_humidity_2m",
            "daily": "precipitation_sum", "forecast_days": 7, "timezone": "auto"
        })
        forecast_req = Request("https://api.open-meteo.com/v1/forecast?" + forecast_params,
                               headers={"User-Agent": "OptiCrop/1.0"})
        with urlopen(forecast_req, timeout=10) as response:
            forecast = json.loads(response.read().decode("utf-8"))
        current = forecast.get("current", {})
        rain_values = forecast.get("daily", {}).get("precipitation_sum", [])
        rain_total = sum(float(value or 0) for value in rain_values)
        return jsonify({
            "success": True,
            "city": f'{place.get("name", city)}, {place.get("country", "")}',
            "temperature": current.get("temperature_2m"),
            "humidity": current.get("relative_humidity_2m"),
            "rainfall": round(rain_total, 1),
            "rainfall_note": "7-day forecast total only; replace with growing-season rainfall estimate for a more suitable model input.",
        })
    except Exception:
        app.logger.exception("Weather lookup failed")
        return jsonify({"success": False, "error": "Weather service is unavailable right now. Please enter the values manually."}), 502


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) if request.is_json else request.form
    try:
        values = parse_inputs(data or {})
        recommendations = top_crops(values)
    except ValueError as error:
        if request.is_json:
            return jsonify({"success": False, "error": str(error)}), 400
        return render_template("result.html", error=str(error)), 400
    except Exception:
        app.logger.exception("Prediction failed")
        if request.is_json:
            return jsonify({"success": False, "error": "Prediction failed."}), 500
        return render_template("result.html", error="Something went wrong while predicting. Please try again."), 500
    if request.is_json:
        return jsonify({"success": True, "recommendations": recommendations, "inputs": values})
    return render_template("result.html", recommendations=recommendations, inputs=values,
                           soil_fields=SOIL_FIELDS, climate_fields=CLIMATE_FIELDS)


if __name__ == "__main__":
    app.run(debug=True)
