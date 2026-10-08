# ==========================================================
# OptiCrop - Crop Recommendation System
# Model: Random Forest Classifier (Top-3 recommendations)
# ==========================================================

import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "crop_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "model", "label_encoder.pkl")
IMAGE_DIR = os.path.join(BASE_DIR, "static", "images")

app = Flask(__name__, template_folder="templates", static_folder="static")

# ----------------------------------------------------------
# Load model and label encoder once at startup
# ----------------------------------------------------------
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# ----------------------------------------------------------
# Input fields: label, unit, allowed range, placeholder
# (the form on the home page is generated from this)
# ----------------------------------------------------------
SOIL_FIELDS = [
    {"name": "N",  "label": "Nitrogen (N)",   "unit": "kg/ha", "min": 0,   "max": 140, "example": 90},
    {"name": "P",  "label": "Phosphorus (P)", "unit": "kg/ha", "min": 5,   "max": 145, "example": 42},
    {"name": "K",  "label": "Potassium (K)",  "unit": "kg/ha", "min": 5,   "max": 205, "example": 43},
    {"name": "ph", "label": "Soil pH",        "unit": "",      "min": 3.5, "max": 9.9, "example": 6.5},
]
CLIMATE_FIELDS = [
    {"name": "temperature", "label": "Temperature",       "unit": "°C", "min": 5, "max": 50,  "example": 25},
    {"name": "humidity",    "label": "Relative humidity", "unit": "%",  "min": 10, "max": 100, "example": 80},
    {"name": "rainfall",    "label": "Rainfall",          "unit": "mm", "min": 15, "max": 300, "example": 200},
]
ALL_FIELDS = SOIL_FIELDS + CLIMATE_FIELDS

# Telugu names
CROP_TE_MAP = {
    "rice": "వరి", "maize": "మొక్కజొన్న", "jute": "జనపనార", "cotton": "ప్రత్తి",
    "coconut": "కొబ్బరి", "papaya": "బొప్పాయి", "orange": "నారింజ", "apple": "ఆపిల్",
    "muskmelon": "ఖర్బూజా", "watermelon": "పుచ్చకాయ", "grapes": "ద్రాక్ష",
    "mango": "మామిడి", "banana": "అరటి", "pomegranate": "దానిమ్మ",
    "lentil": "మసూర్ పప్పు", "blackgram": "మినుములు", "mungbean": "పెసలు",
    "mothbeans": "బొబ్బర్లు", "pigeonpeas": "కందులు", "kidneybeans": "రాజ్మా",
    "chickpea": "శనగలు", "coffee": "కాఫీ",
}


def parse_inputs(data):
    """Read and validate all form values. Raises ValueError with a clear message."""
    values = {}
    for f in ALL_FIELDS:
        raw = data.get(f["name"], "")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{f['label']} must be a number.")
        if not f["min"] <= value <= f["max"]:
            raise ValueError(
                f"{f['label']} must be between {f['min']} and {f['max']} {f['unit']}".strip() + "."
            )
        values[f["name"]] = value
    return values


def top_crops(values, n=3):
    """Return the n most likely crops using Random Forest probabilities."""
    columns = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    sample = pd.DataFrame([values])[columns]

    probabilities = model.predict_proba(sample)[0]
    # Use the model's own class order so names always match probabilities
    class_names = label_encoder.inverse_transform(model.classes_)

    ranked = sorted(zip(class_names, probabilities), key=lambda x: x[1], reverse=True)[:n]

    results = []
    for crop_name, prob in ranked:
        key = crop_name.lower().strip().replace(" ", "")
        has_image = os.path.exists(os.path.join(IMAGE_DIR, f"{key}.jpg"))
        results.append({
            "crop_en": crop_name.capitalize(),
            "crop_te": CROP_TE_MAP.get(key, ""),
            "confidence": round(float(prob) * 100, 1),
            "image": f"/static/images/{key}.jpg" if has_image else None,
        })
    return results


@app.route("/")
def home():
    return render_template("index.html", soil_fields=SOIL_FIELDS, climate_fields=CLIMATE_FIELDS)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) if request.is_json else request.form
    try:
        values = parse_inputs(data or {})
        recommendations = top_crops(values)
    except ValueError as e:
        if request.is_json:
            return jsonify({"success": False, "error": str(e)}), 400
        return render_template("result.html", error=str(e)), 400
    except Exception as e:
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