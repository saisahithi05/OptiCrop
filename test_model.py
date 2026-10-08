import joblib
import pandas as pd

# Load model
model = joblib.load("../model/crop_model.pkl")

# Load label encoder
encoder = joblib.load("../model/label_encoder.pkl")

# Sample input
sample = pd.DataFrame([{
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.87,
    "humidity": 82.00,
    "ph": 6.50,
    "rainfall": 202.93
}])

# Predict
prediction = model.predict(sample)

# Convert numeric prediction back to crop name
crop = encoder.inverse_transform(prediction)

print("Recommended Crop:", crop[0])