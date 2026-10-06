import joblib
import pandas as pd

model = joblib.load("ml/crop_recommendation_model.pkl")
scaler = joblib.load("ml/crop_recommendation_scaler.pkl")

def predict_crops(farmer_data):

    sample_farmer = pd.DataFrame([farmer_data])

    sample_scaled = scaler.transform(sample_farmer)

    probabilities = model.predict_proba(sample_scaled)[0]

    crop_names = model.classes_

    top_5_indices = probabilities.argsort()[-5:][::-1]

    recommendations = []

    for index in top_5_indices:
        recommendations.append({
            "crop": crop_names[index],
            "confidence": round(probabilities[index] * 100, 2)
        })

    return recommendations


sample_farmer = {
    "N": 90,
    "Fe": 25,
    "Mn": 1000,
    "S": 10,
    "Mg": 75,
    "C": 250,
    "P": 42,
    "K": 43,
    "temperature": 21,
    "humidity": 82,
    "ph": 6.5,
    "rainfall": 203
}

recommendations = predict_crops(sample_farmer)

print("\nTop 5 crop recommendations:")

for recommendation in recommendations:
    print(
        f"{recommendation['crop']}: "
        f"{recommendation['confidence']}%"
    )