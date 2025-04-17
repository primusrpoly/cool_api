from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import requests
import os

app = FastAPI()

# Replace with your actual model file URL
MODEL_URL = "https://drive.google.com/file/d/1KNFGjWwJPkITnHi2Yqb3PVBrh3FKcf7O/view?usp=drive_link"
MODEL_PATH = "abr_model.pkl"

# Download model if it's not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from URL...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)

# Load the model
model = joblib.load(MODEL_PATH)

# Define input schema
class Features(BaseModel):
    feature1: float
    feature2: float

@app.post("/predict")
def predict(data: Features):
    prediction = model.predict([[data.feature1, data.feature2]])[0]
    return {"prediction": prediction}
