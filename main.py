from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

# Define your input data format
class Features(BaseModel):
    feature1: float
    feature2: float

model_path = "/var/data/regression_model.pkl"
model = joblib.load(model_path)

@app.post("/predict")
def predict(data: Features):
    # Turn the input into the right shape for the model
    X = np.array([[data.feature1, data.feature2]])
    prediction = model.predict(X)[0]
    return {"prediction": prediction}

