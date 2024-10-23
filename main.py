from fastapi import FastAPI, File, UploadFile
import pandas as pd
import numpy as np
import pickle
import os

app = FastAPI()

with open(os.path.join(os.getcwd(), "model_checkpoint", "lr_model.pkl"), 'rb') as file:
    model = pickle.load(file)

@app.get("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(content.decode('utf-8'))

    features = np.asanyarray(df[['TV', 'Radio']])

    predictions = model.predict(features)

    return {"predictions": predictions.tolist()}

