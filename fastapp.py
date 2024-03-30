from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List
from io import BytesIO
import numpy as np
import scipy.io
from tensorflow.keras.models import load_model
from src.visualization import plot_ecg

app = FastAPI()

# Mount static files (css, js, etc.)

# Templates
templates = Jinja2Templates(directory="templates")

# Load model
model_path = 'models/weights-best.hdf5'
model = load_model(model_path)

# Class labels
classes = ['Normal', 'Atrial Fibrillation', 'Other', 'Noise']

# Data preprocessing function
def preprocess_ecg(uploaded_ecg):
    FS = 300
    maxlen = 30 * FS

    mat = scipy.io.loadmat(BytesIO(uploaded_ecg.file.read()))
    mat = mat["val"][0]

    uploaded_ecg = np.array([mat])

    X = np.zeros((1, maxlen))
    uploaded_ecg = np.nan_to_num(uploaded_ecg)  # removing NaNs and Infs
    uploaded_ecg = uploaded_ecg[0, 0:maxlen]
    uploaded_ecg = uploaded_ecg - np.mean(uploaded_ecg)
    uploaded_ecg = uploaded_ecg / np.std(uploaded_ecg)
    X[0, :len(uploaded_ecg)] = uploaded_ecg.T  # padding sequence
    uploaded_ecg = X
    uploaded_ecg = np.expand_dims(uploaded_ecg, axis=2)
    return uploaded_ecg

# Prediction function
def predict_ecg(ecg):
    prob = model.predict(ecg)
    ann = np.argmax(prob)
    return classes[ann], prob[0]

# Route for the main page
@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route for handling file upload and displaying results
@app.post("/upload", response_class=HTMLResponse)
async def upload_files(request: Request, files: List[UploadFile] = File(...)):
    ecg = preprocess_ecg(files[0])
    pred_class, pred_prob = predict_ecg(ecg)
    return templates.TemplateResponse("result.html", {"request": request, "class": pred_class, "probability": pred_prob})



