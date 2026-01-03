from fastapi import FastAPI,File,UploadFile
from fastapi.responses import JSONResponse
import os
import cv2
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import onnxruntime as ort
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np

from preprocessing.preprocessing import preprocess_image
from model.inference import ONNXModel

app = FastAPI()

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "pituitary",
    "notumour"
]

model = ONNXModel(
    model_path="artifacts/Resnet50_brain_tumour_classifier_merged.onnx",
    class_names=CLASS_NAMES
)

@app.get("/")
def index():
    return {'key':'value'}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    input_tensor = preprocess_image(image_bytes)
    result = model.predict(input_tensor)

    return JSONResponse(content=result)