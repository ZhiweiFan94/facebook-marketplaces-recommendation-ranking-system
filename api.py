#%%
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from torchvision.io import read_image
import torchvision.models as models
import torchvision.transforms as transforms
import json
import faiss
import uvicorn
import pickle
import os
import torch
import torch.nn as nn
import numpy as np

# FeatureExtractor class definition
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        # Loading ResNet50 pre-trained model and adding custom layers
        self.main = models.resnet50(weights=None)
        self.main.fc = nn.Sequential(
            nn.Linear(2048, 1000),
            nn.ReLU(),
            nn.Linear(1000, 13)
        )
        # Load model weights
        self.main.load_state_dict(torch.load('resnet_epoch_9_val_loss_1.4662_val_acc_55.63.pth'))
        self.main.fc = self.main.fc[0]  # Drop last two layers

    def forward(self, image):
        return self.main(image)

    def predict(self, image):
        with torch.no_grad():
            return self.forward(image) 

# Load decoder to transfer seraching images 
with open('image_decoder.pkl','rb') as f:
    decoder = pickle.load(f)

#%%
# Loading FAISS model
try:
    with open('image_embeddings.json', 'r') as f:
        embeddings_dict = json.load(f)
    database_vectors = np.array(list(embeddings_dict.values())).astype('float32')
    dimension = 1000
    index = faiss.IndexFlatL2(dimension)
    index.add(database_vectors)
except:
    raise OSError("No FAISS model found. Check your file paths.")

# Transformations
transform = transforms.Compose([
    transforms.Lambda(lambda tensor: tensor.float() / 255.0),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# FastAPI app
app = FastAPI()
print("Starting server")
feature_extractor = FeatureExtractor()

@app.get('/healthcheck')
def healthcheck():
    return {"message": "API is running"}

@app.post('/predict/feature_embedding')
def predict_image(image: UploadFile = File(...)):
    # Save the uploaded file to a temporary location when use read_iamge
    temp_file = "temp_image.jpg"
    with open(temp_file, "wb") as buffer:
        buffer.write(image.file.read())
    # Use read_image to read the image from the saved location
    image_tensor = read_image(temp_file)
    # Delete the temporary file after reading
    os.remove(temp_file)

    # Implement feature extraction
    image_tensor = transform(image_tensor).unsqueeze(0)
    features = feature_extractor.predict(image_tensor).squeeze().tolist()
    return JSONResponse(content={"features": features})
 

@app.post('/predict/similar_images')
def predict_combined(image: UploadFile = File(...), text: str = Form(...)):
    # Extract feature as above 
    temp_file = "temp_image.jpg"
    with open(temp_file, "wb") as buffer:
        buffer.write(image.file.read())
    image_tensor = read_image(temp_file)
    os.remove(temp_file)
    image_tensor = transform(image_tensor).unsqueeze(0)
    vectors = feature_extractor.predict(image_tensor)
    features = np.array(vectors).astype('float32').reshape(1,1000)

    # FAISS return 5 similar image indices
    k = 5
    distances, indices = index.search(features, k)
    similar_indices = indices[0].tolist()
    return JSONResponse(content={"similar_index": similar_indices})

if __name__ == '__main__':
    uvicorn.run("api:app", host="0.0.0.0", port=8080)