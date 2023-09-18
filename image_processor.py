
# %% Fine tune ResNet50 pretrained model for learning 
import torch
import pandas as pd
import json
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings('ignore')
from torchvision.io import read_image

# Load the previously trained model weights
trained_model_state_dict = torch.load('/model_evaluation/model_20230907_150639/weights/resnet_epoch_9_val_loss_1.4662_val_acc_55.63.pth')
# Load the pretrained ResNet-50 model
feature_resnet = models.resnet50(pretrained=False)
# rebuild the training model and load weights from pretrained parameters
feature_resnet.fc = nn.Sequential(
    nn.Linear(2048, 1000),
    nn.ReLU(),
    nn.Linear(1000, 13)
    )
feature_resnet.load_state_dict(trained_model_state_dict)
# drop the last two layers -- transfer from classification to feature extraction model with 1000 neurons
feature_resnet.fc = feature_resnet.fc[0]


#%% Extract feature using the extraction model to build vectors for FAISS database
# define transformation similar to traning model
class Rescale:
    #Rescale the image tensor values from uint8 [0, 255] to float32 [0.0, 1.0]
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.float() / 255.0

transform = transforms.Compose([
    Rescale(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# %%load traning set paths and capture the feature vectors
sample_path = pd.read_csv('training_sample_paths.csv')
sample_path['Sample_Path'] = sample_path['Sample_Path'].str.replace('\\','/')
embedding_dict = {}

for img_path in sample_path['Sample_Path']:
    # extract feature using 'feature_extract' model
    image = read_image(img_path)
    image = transform(image)
    image = image.unsqueeze(0)
    output_features = feature_resnet(image)

    # transfer the tensor type to list and save in json format (1-D)
    embedding_vector = output_features.squeeze().tolist()
    key = img_path.split('/')[-1].split('.')[0]
    embedding_dict[key] = embedding_vector
    
    # dump key, value into json file
    with open('image_embeddings.json','w') as f:
        json.dump(embedding_dict, f)