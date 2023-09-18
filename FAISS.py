#%% import libraries
import json
import numpy as np
import faiss
import pandas as pd
from PIL import Image

#%% load dict: image idx and feature vector from json
with open('image_embeddings.json', 'r') as f:
    embeddings_dict = json.load(f)

# extract vectors and convert them to a numpy matrix
database_vectors = np.array(list(embeddings_dict.values())).astype('float32')
database_keys = np.array(list(embeddings_dict.keys())).astype('str')

# define dimential and searching method
dimension = 1000  # number of features
index = faiss.IndexFlatL2(dimension)

# load feature vectors to the index
index.add(database_vectors)

#%% select a query vector 
q_idx = 0
query_vector = database_vectors[q_idx,:].reshape(1,1000) # just pick a random one from database
query_img = database_keys[q_idx]
print(query_img)

# searching with faiss
k = 5 # find number of similar images as output
# 'indices' will contain indices of the nearest vectors in your database
# 'distances' will contain L2 distances to these vectors
distances, indices = index.search(query_vector, k)

faiss_idx=[]
faiss_name=[]
print('Top closest images in the database:')
for i, idx in enumerate(indices[0]):
    faiss_idx.append(idx)
    faiss_name.append(database_keys[idx])
    print(f'Rank {i + 1}: Image at index {idx} with L2 distance: {distances[0][i]}')
    print(f'the name of the image is {database_keys[idx]}')

# %%
# Define the folder and extension
folder_path = 'cleaned_images'
extension = '.jpg'

# List to hold all the images (starting with the query image)
all_images = [Image.open(folder_path + '/' + query_img + extension)]
for idx in indices[0]:
    all_images.append(Image.open(folder_path + '/' + database_keys[idx] + extension))

# Assuming all images are the same size, get width and height of one image
img_width, img_height = all_images[0].size

# Create an empty image to hold the 2x3 grid
grid_width = 3 * img_width
grid_height = 2 * img_height
grid_image = Image.new('RGB', (grid_width, grid_height))

# Place images on the grid
for i in range(2):  # 2 rows
    for j in range(3):  # 3 columns
        img = all_images[i*3 + j]
        grid_image.paste(img, (j * img_width, i * img_height))

# Show the combined image that similar to the input one
grid_image.show()