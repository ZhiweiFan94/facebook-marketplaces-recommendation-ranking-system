# Facebook Marketplace Recommendation Ranking System

## load and clean tabular and image dataset
### For tabular datasets: 
Create 'clean_tabular_data.py' to clean 'Product.csv' and 'images.csv'. The program includes three blocks
- Loading the files from AWS EC2
- Check null values and extract categories of all products, classify each catagory with list of customed index, build the encoder and decoder
- merge products and images files where the ids are matched, save them into 'traning_data.csv'

### For image dataset:
Create 'clean_images.py' to standardize the images on their sizes, channels for being consistent. The program contains the following functions:
- Loading the images using 'with Image.open' 
- Resize the image with same size of 512 pixels
- Convert all imgages to RGB format, viz, channels=3
- save images into the new folder


 