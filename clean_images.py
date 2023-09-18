#######Clean the image dataset#########
# %% clean image dataset
from PIL import Image
import os

def clean_image_data(input_folder, output_folder, final_size):
 
    # List all files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    
    for image_file in image_files:
        input_path = os.path.join(input_folder, image_file)
        
        # Open the image
        with Image.open(input_path) as image:
            # Check and convert to RGB, to make consistent channels
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize the image
            size = image.size
            ratio = float(final_size) / max(size)
            new_image_size = tuple([int(x * ratio) for x in size])
            resized_image = image.resize(new_image_size, Image.ANTIALIAS)
            
            # Create a new image with the desired final_size
            new_im = Image.new('RGB', (final_size, final_size))
            new_im.paste(resized_image, ((final_size - new_image_size[0]) // 2, (final_size - new_image_size[1]) // 2))
            
            # Save the cleaned image to the output folder
            output_path = os.path.join(output_folder, image_file)
            new_im.save(output_path)
    
    print("Image preprocessing complete.")


if __name__ == '__main__':
    input_folder = 'images/'
    output_folder = 'cleaned_images/'
    final_size = 224  # to match the size of resnet Imagnet package for better training

    clean_image_data(input_folder, output_folder, final_size)

# %%
