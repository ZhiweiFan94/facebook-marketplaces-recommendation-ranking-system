import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import os
import time
import pandas as pd
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')


class ImagesDataset(Dataset):
    """
    dataset composition
    """
    def __init__(self,transform=None, target_transform=None):
        self.img_labels = pd.read_csv('training_data.csv',lineterminator='\n')
        self.img_dir = 'cleaned_images'
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, idx):
        # image names is listed in column named 'id_x'
        img_path = os.path.join(self.img_dir, self.img_labels['id_x'].iloc[idx]+'.jpg')
        # read_image will return torch.uint8 type and value in range 0-255 which should processed later.
        image = read_image(img_path)
        # position '-1' corresponds to encoded category number
        label = self.img_labels['encoded_category'].iloc[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    

    def __len__(self):
        return len(self.img_labels)
    
    #used for record files selected for training 
    def get_image_path(self, idx):
        return os.path.join(self.img_dir, self.img_labels['id_x'].iloc[idx] + '.jpg')


class Rescale:
    """
    Rescale the image tensor values from uint8 [0, 255] to float32 [0.0, 1.0]
    """
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.float() / 255.0


transform = transforms.Compose([
    Rescale(), #transform into [0,1] range from [0,255]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalized std and mean from ImageNet packages for fine-tune weights
])

dataset = ImagesDataset(transform=transform)

# %%
# Split the dataset into training and validation sets
train_ratio = 0.8  # Adjust the ratio as needed
validation_ratio = 0.1
train_size = int(train_ratio * len(dataset))
val_size = int(validation_ratio * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
batch_num = 128
train_dataloader = DataLoader(train_dataset, batch_size=batch_num, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_num, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_num, shuffle=False)

#save selected training samples for feature extraction
train_paths = [dataset.get_image_path(i) for i in train_dataset.indices]
df_train_paths = pd.DataFrame(train_paths, columns=['Sample_Path'])
df_train_paths.to_csv('training_sample_paths.csv', index=False)


def train(models, num_epochs=10):
    # fine tune the resnet 50 model
    pretrained_resnet = models.resnet50(pretrained=True)

    # Freeze all layers initially
    for param in pretrained_resnet.parameters():
        param.requires_grad = False
    # Unfreeze the last 2 layers
    unfreeze_layers = ['layer4', 'fc']
    for name, layer in pretrained_resnet.named_children():
        if name in unfreeze_layers:
            for param in layer.parameters():
                param.requires_grad = True
    # There are 13 categories in our image database
    number_category = 13
    # Modify the output layer to match the number of classes
    pretrained_resnet.fc = nn.Sequential(
        nn.Linear(2048, 1000),
        nn.ReLU(),
        nn.Linear(1000, number_category)
        )
    
    # Define different learning rates for different layers
    learning_rate = 0.001
    # use momentum to accelrate steep convergence and out from local minima; weight_decay to add penalty to large weight for overfit 
    optimizer = optim.SGD([
        {'params': pretrained_resnet.layer4.parameters(), 'lr': learning_rate},  
        {'params': pretrained_resnet.fc[2].parameters(), 'lr': learning_rate * 10}  
    ], lr=learning_rate, momentum=0.9, weight_decay=0.001) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # record loss every training batch
    writer = SummaryWriter() #from lib tensorboard
    batch_idx = 0

    # Training loop
    for epoch in range(num_epochs):
        for inputs, labels in train_dataloader:
            outputs = pretrained_resnet(inputs)
            loss = F.cross_entropy(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            print(loss.item())
            optimizer.step()
            writer.add_scalar('loss',loss.item(),batch_idx)
            batch_idx +=1
            print(batch_idx/len(train_dataloader))

        # Validation loop
        pretrained_resnet.eval()  # Set the model to evaluation mode (turns off dropout, batch normalization, etc.)
        with torch.no_grad():  # Context manager to disable gradient computation
            val_loss = 0.0  # Initialize the validation loss
            correct = 0  # Initialize the number of correctly predicted samples
            total = 0  # Initialize the total number of samples
            for inputs, labels in val_dataloader:  # Loop through validation data
                outputs = pretrained_resnet(inputs)  # Get model predictions for the inputs
                val_loss += F.cross_entropy(outputs, labels).item()  # Calculate the validation loss
                _, predicted = outputs.max(1)  # Get the class predictions by selecting the maximum output
                total += labels.size(0)  # Increment the total count by the batch size
                correct += predicted.eq(labels).sum().item()  # Count how many predictions are correct
                print(predicted)

        average_val_loss = val_loss / len(val_dataloader)  # Calculate the average validation loss
        val_accuracy = 100 * correct / total

        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {val_accuracy:.2f}%')
        print(f'Average Validation Loss: {average_val_loss:.2f}')

        # Save model weights in timestamped folders
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        model_folder = os.path.join('model_evaluation', f'model_{timestamp}')
        os.makedirs(model_folder)

        # Save model weights in the "weights" folder
        weights_folder = os.path.join(model_folder, 'weights')
        os.makedirs(weights_folder)

        # Save model weights with epoch information
        model_name = f'resnet_epoch_{epoch}_val_loss_{average_val_loss:.4f}_val_acc_{val_accuracy:.2f}.pth'
        model_path = os.path.join(weights_folder, model_name)
        torch.save(pretrained_resnet.state_dict(), model_path)

        # Save metrics in the model folder
        metrics_filename = os.path.join(model_folder, 'metrics.txt')
        with open(metrics_filename, 'w') as metrics_file:
            metrics_file.write(f'Epoch: {epoch}\n')
            metrics_file.write(f'Validation Loss: {average_val_loss:.4f}\n')
            metrics_file.write(f'Validation Accuracy: {val_accuracy:.2f}%\n')

        scheduler.step() #take scheduler to tune the convergence step size


train(models)
