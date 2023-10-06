import os
import torch
import time
import warnings
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import pandas as pd
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings('ignore')


class ImagesDataset(Dataset):
    """
    Initialize a dataset class for training
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


def split_and_save_dataset(dataset, train_ratio=0.8, validation_ratio=0.1, batch_num=128, save_path='training_sample_paths.csv'):
    """
    Split the dataset into training, validation, and test sets, and save selected training samples for feature extraction.

    Parameters:
    - dataset: The dataset to be split.
    - train_ratio: The ratio of the dataset to be used for training.
    - validation_ratio: The ratio of the dataset to be used for validation.
    - batch_num: Batch size for the DataLoader.
    - save_path: Path to save the training sample paths.

    Returns:
    - train_dataloader, val_dataloader, test_dataloader: DataLoaders for training, validation, and testing.
    """
    train_size = int(train_ratio * len(dataset))
    val_size = int(validation_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_num, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_num, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_num, shuffle=False)

    # Save selected training samples for feature extraction
    train_paths = [dataset.get_image_path(i) for i in train_dataset.indices]
    df_train_paths = pd.DataFrame(train_paths, columns=['Sample_Path'])
    df_train_paths.to_csv(save_path, index=False)

    return train_dataloader, val_dataloader, test_dataloader


class ResNetTrainer:
    """
    Train, validate and save the model to fit the dataset

    Methods:
    - train(): The transfer learning based on Resnet50 model.
    - validate(): The validation accuracy of the model.
    - save_model(): save the weights of model into .pth format.
    """
    def __init__(self, train_dataloader, val_dataloader, num_epochs):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_epochs = num_epochs
        self.pretrained_resnet = self.initialize_model()
        self.optimizer = self.set_optimizer()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        self.writer = SummaryWriter()

    def initialize_model(self):
        model = models.resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        unfreeze_layers = ['layer4', 'fc']
        for name, layer in model.named_children():
            if name in unfreeze_layers:
                for param in layer.parameters():
                    param.requires_grad = True
        model.fc = nn.Sequential(
            nn.Linear(2048, 1000),
            nn.ReLU(),
            nn.Linear(1000, 13)
        )
        return model

    def set_optimizer(self):
        learning_rate = 0.001
        optimizer = optim.SGD([
            {'params': self.pretrained_resnet.layer4.parameters(), 'lr': learning_rate},
            {'params': self.pretrained_resnet.fc[2].parameters(), 'lr': learning_rate * 10}
        ], lr=learning_rate, momentum=0.9, weight_decay=0.001)
        return optimizer

    def train(self):
        batch_idx = 0
        for epoch in range(self.num_epochs):
            for inputs, labels in self.train_dataloader:
                outputs = self.pretrained_resnet(inputs)
                loss = F.cross_entropy(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print(f'The current loss is {loss}')
                self.writer.add_scalar('loss', loss.item(), batch_idx)
                batch_idx += 1

            val_loss, val_accuracy = self.validate()
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Validation Accuracy: {val_accuracy:.2f}%')
            print(f'Average Validation Loss: {val_loss:.2f}')
            self.save_model(epoch, val_loss, val_accuracy)
            self.scheduler.step()

    def validate(self):
        self.pretrained_resnet.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.val_dataloader:
                outputs = self.pretrained_resnet(inputs)
                val_loss += F.cross_entropy(outputs, labels).item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        average_val_loss = val_loss / len(self.val_dataloader)
        val_accuracy = 100 * correct / total
        return average_val_loss, val_accuracy

    def save_model(self, epoch, val_loss, val_accuracy):
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        model_folder = os.path.join('model_evaluation', f'model_{timestamp}')
        os.makedirs(model_folder, exist_ok=True)
        weights_folder = os.path.join(model_folder, 'weights')
        os.makedirs(weights_folder, exist_ok=True)
        model_name = f'resnet_epoch_{epoch}_val_loss_{val_loss:.4f}_val_acc_{val_accuracy:.2f}.pth'
        model_path = os.path.join(weights_folder, model_name)
        torch.save(self.pretrained_resnet.state_dict(), model_path)
        metrics_filename = os.path.join(model_folder, 'metrics.txt')
        with open(metrics_filename, 'w') as metrics_file:
            metrics_file.write(f'Epoch: {epoch}\n')
            metrics_file.write(f'Validation Loss: {val_loss:.4f}\n')
            metrics_file.write(f'Validation Accuracy: {val_accuracy:.2f}%\n')

#load and split dataset
transform = transforms.Compose([
    lambda tensor: tensor.float()/255.0, #transform into [0,1] range from [0,255]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalized std and mean from ImageNet packages for fine-tune weights
])
dataset = ImagesDataset(transform=transform)
train_dataloader, val_dataloader, test_dataloader = split_and_save_dataset(dataset)

#%%create instance and train the model
trainer = ResNetTrainer(train_dataloader, val_dataloader, num_epochs=10)
trainer.train()