import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import wandb
from model import FlexibleCNN
import random
import numpy as np
from tqdm import tqdm

class iNaturalistDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.classes.remove(".DS_Store")
        #print(self.classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name == ".DS_Store":
                    continue
                self.images.append(os.path.join(class_dir, img_name))
                self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        
        # Set random seeds for reproducibility
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        
        # Data transforms
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load dataset
        dataset = iNaturalistDataset(r"inaturalist_12K\train", transform=train_transform)
        
        # Split into train and validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Apply validation transform to validation dataset
        val_dataset.dataset.transform = val_transform
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        
        # Initialize model
        model = FlexibleCNN(
            num_classes=10,
            conv_filters=config.conv_filters,
            activation=config.activation,
            dense_neurons=config.dense_neurons,
            dropout_rate=config.dropout_rate,
            use_batch_norm=config.use_batch_norm
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"Using device: {device}")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # Training loop
        best_val_acc = 0.0
        for epoch in tqdm(range(config.epochs)):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            print(f"Epoch [{epoch+1}/{config.epochs}], "
                  f"Train Loss: {train_loss / len(train_loader):.4f}, "
                  f"Train Acc: {100. * train_correct / train_total:.2f}%, "
                  f"Val Loss: {val_loss / len(val_loader):.4f}, "
                  f"Val Acc: {100. * val_correct / val_total:.2f}%")
            # Log metrics
            wandb.log({
                'train_loss': train_loss / len(train_loader),
                'train_acc': 100. * train_correct / train_total,
                'val_loss': val_loss / len(val_loader),
                'val_acc': 100. * val_correct / val_total,
                'epoch': epoch
            })
            
            # Save best model
            val_acc = 100. * val_correct / val_total
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'best_model.pth')

if __name__ == '__main__':
    # Define sweep configuration
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'val_acc',
            'goal': 'maximize'
        },
        'parameters': {
            'seed': {
                'values': [42]
            },
            'batch_size': {
                'values': [8]
            },
            'learning_rate': {
                'values': [0.001, 0.0001]
            },
            'epochs': {
                'value': 20
            },
            'conv_filters': {
                'values': [
                    [32, 32, 32, 32, 32],  # Same filters
                    [32, 64, 128, 256, 512],  # Doubling
                    [512, 256, 128, 64, 32]  # Halving
                ]
            },
            'activation': {
                'values': ['relu', 'gelu', 'silu', 'mish']
            },
            'dense_neurons': {
                'values': [256, 512, 1024]
            },
            'dropout_rate': {
                'values': [0.0, 0.2, 0.3]
            },
            'use_batch_norm': {
                'values': [True, False]
            }
        }
    }
    
    sweep_id = wandb.sweep(sweep_config, project="DL_A2")
    wandb.agent(sweep_id, function=train, count=20) 