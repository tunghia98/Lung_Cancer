import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json

class CTScanDataset(Dataset):
    def __init__(self, data_dir, split, transform=None):
        self.data_dir = data_dir
        self.split_info_file = os.path.join(self.data_dir, 'split_info.json')
        self.split = split
        self.transform = transform
        self.images, self.labels = self.get_images_and_labels()        

    def get_images_and_labels(self):
        with open(self.split_info_file, 'r') as fp:
            self.split_info = json.load(fp)
            self.split_info = self.split_info[self.split]
        
        label_names = list(self.split_info.keys())
        images, labels = [], []
        for label, image_paths in self.split_info.items():
            for image_path in image_paths:
                labels.append(label_names.index(label))
                images.append(image_path)
        return images, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(), 
            transforms.RandomRotation(15), 
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
        ]),
        "test": transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(), 
            transforms.RandomRotation(15), 
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
        ])
    }

def load_data(data_dir, batch_size=32, split='train'):
    dataset = CTScanDataset(data_dir, split, transform = transforms[split])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
