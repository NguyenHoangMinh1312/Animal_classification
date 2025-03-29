from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import torch

"""
This class is used to load the dataset
"""
class AnimalDataset(Dataset):
    def __init__(self, root, mean = None, std = None, train = True, size = 224):
        self.image_paths = []
        self.labels = []
        self.categories = os.listdir(root)
        self.mean, self.std = mean, std
        self.size = size
        
        for id, category in enumerate(self.categories):
            tmp_image_paths = []
            tmp_labels = []
            for images in os.listdir(os.path.join(root, category)):
                tmp_image_paths.append(os.path.join(root, category, images))
                tmp_labels.append(id)

            split_idx = int(0.8 * len(tmp_image_paths))
            if train:
                self.image_paths.extend(tmp_image_paths[:split_idx])
                self.labels.extend(tmp_labels[:split_idx])
            else:
                self.image_paths.extend(tmp_image_paths[split_idx:])
                self.labels.extend(tmp_labels[split_idx:])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.size, self.size))
        image = image / 255     # Normalize the image
        if self.mean is not None and self.std is not None:
            image = (image - self.mean) / self.std   # Standardize the image
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).float()     # Convert the image to tensor
        
        label = torch.tensor(self.labels[index], dtype = torch.long)
        return image, label

