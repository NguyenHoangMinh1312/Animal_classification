"""
Read the dataset
"""
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import torch

class AnimalDataset(Dataset):
    def __init__(self, root, mean = None, std = None, train = True, ratio= 0.8):
        super().__init__()
        self.root = root
        self.mean = mean
        self.std = std
        self.img_paths = []
        self.labels = []
        self.classes = os.listdir(root)
        self.classes.remove("README.txt")

        for class_name in self.classes:
            tmp_img_paths = []
            tmp_labels = []
            for img_name in os.listdir(os.path.join(root, class_name)):
                tmp_img_paths.append(os.path.join(root, class_name, img_name))
                tmp_labels.append(self.classes.index(class_name))

            split_idx = int(len(tmp_labels) * ratio)
            if train:
                self.img_paths.extend(tmp_img_paths[:split_idx])
                self.labels.extend(tmp_labels[:split_idx])
            else:
                self.img_paths.extend(tmp_img_paths[split_idx:])
                self.labels.extend(tmp_labels[split_idx:])
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, id):
        img_path = self.img_paths[id]
        label = self.labels[id]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255
        if self.mean is not None and self.std is not None:
            img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()

        label = torch.tensor(label, dtype = torch.long)
        return img, label


    




            





