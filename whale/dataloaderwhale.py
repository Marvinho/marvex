# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 09:12:16 2018
@author: MRVN
"""

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
from torchvision.transforms import ToPILImage, Resize, RandomHorizontalFlip, RandomRotation, ToTensor, Compose
from PIL import Image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode
"""
labels_frame = pd.read_csv('./train.csv')
root_dir = "./train"
n = 65
img_name = labels_frame.iloc[n, 0]
print(img_name)
matches = labels_frame.iloc[n, 1].as_matrix()
print(matches)
#matches = matches.astype('str').reshape(-1, 2)
print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(matches.shape))
print('First 4 Landmarks: {}'.format(matches[:4]))
"""
class HumpbackWhaleDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.arr = set()
        for i in range(len(self.labels_frame)):
            self.arr.add(self.labels_frame.iloc[i,1])
        self.arr = list(self.arr)

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.labels_frame.iloc[idx, 0])
        image = Image.open(img_name)
        image = image.convert("RGB")
        image = np.array(image)
        labels = self.arr.index(self.labels_frame.iloc[idx, 1])
        #print(labels)
        labels = torch.FloatTensor([labels])
        #labels = labels.tolist()
        #sample = {'image': image, 'labels': labels}

        if self.transform:
            image = self.transform(image)

        return image, labels
#img_size = (224, 224)
#img = image.convert('RGB')
#whale_dataset = HumpbackWhaleDataset(csv_file = './train.csv', root_dir= "./train")

#fig = plt.figure()
"""
for i in range(10):
    sample = whale_dataset[i]
    print(i, sample['image'].shape, sample['labels'])
"""
#tsfm = Transform(params)
#transformed_sample = tsfm(sample)

#composed = Compose([ToPILImage(), Resize(img_size), RandomHorizontalFlip(), RandomRotation(degrees = 30, center = None), ToTensor()])

# Apply each of the above transforms on sample.
#fig = plt.figure()
#sample = whale_dataset[65]
#print(sample)
"""
for i, tsfrm in enumerate([composed]):
    transformed_sample = tsfrm(sample["image"])
    print(transformed_sample)
    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(sample["labels"])
    #plt.imshow(transformed_sample)
plt.show()
"""
#transformed_dataset = HumpbackWhaleDataset(csv_file = './train.csv', root_dir= "./train", transform = composed )

