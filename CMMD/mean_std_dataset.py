import numpy as np
import pandas as pd

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

import cv2

from tqdm import tqdm
import os
import matplotlib.pyplot as plt


####### PARAMS
data_path1 = r"D:\IAAA_CMMD\manifest-1616439774456\test_simpleStik\test_meta/"
data_path= r"D:\IAAA_CMMD\manifest-1616439774456/"
df = pd.read_csv(data_path1 + 'out_merg.csv')
df.head()
df['classification'] = df['classification'].apply(lambda x: 0 if x == 'Benign' else 1)

device      = torch.device('cpu') 
num_workers = 0
image_size  = 512 
batch_size  = 8

class CMMD(Dataset):
    
    def __init__(self, 
                 data, 
                 directory, 
                 transform = None):
        self.data      = data
        self.directory = directory
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        # import
        path  = os.path.join(self.directory, self.data.iloc[idx]['Img_name'])
        image = cv2.imread(path, cv2.COLOR_BGR2RGB)
            
        # augmentations
        if self.transform is not None:
            image = self.transform(image = image)['image']
        
        return image
    

augs = A.Compose([A.Resize(height = image_size, 
                           width  = image_size),
                  A.Normalize(mean = (0.0624, 0.0624, 0.0624),
                              std  = (0.1319, 0.1319, 0.1319)),
                  ToTensorV2()])




# dataset
image_dataset = CMMD(data      = df, 
                         directory = data_path + 'png_test_2/',
                         transform = augs)

# data loader
image_loader = DataLoader(image_dataset, 
                          batch_size  = batch_size, 
                          shuffle     = False, 
                          num_workers = num_workers,
                          pin_memory  = True)

a=iter(image_loader)
a1=next(a)

# display images
for batch_idx, inputs in enumerate(image_loader):
    fig = plt.figure(figsize = (14, 7))
    for i in range(8):
        ax = fig.add_subplot(2, 4, i + 1, xticks = [], yticks = [])     
        plt.imshow(inputs[i].numpy().transpose(1, 2, 0))
    break



# placeholders
psum    = torch.tensor([0.0, 0.0, 0.0])
psum_sq = torch.tensor([0.0, 0.0, 0.0])

# loop through images
for inputs in tqdm(image_loader):
    psum    += inputs.sum(axis        = [0, 2, 3])
    psum_sq += (inputs ** 2).sum(axis = [0, 2, 3])
    
    
    
count = len(df) * image_size * image_size

# mean and std
total_mean = psum / count
total_var  = (psum_sq / count) - (total_mean ** 2)
total_std  = torch.sqrt(total_var)

# output
print('mean: '  + str(total_mean))
print('std:  '  + str(total_std))
