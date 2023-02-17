
import numpy as np
import torch
from torchvision import transforms
import random
from pathlib import Path
#from typing import Tuple, Dict, List
import pydicom
import glob
import sys
import os
from torch.utils.data import Dataset,DataLoader
from pydicom.pixel_data_handlers.util import apply_voi_lut
import matplotlib.pyplot as plt
NUM_WORKERS=0
PIN_MEMORY=True    



transform_1 =  transforms.Compose([transforms.ToTensor()])

def read_xray(path, voi_lut = True, fix_monochrome = True):
    dicom = pydicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
               
    # depending on this value, MRI may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME2":
        data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
        
    return data



class Dataset_(Dataset):
    def __init__(self, image_dir,transform=None):
        self.image_dir = image_dir
       # self.genes=path_genes_data
        self.images = list(image_dir.glob("*/*/*/*.dcm"))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        ## reading image ###
        img_path = os.path.join(self.image_dir, self.images[index])
        image = read_xray(img_path)
        #plt.figure(figsize = (12,12))
        #plt.imshow(image, 'gray')
        image_name=self.images[index]
        if self.transform is not None:
            a = self.transform(image=image)
            image = a['image']
            #image=np.transpose(image, (2, 0, 1))
            
        return image,self.images[index]



 
def Data_Loader( test_dir,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    
    load_data = Dataset_( image_dir=test_dir,transform=transform_1)

    data_loader = DataLoader(load_data,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    
    return data_loader


data_path = Path(r"E:\IAAA_CMMD\manifest-1616439774456")
images_folder = data_path / "CMMD"
loader=Data_Loader(images_folder,4)
a=iter(loader)
a1=next(a)
