# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 10:34:43 2023

@author: omnia
"""
import cv2
import torch
import torchvision.transforms as transforms

def cv2_read(img_path):
    
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_three = cv2.merge([gray,gray,gray])
    img__ = cv2.resize(gray_three, (96,96), interpolation = cv2.INTER_AREA)
    img_tensor = transforms.functional.to_tensor(img__)
    #img_t = torch.tensor(img__, dtype=torch.float, device='cpu')

    images = img_tensor.unsqueeze(0)    
    return images

path = "E:\IAAA_CMMD\manifest-1616439774456\png_test_2\D1-0001_1-1.png"
img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

#get dimensions of image
dimensions = img.shape
# height, width, number of channels in image
height = img.shape[0]
width = img.shape[1]
channels = img.shape[2]
height, width = img.shape[:2]

print(f" Input image Height: {height}, Width: {width} and Channel: {channels} ")
im1 = cv2_read(path)


### Test if the image diminsion work as an input to CNN
import timm
def load_model():

    model =  timm.create_model('convnext_base', pretrained=True,num_classes=2) 


    # Disable gradients on all model parameters to freeze the weights
    for param in model.parameters():
        param.requires_grad = False

    for param in model.head.parameters():
        param.requires_grad = False

    # Unfreeze the last stage
    for param in model.stages[3].parameters():
        param.requires_grad = True
    
    return model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_3= load_model()
model_3 = model_3.to(device=DEVICE,dtype=torch.float)

out = model_3(im1)
