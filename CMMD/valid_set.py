# this preproccessing for cases that have 4 images per patients, and 2 classification (Bening and Malignant) 

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 15:15:24 2023

@author: omnia
"""



import glob
import os
import pandas as pd

import SimpleITK as sitk
import numpy as np
valid = pd.read_csv(r"E:/IAAA_CMMD/manifest-1616439774456/valid.csv")
valid.columns
valid = valid.drop(columns=['Number of Images','number','img_count'])

ID = valid.iloc[:, 0].tolist()
LR = valid.iloc[:, 1].tolist()

# AGE = df.iloc[:, 2].tolist()
# ABN = df.iloc[:, 5].tolist()

LABEL = valid.iloc[:,4].tolist()

# print(ID[68])
# print(LABEL[68])


# path1 = r'E:\IAAA_CMMD\manifest-1616439774456\CMMD/' + ID[68]
# path2 = path1+ '/'+ next(os.walk(path1))[1][0]
# path3 = path2+ '/'+ next(os.walk(path2))[1][0]

# img_names = os.listdir(path3)
# print(len(img_names))


all_gts_L = []
all_gts_R = []
all_gts = []
all_imgs_in_folder  = []
x= range(0, 10)


### 4 images condition 
#for i in range(len(ID)):
for j in x:
    print('first index',j)
    path1 = r'E:/IAAA_CMMD/manifest-1616439774456/dif_classifcation/' + ID[j]
    path2 = path1+ '/'+ next(os.walk(path1))[1][0]
    path3 = path2+ '/'+ next(os.walk(path2))[1][0]
    img_names = os.listdir(path3)
    
    if len(img_names) ==2:
        for i in range(len(img_names)):
            img_path =  path3 + '/' + img_names[i]
            img = sitk.ReadImage(img_path)
            img = sitk.GetArrayFromImage(img).astype('float32')
            img = np.squeeze(img)
            all_imgs_in_folder.append(img)
            
            gt1 = LABEL[j]
            gt2 = LABEL[j]
        all_gts.append(gt1)
        all_gts.append(gt2)
            
    
for j in x:    
    path1 = r'E:/IAAA_CMMD/manifest-1616439774456/dif_classifcation/' + ID[j]
    path2 = path1+ '/'+ next(os.walk(path1))[1][0]
    path3 = path2+ '/'+ next(os.walk(path2))[1][0]
    img_names = os.listdir(path3)
    
    if LR[j] == 'L':
        for i in range(len(img_names)-2): # label the first two images with lable assign to that id
            img_path =  path3 + '/' + img_names[i]
            img = sitk.ReadImage(img_path)
            img = sitk.GetArrayFromImage(img).astype('float32')
            img = np.squeeze(img)
            all_imgs_in_folder.append(img)
            
            gt1 = LABEL[j]
            gt2 = LABEL[j]
        all_gts_L.append(gt1)
        all_gts_L.append(gt2)
            

    else:
        for i in range(len(img_names)-2): # label the first two images with lable assign to that id for R breast
            img_path =  path3 + '/' + img_names[i]
            img = sitk.ReadImage(img_path)
            img = sitk.GetArrayFromImage(img).astype('float32')
            img = np.squeeze(img)
            all_imgs_in_folder.append(img)
            
            gt1 = LABEL[j]
            gt2 = LABEL[j]
        all_gts_R.append(gt1)
        all_gts_R.append(gt2)
    
        
        
