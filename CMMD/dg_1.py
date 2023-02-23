import pandas as pd
import SimpleITK as sitk
from sklearn.preprocessing import LabelEncoder
import time
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






## Data analysis##

train = pd.read_csv(r"E:/IAAA_CMMD/manifest-1616439774456/train.csv")
train.columns

bening_ = train.loc[train['classification'] == 'Benign', 'Number of Images'].sum()
Malignant_ = train.loc[train['classification'] == 'Malignant', 'Number of Images'].sum()

print(f'Nr. of Benign MRI (per-patient) in all dataset: {bening_}')
print(f'Nr. of Malignant MRI (per-patient) in all dataset: {Malignant_}')
train = train.drop(columns=['Number of Images','number'])
train.columns


#### Data imputations 

# age impute
train.iloc[:, 2]= (train.iloc[:, 2]- train.iloc[:, 2].mean()) / train.iloc[:, 2].std()


#train.iloc[:, 5].tail(20)

le = LabelEncoder()

# Encode the categorical variable abnormality and LeftRight
train['abnormality'] = le.fit_transform(train['abnormality'])
train['LeftRight'] = le.fit_transform(train['LeftRight'])


#train.fillna(0,inplace=True)
# change the class label to 0 or 1
train['classification'] = train['classification'].apply(lambda x: 0 if x == 'Benign' else 1)
train




###############
ID_1 = train['ID1'].tolist()
ID = train.iloc[:, 0].tolist()
LR = train.iloc[:, 1].tolist()

AGE = train.iloc[:, 2].tolist()
ABN = train.iloc[:, 3].tolist()

LABEL = train.iloc[:,4].tolist()


all_gts2 = [] # this is for tow images inside a folder
all_gts4 = [] # this is for four images inside a folder
#all_gts = []
all_imgs_in_folder  = []
x= range(0, 23)


start = time.time()

### 4 images condition
 
#for j in range(len(ID)):
for j in x:
    #print('first index',j)
    path1 = r'E:/IAAA_CMMD/manifest-1616439774456/CMMD/' + ID[j]
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
        all_gts2.append(gt1)
        all_gts2.append(gt2)
        
        
    else:
        for i in range(len(img_names)):
            #print(len(img_names))
            img_path =  path3 + '/' + img_names[i]
            img = sitk.ReadImage(img_path)
            img = sitk.GetArrayFromImage(img).astype('float32')
            img = np.squeeze(img)
            all_imgs_in_folder.append(img)
            
            gt1 = LABEL[j]
            gt2 = LABEL[j]
            gt3 = LABEL[j]
            gt4 = LABEL[j]
        all_gts4.append(gt1)
        all_gts4.append(gt2)
        all_gts4.append(gt3)
        all_gts4.append(gt4)
        
end = time.time()
print('time spent to execuate',end - start)       
            


transform_1 =  transforms.Compose([transforms.ToTensor()])

def read_MRI(path, voi_lut = True, fix_monochrome = True):
    dicom = pydicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
               
    # depending on this value, MRI may look inverted - fix that:
    # if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME2":
    #     data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.float64)
        
    return data


class Dataset_(Dataset):
    def __init__(self, image_dir,df_meta,transform=None):
        self.image_dir = image_dir # --> r'E:/IAAA_CMMD/manifest-1616439774456/CMMD/'
        
        self.df_meta=df_meta # --> csv file
        self.ID = df_meta['ID1'].tolist() #df_meta.iloc[:, 0].tolist()
        self.LR = df_meta['LeftRight'].tolist()#df_meta.iloc[:, 1].tolist()
        self.AGE = df_meta['Age'].tolist()
        self.ABN = df_meta['abnormality'].tolist()

        self.LABEL = df_meta['classification'].tolist()
        self.SUB = df_meta['subtype'].tolist()

        self.path1 = image_dir #+ ID[j]
        #self.img = img
        #self.path2 = path1 #  path1+ '/'+ next(os.walk(path1))[1][0]
        #self.path3 = path2#path2+ '/'+ next(os.walk(path2))[1][0]
        #self.img_names = img_names#os.listdir(path3)

        #AGE = train.iloc[:, 2].tolist()
        #ABN = train.iloc[:, 3].tolist()

        #LABEL = train.iloc[:,4].tolist()
       # self.genes=path_genes_data
        #self.images = list(image_dir.glob("*/*/*/*.dcm"))
        self.transform = transform

    def __len__(self):
        return self.df_meta.shape[0]
    

    def __getitem__(self, index):
        ## reading image ###
        ID = self.ID
        LABEL = self.LABEL
        img_path1 = self.path1+ ID[index]
        img_path2 = img_path1 + '/'+ next(os.walk(img_path1))[1][0]
        img_path3 = img_path2 + '/'+ next(os.walk(img_path2))[1][0]
        
        img_names = os.listdir(img_path3)
        
        if len(img_names) ==2:
            for i in range(len(img_names)):
                img_path =  img_path3 + '/' + img_names[i]
                img = sitk.ReadImage(img_path)
                img = sitk.GetArrayFromImage(img).astype('float32')
                img = np.squeeze(img)
                all_imgs_in_folder.append(img)
                
                gt1 = LABEL[index]
                gt2 = LABEL[index]
            all_gts2.append(gt1)
            all_gts2.append(gt2)
            
            
        # else:
        #     for i in range(len(img_names)):
        #         #print(len(img_names))
        #         img_path =  path3 + '/' + img_names[i]
        #         img = sitk.ReadImage(img_path)
        #         img = sitk.GetArrayFromImage(img).astype('float32')
        #         img = np.squeeze(img)
        #         all_imgs_in_folder.append(img)
                
        #         gt1 = LABEL[j]
        #         gt2 = LABEL[j]
        #         gt3 = LABEL[j]
        #         gt4 = LABEL[j]
        #     all_gts4.append(gt1)
        #     all_gts4.append(gt2)
        #     all_gts4.append(gt3)
        #     all_gts4.append(gt4)

        #img_path = os.path.join(self.image_dir, self.images[index])
        print(img_names)
        #image = read_MRI(img_path)
        #plt.figure(figsize = (12,12))
        #plt.imshow(image, 'gray')

        #image_name=self.images[index]
        ### read the image name from the clinical data ###
  

        #if self.transform is not None:
            # a = self.transform(image=image)
            # image = a['image']
            #image=np.transpose(image, (2, 0, 1))
            
        return all_imgs_in_folder,all_gts2#,self.images[index]
        #return image,self.images[index]




 
def Data_Loader( test_dir,clinical_path,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    
    test_ids = Dataset_( image_dir=test_dir,df_meta=clinical_path)#,transform=transform_1)

    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    
    return data_loader


dir_ = r'E:/IAAA_CMMD/manifest-1616439774456/CMMD/'


# data_path = Path(r"E:\IAAA_CMMD\manifest-1616439774456")
# images_folder = data_path / "CMMD"
# clinical_path=r'E:/IAAA_CMMD/manifest-1616439774456/all_data.csv'

loader=Data_Loader(dir_,train,6)
a=iter(loader)
a1=next(a)



