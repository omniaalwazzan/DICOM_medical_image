import glob
import os
import pandas as pd

import SimpleITK as sitk
import numpy as np
train = pd.read_csv(r"E:/IAAA_CMMD/manifest-1616439774456/train.csv")
train.columns

bening_ = train.loc[train['classification'] == 'Benign', 'Number of Images'].sum()
Malignant_ = train.loc[train['classification'] == 'Malignant', 'Number of Images'].sum()

print(f'Nr. of Benign MRI (per-patient) in all dataset: {bening_}')
print(f'Nr. of Malignant MRI (per-patient) in all dataset: {Malignant_}')
train = train.drop(columns=['Number of Images','number'])

ID = train.iloc[:, 0].tolist()
LR = train.iloc[:, 1].tolist()

# AGE = df.iloc[:, 2].tolist()
# ABN = df.iloc[:, 5].tolist()

LABEL = train.iloc[:,4].tolist()


all_gts2 = []
all_gts4 = []
#all_gts = []
all_imgs_in_folder  = []
x= range(0, 23)


### 4 images condition 
#for i in range(len(ID)):
for j in x:
    print('first index',j)
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
            print(len(img_names))
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
        
        
            
