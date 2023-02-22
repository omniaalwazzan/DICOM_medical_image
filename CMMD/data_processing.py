

import glob
import os
import pandas as pd

import SimpleITK as sitk
import numpy as np
df = pd.read_excel(os.path.join(os.getcwd(),
                              'E:\IAAA_CMMD/manifest-1616439774456', 
                              'CMMD_clinicaldata_revision.xlsx'))



ID = df.iloc[:, 0].tolist()
LR = df.iloc[:, 1].tolist()

# AGE = df.iloc[:, 2].tolist()
# ABN = df.iloc[:, 5].tolist()

LABEL = df.iloc[:,5].tolist()
df['ID1'].str.startswith('D1')
# print(ID[68])
# print(LABEL[68])


path1 = r'E:\IAAA_CMMD\manifest-1616439774456\CMMD/' + ID[1870]
path2 = path1+ '/'+ next(os.walk(path1))[1][0]
path3 = path2+ '/'+ next(os.walk(path2))[1][0]


img_names = os.listdir(path3)
print(len(img_names))
# a = 'D' + str(ids[0][i])[0] +'-' + str(ids[0][i])[1:5]
 

# check if we have two exact rows with same classification
duplicates_mask = df.duplicated(subset=['ID1', 'classification'], keep='first')

# select the rows that are not duplicated
unique_df = df[~duplicates_mask]
 
print(f'Nr. rows in train set: {unique_df.shape[0]}') 
print(unique_df.classification.value_counts())



df1 = pd.read_csv(r".\with_id.csv")
df2 = pd.read_csv(r'.fold1\f1_train - Copy.csv')
           
new = pd.merge(df1, df2, how="inner", on=["ID"])
new.to_csv(r".\f1_train_id.csv")
        
