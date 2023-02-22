

import glob
import os
import pandas as pd
import shutil
import SimpleITK as sitk
import numpy as np
df = pd.read_excel(os.path.join(os.getcwd(),
                                'E:\IAAA_CMMD/manifest-1616439774456',
                                'CMMD_clinicaldata_revision.xlsx'))


ID = df.iloc[:, 0].tolist()
LR = df.iloc[:, 1].tolist()

# AGE = df.iloc[:, 2].tolist()
# ABN = df.iloc[:, 5].tolist()

LABEL = df.iloc[:, 5].tolist()
df['ID1'].str.startswith('D1')
# print(ID[68])
# print(LABEL[68])


path1 = r'E:\IAAA_CMMD\manifest-1616439774456\CMMD/' + ID[1870]
path2 = path1 + '/' + next(os.walk(path1))[1][0]
path3 = path2 + '/' + next(os.walk(path2))[1][0]


img_names = os.listdir(path3)
print(len(img_names))


# check if we have two exact rows with same classification
duplicates_mask = df.duplicated(subset=['ID1', 'classification'], keep='first')

# select the rows that are not duplicated
unique_df = df[~duplicates_mask]

print(f'Nr. rows in train set: {unique_df.shape[0]}')
print(unique_df.classification.value_counts())


# df1 = pd.read_csv(r"E:/IAAA_CMMD/cmmd_with_id.csv")
# df2 = pd.read_csv(r'E:/IAAA_CMMD/manifest-1616439774456/CMMD_clinicaldata.csv')

# new = pd.merge(df1, unique_df, how="inner", on=["ID1"])
# new.to_csv(r"E:/IAAA_CMMD/manifest-1616439774456/combine_clincal.csv")

df3 = pd.read_csv(r"E:/IAAA_CMMD/manifest-1616439774456/combine_clincal.csv")
df3.columns
# Performing our operation
bening_ = df3.loc[df3['classification'] == 'Benign', 'Number of Images'].sum()
Malignant_ = df3.loc[df3['classification'] == 'Malignant', 'Number of Images'].sum()

print(f'Nr. of Benign MRI (per-patient) in all dataset: {bening_}')
print(f'Nr. of Malignant MRI (per-patient) in all dataset: {Malignant_}')




# count duplicate IDs
duplicated_ids = df3[df3.duplicated(subset=['ID1'], keep=False)]['ID1'].value_counts()
df4 = (duplicated_ids[duplicated_ids > 1])
df4 = df4.to_frame()


#df5 =df3.loc[df3['ID1']==df4.index]

# print the duplicated IDs that appear more than once
print(duplicated_ids[duplicated_ids > 1])

# move folders that has two classifications for the same id

PATCHES_PATH = r"E:\IAAA_CMMD\manifest-1616439774456\CMMD" # image folder 

IMG_DIR=os.listdir(PATCHES_PATH) 

Dist_PATH=r"E:/IAAA_CMMD/manifest-1616439774456/dif_classifcation" 

#Training Split
for file in df4.index:
    for name in IMG_DIR:
        if name.startswith(file):
            sourc = os.path.join(PATCHES_PATH,name )
            distination = os.path.join(Dist_PATH, name)
            shutil.move(sourc, distination)
