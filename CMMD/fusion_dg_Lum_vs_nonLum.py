
dir_ = "/data/DERI-MMH/CMMD/CMMD_all_png"
#meta_csv = "/data/DERI-MMH/CMMD/cvs/cmmd_all_merg_png.csv"

#dir_ = r"D:\IAAA_CMMD\manifest-1616439774456\CMMD_all_png"
#meta_csv = r'E:/IAAA_CMMD/manifest-1616439774456/CMMD_all_meta/cmmd_all_merg_png.csv'


meta_csv_train = "/data/DERI-MMH/CMMD/SUBTYPE/L_vs_nonL/train.csv"
meta_csv_val = "/data/DERI-MMH/CMMD/SUBTYPE/L_vs_nonL/test.csv"
meta_csv_test = "/data/DERI-MMH/CMMD/SUBTYPE/L_vs_nonL/val.csv"

PIN_MEMORY=True    
NUM_WORKERS = 8
batch_size = 32




import pandas as pd
import numpy as np
import cv2
import os
import torch
from torch.utils.data import Dataset,DataLoader,random_split
import torch.optim as optim
import torch.nn as nn
from torchvision.transforms import transforms

from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix,f1_score,roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def dataImpute(meta_csv_train):
    
## Data Analysis ##
    train = pd.read_csv(meta_csv_train)
    train.columns
    bening_ = train.loc[train['subtype'] == 0, 'Number of Images'].sum()
    Malignant_ = train.loc[train['subtype'] == 1, 'Number of Images'].sum()
    print(f'Nr. of Non-Luminal MRI (per-patient) in training set: {bening_}')
    print(f'Nr. of Luminal MRI (per-patient) in training set: {Malignant_}')
    train = train.drop(columns=['Number of Images', 'number'])
    train.columns
    # age impute
    train.iloc[:,3]= (train.iloc[:, 3]- train.iloc[:, 3].mean()) / train.iloc[:, 3].std()
    le = LabelEncoder()
    train['abnormality'] = le.fit_transform(train['abnormality'])
    train['LeftRight'] = le.fit_transform(train['LeftRight'])
    return train

train_data = dataImpute(meta_csv_train)
train_data['ID1'].nunique()

val_data = dataImpute(meta_csv_val)
val_data['ID1'].nunique()

test_data = dataImpute(meta_csv_test)
test_data['ID1'].nunique()

#train.classification.value_counts()

class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img

size = 96
s =1  
data_transforms = transforms.Compose([#transforms.RandomResizedCrop(size=size),
                                        transforms.ToPILImage(),
                                        transforms.RandomHorizontalFlip(),
                                        #transforms.RandomApply([color_jitter], p=0.8),
                                        GaussianBlur(kernel_size=int(0.1 * size)),
                                        transforms.RandomGrayscale(p=0.2),
                                        transforms.ToTensor()])   



class Dataset_(Dataset):
    def __init__(self, image_dir,df_meta,transform=None):
        self.image_dir = image_dir # --> r'E:/IAAA_CMMD/manifest-1616439774456/CMMD/'
        
        self.df_meta=df_meta # --> csv file
        self.LABEL = df_meta['subtype']#.tolist()
        #self.AGE = df_meta['Age']
        #self.LR = df_meta['LeftRight']
        #self.ABN = df_meta['abnormality']
        #self.SUB = df_meta['subtype']        
        self.images = df_meta['Image_name']
        #self.images = df_meta['Img_name']


        self.transform = transform

    def __len__(self):
        return self.df_meta.shape[0]
    
    def __getitem__(self, index):
        ## reading image ###
        img_path = os.path.join(self.image_dir, self.images[index])
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_three = cv2.merge([gray,gray,gray])
        image = cv2.resize(gray_three, (224,224), interpolation = cv2.INTER_AREA)
        #img__ = img__.transpose([2,0,1])
        image_name=self.images[index]
        gt = self.LABEL[index]
        
        c1=self.df_meta.loc[self.df_meta['Image_name']==image_name]

        c2 = c1.iloc[:, 2:-3]
        c2 =np.array(c2)
        


        #cLin_features = np.array(cLin_features)
        if self.transform is not None:
            image = self.transform(image = image)['image']
	    #a = self.transform(img__)
        #if self.transform is not None:
            # a = self.transform(image=image)
            # image = a['image']
            #image=np.transpose(image, (2, 0, 1))
            
        #return img__,LR_,AGE_,ABN_,gt,c2,self.images[index]    
            
        return image,c2,gt,self.images[index]

## Read the data ##
train_set = Dataset_(image_dir=dir_,df_meta=train_data,transform=data_transforms)
val_set = Dataset_(image_dir=dir_,df_meta=val_data,transform=data_transforms)
test_set = Dataset_(image_dir=dir_,df_meta=test_data,transform=data_transforms)


## Data Spliting ##

# # Define the sizes of the training, validation, and testing sets
# train_size = int(0.7 * len(load_data))
# val_size = int(0.15 * len(load_data))
# test_size = len(load_data) - train_size - val_size

# # Use random_split to split the dataset into training, validation, and testing sets
# train_set, val_set, test_set = random_split(load_data, [train_size, val_size, test_size])

# gt_t = train_set.dataset.LABEL[train_set.indices]

# gt_v = val_set.dataset.LABEL[val_set.indices]

# gt_te = test_set.dataset.LABEL[test_set.indices]


## Data Generator ##
def fusion_dg():
    train_loader = DataLoader(train_set, batch_size=batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY,shuffle=False)
    
    print('Len. of train data', len(train_loader.dataset))
    print('Len. of valid data', len(val_loader.dataset))
    print('Len. of test data', len(test_loader.dataset))
    
    return train_loader, val_loader, test_loader


#train_loader, val_loader, test_loader = fusion_dg()


# validate dg is working
#a=iter(train_loader)
#a1=next(a)
#meta = a1[1]
#print('meta has',a1[1].numpy())

#print('meta',meta.shape)
#print('gt has',a1[2].numpy())
#print('gt shape',a1[2].shape)

   
