
dir_ = r"D:\IAAA_CMMD\manifest-1616439774456\png_test_2"
meta_csv = r"D:/IAAA_CMMD/manifest-1616439774456/test/out_merg.csv"

PIN_MEMORY=True    
NUM_WORKERS=0
batch_size = 3
n_epochs = 10
print_every = 10
LR = 0.001 

import pandas as pd
import numpy as np
import cv2
import os
import torch
from torch.utils.data import Dataset,DataLoader,random_split
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix,f1_score,roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Data Analysis ##
train = pd.read_csv(meta_csv)
train.columns

bening_ = train.loc[train['classification'] == 'Benign', 'Number of Images'].sum()
Malignant_ = train.loc[train['classification'] == 'Malignant', 'Number of Images'].sum()

print(f'Nr. of Benign MRI (per-patient) in all dataset: {bening_}')
print(f'Nr. of Malignant MRI (per-patient) in all dataset: {Malignant_}')
train = train.drop(columns=['Number of Images','number'])
train.columns


#### Data imputations 

# age impute
train.iloc[:,3]= (train.iloc[:, 3]- train.iloc[:, 3].mean()) / train.iloc[:, 3].std()
# Encode the categorical variable abnormality and LeftRight
le = LabelEncoder()
train['abnormality'] = le.fit_transform(train['abnormality'])
train['LeftRight'] = le.fit_transform(train['LeftRight'])
# change the class label to 0 or 1
train['classification'] = train['classification'].apply(lambda x: 0 if x == 'Benign' else 1)


class Dataset_(Dataset):
    def __init__(self, image_dir,df_meta,transform=None):
        self.image_dir = image_dir # --> r'E:/IAAA_CMMD/manifest-1616439774456/CMMD/'
        
        self.df_meta=df_meta # --> csv file
        self.LABEL = df_meta['classification']#.tolist()        
        self.images = os.listdir(image_dir)
        #self.images = df_meta['Image_name']
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
        img__ = cv2.resize(gray_three, (96,96), interpolation = cv2.INTER_AREA)
        img__ = img__.transpose([2,0,1])
        image_name=self.images[index]
        gt = self.LABEL[index]
        
        if self.transform is not None:
            a = self.transform(img__)
        #if self.transform is not None:
            # a = self.transform(image=image)
            # image = a['image']
            #image=np.transpose(image, (2, 0, 1))
            
        return img__,gt,self.images[index]

## Read the data ##
load_data = Dataset_(image_dir=dir_,df_meta=train)



## Data Spliting ##

# Define the sizes of the training, validation, and testing sets
train_size = int(0.7 * len(load_data))
val_size = int(0.1 * len(load_data))
test_size = len(load_data) - train_size - val_size

# Use random_split to split the dataset into training, validation, and testing sets
train_set, val_set, test_set = random_split(load_data, [train_size, val_size, test_size])


## Data Generator ##

train_loader = DataLoader(train_set, batch_size=batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY,shuffle=False)

a=iter(train_loader)
a1=next(a)


print('Len. of train data', len(train_loader.dataset))
print('Len. of valid data', len(val_loader.dataset))
print('Len. of test data', len(test_loader.dataset))

     
       
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

def accuracy(out, labels):
    _,pred = torch.max(out, dim=1)
    return torch.sum(pred==labels).item()


model = load_model().to(device)
optimizer = optim.Adam(model.parameters(),lr=LR)
loss_fn1 = nn.CrossEntropyLoss()




valid_loss_min = np.Inf
val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step = len(train_loader)
for epoch in range(1, n_epochs+1):
    running_loss = 0.0
    correct = 0
    total=0
    print(f'Epoch {epoch}\n')
    for batch_idx, (data_, target_,img_n) in enumerate(train_loader):
        data_, target_ = data_.to(device), target_.to(device)
        optimizer.zero_grad()
        
        outputs = model(data_.float())
        loss = loss_fn1(outputs, target_)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _,pred = torch.max(outputs, dim=1)
        correct += torch.sum(pred==target_).item()
        total += target_.size(0)
        if (batch_idx) % 20 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
    train_acc.append(100 * correct / total)
    train_loss.append(running_loss/total_step)
    print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct/total):.4f}')
    batch_loss = 0
    total_t=0
    correct_t=0
    with torch.no_grad():
        model.eval()
        for data_t, target_t, img_n in (val_loader):
            data_t, target_t = data_t.to(device), target_t.to(device)
            outputs_t = model(data_t.float())
            loss_t = loss_fn1(outputs_t, target_t)
            batch_loss += loss_t.item()
            _,pred_t = torch.max(outputs_t, dim=1)
            correct_t += torch.sum(pred_t==target_t).item()
            total_t += target_t.size(0)
        val_acc.append(100 * correct_t/total_t)
        val_loss.append(batch_loss/len(val_loader))
        network_learned = batch_loss < valid_loss_min
        print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t/total_t):.4f}\n')

        
        if network_learned:
            valid_loss_min = batch_loss
            torch.save(model.state_dict(), 'resnet.pt')
            print('Improvement-Detected, save-model')
    model.train()



fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
plt.plot(range(1,len(val_loss)+1),val_loss,label='Validation Loss')
# find position of lowest validation loss
font1 = {'size':20}
minposs = val_loss.index(min(val_loss))+1 
#plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
plt.title("Learning Curve Graph",fontdict = font1)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0, 1) # consistent scale
plt.xlim(0, len(train_loss)+1) # consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
#plt.show()
fig.savefig('loss_plot.png', bbox_inches='tight')




plt.figure(figsize=(10,5))
plt.title("Training and Validation acc")
plt.plot(val_acc,label="val")
plt.plot(train_acc,label="train")
plt.xlabel("iterations")
plt.ylabel("Acc")
plt.legend()
fig.savefig('acc_plot.png', bbox_inches='tight')

    
    

truelabels = []
predictions = []
model.eval()
model.to(device=device,dtype=torch.float32)
print("Getting predictions from test set...")
for image_patch,gt,img_name in test_loader:

    image_patch = image_patch.to(device=device,dtype=torch.float32)
    gt = gt.to(device=device,dtype=torch.long)   
    for label in gt.cpu().data.numpy():
        truelabels.append(label)
    for prediction in model(image_patch).cpu().data.numpy().argmax(1):
        predictions.append(prediction)     
        
        
## ploting
cm = confusion_matrix(truelabels, predictions)
classes= ['Benign', 'Malignant']
#classes= ['0']
tick_marks = np.arange(len(classes))

df_cm = pd.DataFrame(cm, index = classes, columns = classes)
plt.figure(figsize = (7,7))
sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g')
plt.xlabel("Predicted label", fontsize = 20)
plt.ylabel("Ground Truth", fontsize = 20)
plt.savefig('cm.png', bbox_inches='tight')

#plt.show()
#df_cm.to_csv('cm_exp46_2_earlystopping.csv')
#df_cm.to_csv('cm_convnext_concat_mlp_earlystopping.csv')

print(classification_report(truelabels, predictions))

print ('F1-score micro equals:')
print(f1_score(truelabels, predictions, average='micro'))

false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(truelabels, predictions)

print('roc_auc_score: ', roc_auc_score(truelabels, predictions))


plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristi')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
fig.savefig('roc.png', bbox_inches='tight')
