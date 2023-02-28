dir_ = r"D:\IAAA_CMMD\manifest-1616439774456\CMMD_all_png"
meta_csv = r"D:/IAAA_CMMD/manifest-1616439774456/CMMD_all_meta/cmmd_all_merg_png.csv"

PIN_MEMORY=True    
NUM_WORKERS=0
batch_size = 10

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix,f1_score
import time
import numpy as np
import torch
from torchvision import transforms
import random
from torch.utils.data import Dataset,DataLoader,random_split
import matplotlib.pyplot as plt
import seaborn as sns
device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import cv2
import torch.nn as nn
import torch.optim as optim




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
val_size = int(0.2 * len(load_data))
test_size = len(load_data) - train_size - val_size

# Use random_split to split the dataset into training, validation, and testing sets
train_set, val_set, test_set = random_split(load_data, [train_size, val_size, test_size])

# Define the batch size and create DataLoaders for each set

## Data Generator ##

train_loader = DataLoader(train_set, batch_size=batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY,shuffle=False)

# a=iter(train_loader)
# a1=next(a)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=True, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss



def train_model(model, device, train_loader,valid_loader,patience, optimizer,criterion,n_epochs):
    
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    for epoch in range(1, n_epochs + 1):

        ###################
        # train the model #
        ###################
        model.train() # prep model for training
        model.to(device=device,dtype=torch.float32)
        for batch, (image_patch, gt,img_name) in enumerate(train_loader, 1):           
            image_patch = image_patch.to(device=device,dtype=torch.float32)
            gt = gt.to(device=device,dtype=torch.long)          
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(image_patch)
            # calculate the loss
            loss = criterion(output, gt)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())

        ######################    
        # validate the model #
        ######################
        model.eval() # prep model for evaluation
        #for image_patch, gt,img_name in valid_loader:
        for  batch,(image_patch, gt,img_name) in enumerate(valid_loader,1):
            image_patch = image_patch.to(device=device,dtype=torch.float32)
            gt = gt.to(device=device,dtype=torch.long)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(image_patch)
            # calculate the loss
            loss = criterion(output, gt)
            # record validation loss
            valid_losses.append(loss.item())

        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        epoch_len = len(str(n_epochs))
        
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        print(print_msg)
        
        # clear lists to track next epoch
        #train_losses = []
        #valid_losses = []
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))

    return  model, avg_train_losses, avg_valid_losses


                
       ############################################################
       
       
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

model = load_model().to(device)

## Main ##
loss_fn1 = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.0005)
#optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
n_epochs = 4
# early stopping patience; how long to wait after last time validation loss improved.
patience = 3
# training 
model, train_loss, valid_loss = train_model(model, device,train_loader,val_loader,patience, optimizer,loss_fn1,n_epochs)


    
# visualize the loss as the network trained
fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')

# find position of lowest validation loss
font1 = {'size':20}
minposs = valid_loss.index(min(valid_loss))+1 
plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
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

### Testing

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
tick_marks = np.arange(len(classes))

df_cm = pd.DataFrame(cm, index = classes, columns = classes)
plt.figure(figsize = (7,7))
sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g')
plt.xlabel("Predicted label", fontsize = 20)
plt.ylabel("Ground Truth", fontsize = 20)
#plt.show()
df_cm.to_csv('cm_exp46_2_earlystopping.csv')
#df_cm.to_csv('cm_convnext_concat_mlp_earlystopping.csv')

print(classification_report(truelabels, predictions))

print ('F1-score micro equals:')
print(f1_score(truelabels, predictions, average='micro'))
