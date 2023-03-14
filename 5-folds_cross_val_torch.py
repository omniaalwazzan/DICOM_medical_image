from PIL import Image 
import pandas as pd
import cv2
import os
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import classification_report,confusion_matrix,f1_score,roc_curve, roc_auc_score
from sklearn import metrics

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,random_split
from torchvision.transforms import transforms
from torchvision.models import resnet18


transform = transforms.Compose([transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


dir_ = r"D:\IAAA_CMMD\manifest-1616439774456\CMMD_all_png"
#dir_ = "/data/DERI-MMH/CMMD/CMMD_all_png"
#meta_df =  pd.read_csv("D:\IAAA_CMMD\manifest-1616439774456\CMMD_all_meta\cmmd_all_merg_png.csv")
meta_df = pd.read_csv(r"D:\IAAA_CMMD\manifest-1616439774456\test_simpleStik\test_meta\out_merg.csv")
meta_df['classification'] = meta_df['classification'].apply(lambda x: 0 if x == 'Benign' else 1)




class CustomDataset(Dataset):
    def __init__(self, df,image_dir, transform=None):
        self.image_dir = image_dir
        self.df = df
        self.transform = transform
        

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        # Get the unique ID of the sample
        unique_id = self.df.iloc[idx]['ID1']
        
        # Get all samples with the same ID
        samples = self.df[self.df['ID1'] == unique_id]
        
        # Choose a random sample from the list
        sample = samples.sample()
        
        # Get the image path and label
        img_path = os.path.join(self.image_dir, sample.iloc[0]['Image_name'])
        #print(img_path)
        #image_path = sample.iloc[0]['Image_name']
        label = sample.iloc[0]['classification']
        
        # Load the image and apply transformations
        #image = Image.open(image_path)
        image = cv2.imread(img_path)
        #image = cv2.resize(image, (224,224), interpolation = cv2.INTER_AREA)
        if self.transform:
            image = self.transform(image)
        
        return image, label




def create_data_loaders(dataset, fold_idx):
    # Split the dataset into training and validation sets
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_indices, val_indices = list(kf.split(dataset))[fold_idx]
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False)

    return train_loader, val_loader




# Define the model
model = resnet18(pretrained=True)
model.fc = nn.Linear(512, 2)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 5

# Loop through the folds
for fold_idx in range(5):
    # Create the dataset and data loaders for this fold
    dataset = CustomDataset(meta_df,dir_,transform=transform)
    train_loader, val_loader = create_data_loaders(dataset, fold_idx)
    total_step = len(train_loader)


    # Train and validate the model
    for epoch in range(1, n_epochs+1):
        running_loss = 0.0
        correct = 0
        total = 0
        train_acc =[]
        train_loss =[]
        for batch_idx, (data_ ,target_) in enumerate(train_loader):
            optimizer.zero_grad()
            
            outputs = model(data_.float())
            #print('shape of out',outputs.shape)
            loss = criterion(outputs, target_)
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
        
        # Evaluate the model on the validation set
        y_true = []
        y_pred = []
    model.eval()
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            y_true.extend(target.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Compute the confusion matrix for this fold
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion matrix for fold {fold_idx}:")
    print(cm)
    print(classification_report(y_true, y_pred))
    print ('F1-score micro equals: ', f1_score(y_true, y_pred, average='micro'))
    print ('F1-score macro equals: ', f1_score(y_true, y_pred, average='macro'))
    false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_true, y_pred)
    metrics.auc(false_positive_rate1, true_positive_rate1)
    print('roc_auc_score: ', roc_auc_score(y_true, y_pred))

   
