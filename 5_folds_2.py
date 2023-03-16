#dir_ = "/data/DERI-MMH/CMMD/CMMD_all_png"
#meta_ = "/data/DERI-MMH/CMMD/cvs/cmmd_all_merg_png.csv"
batch_size = 16

from torch.utils.data import Dataset
from PIL import Image 
import torch
from torch.utils.data import Dataset,DataLoader,random_split
import pandas as pd
import cv2
import os
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix,f1_score,roc_curve, roc_auc_score
from sklearn import metrics
from sklearn.model_selection import KFold
from torchvision.transforms import transforms
from torchvision.models import resnet18
import torch.optim as optim
import torch.nn as nn
import timm
device = "cuda" if torch.cuda.is_available() else "cpu"
#torch.backends.cudnn.benchmark = True
#from moabAtn import MOAB # this has 4 branch of att 
dir_ = r"D:\IAAA_CMMD\manifest-1616439774456\CMMD_all_png"
#meta_df =  pd.read_csv("D:\IAAA_CMMD\manifest-1616439774456\CMMD_all_meta\cmmd_all_merg_png.csv")
meta_df = pd.read_csv(r"D:\IAAA_CMMD\manifest-1616439774456\test_simpleStik\test_meta\out_merg.csv")


transform = transforms.Compose([transforms.ToPILImage(),
    transforms.Resize((384, 384)),
    transforms.ToTensor()
    #transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
])

#dir_ = r"D:\IAAA_CMMD\manifest-1616439774456\CMMD_all_png"
#meta_df =  pd.read_csv("D:\IAAA_CMMD\manifest-1616439774456\CMMD_all_meta\cmmd_all_merg_png.csv")
#meta_df = pd.read_csv(r"D:\IAAA_CMMD\manifest-1616439774456\test_simpleStik\test_meta\out_merg.csv")
#meta_df = pd.read_csv(meta_)
meta_df['classification'] = meta_df['classification'].apply(lambda x: 0 if x == 'Benign' else 1)

class CustomDataset(Dataset):
    def __init__(self, df,image_dir, transform=None):
        self.image_dir = image_dir
        self.df = df
        self.transform = transform
        self.images = df['Image_name']

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

        return image, label, self.images[idx]


def create_data_loaders(dataset, fold_idx):
    # Split the dataset into training and validation sets
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_indices, val_indices = list(kf.split(dataset))[fold_idx]
    print('train indices of fold',fold_idx, 'is',train_indices)
    print('val indices of fold',fold_idx,'is', val_indices)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,shuffle=False)

    return train_loader, val_loader




# Define the model
#model = resnet18(pretrained=True)
#model.fc = nn.Linear(512, 2)

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




# Define the loss function and optimizer
n_epochs = 20

# Creating a function to report confusion metrics
def confusion_metrics(conf_matrix):
    # save confusion matrix and slice into four pieces
    TP = conf_matrix[0][0]
    TN = conf_matrix[1][1]
    FP = conf_matrix[1][0]
    FN = conf_matrix[0][1]
    print('True Positives:', TP)
    print('True Negatives:', TN)
    print('False Positives:', FP)
    print('False Negatives:', FN)
    
    # calculate accuracy
    conf_accuracy = (float (TP+TN) / float(TP + TN + FP + FN))
    
    # calculate mis-classification
    conf_misclassification = 1- conf_accuracy
    
    # calculate the sensitivity
    conf_sensitivity = (TP / float(TP + FN))
    # calculate the specificity
    conf_specificity = (TN / float(TN + FP))
    
    # calculate precision
    conf_precision = (TN / float(TN + FP))
    # calculate f_1 score
    conf_f1 = 2 * ((conf_precision * conf_sensitivity) / (conf_precision + conf_sensitivity))
    # calculate the False positive rate (FPR)
    conf_fpr = 1 - conf_sensitivity
    # calculate the Error rate (ERR)
    conf_err = (float (FP+FN) / float(TP + TN + FP + FN))
    
    print('-'*50)
    print(f'Accuracy: {round(conf_accuracy,2)}') 
    print(f'Mis-Classification: {round(conf_misclassification,2)}') 
    print(f'Sensitivity: {round(conf_sensitivity,2)}') 
    print(f'Specificity: {round(conf_specificity,2)}') 
    print(f'Precision: {round(conf_precision,2)}')
    print(f'False positive rate (FPR) Score: {round(conf_fpr,2)}')
    print(f'Error rate (ERR) Score: {round(conf_err,2)}')
    print(f'f_1 Score: {round(conf_f1,2)}')

    
    

def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            #print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


# Loop through the folds
for fold_idx in range(5):
    # Create the dataset and data loaders for this fold
    dataset = CustomDataset(meta_df,dir_,transform=transform)
    train_loader, val_loader = create_data_loaders(dataset, fold_idx)
    total_step = len(train_loader)
    print('Len of train loader in fold#',fold_idx,'is',len(train_loader))
    print('Len of val loader in fold#',fold_idx,'is',len(val_loader))
    model = load_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.apply(reset_weights)
    # Train and validate the model
    for epoch in range(1, n_epochs+1):
        running_loss = 0.0
        correct = 0
        total = 0
        train_acc =[]
        train_loss =[]
        model.train()
        for batch_idx, (data_ ,target_,img_n) in enumerate(train_loader):
            #model.train()
            optimizer.zero_grad()
            data_, target_ = data_.to(device), target_.to(device)
            outputs = model(data_.float())
            #print('shape of out',outputs.shape)
            loss = criterion(outputs, target_)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _,pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred==target_).item()
            total += target_.size(0)
            if (batch_idx) % 40 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
        train_acc.append(100 * correct / total)
        train_loss.append(running_loss/total_step)
        print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct/total):.4f}')
        
        # Evaluate the model on the validation set
        y_true = []
        y_pred = []
    with torch.no_grad():
        model.eval()
        for data, target,img_name in val_loader:
            data = data.to(device)
            target = target.to(device)
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
    confusion_metrics(cm)


   


