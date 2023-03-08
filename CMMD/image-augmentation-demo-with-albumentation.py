#''Use case from https://www.kaggle.com/alexanderliao/image-augmentation-demo-with-albumentation/'''

"""
Created on Wed Mar  8 21:30:33 2023

@author: Omnia
"""
import cv2
from PIL import Image
import numpy as np
import math
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

data_path1 = r"D:\IAAA_CMMD\manifest-1616439774456\test_simpleStik\test_meta/"
data_path= r"D:\IAAA_CMMD\manifest-1616439774456/"
df = pd.read_csv(data_path1 + 'out_merg.csv')

def preprocess_image(image_path, desired_size=512):
    im = Image.open(image_path).convert('RGB')
    im = im.resize((desired_size, )*2, resample=Image.LANCZOS)
#     im = im.resize((desired_size, )*2)
    
    return im

def get_pad_width(im, new_shape, is_rgb=True):
    pad_diff = new_shape - im.shape[0], new_shape - im.shape[1]
    t, b = math.floor(pad_diff[0]/2), math.ceil(pad_diff[0]/2)
    l, r = math.floor(pad_diff[1]/2), math.ceil(pad_diff[1]/2)
    if is_rgb:
        pad_width = ((t,b), (l,r), (0, 0))
    else:
        pad_width = ((t,b), (l,r))
    return pad_width


N = df.shape[0]
x_test = np.empty((N, 512, 512, 3), dtype=np.uint8)

for i, image_id in enumerate(tqdm(df['Img_name'])):
    x_test[i, :, :, :] = preprocess_image(
        f'D:\IAAA_CMMD\manifest-1616439774456\png_test_2/{image_id}'
    )
    
    
from albumentations import *
import time

IMG_SIZE = (512,512)



def albaugment(aug0, img):
    
    return aug0(image=img)['image']
idx=8
image1=x_test[idx]

def show_image(image,figsize=None,title=None):
    
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
#     else: # crash!!
#         fig = plt.figure()
        
    if image.ndim == 2:
        plt.imshow(image,cmap='gray')
    else:
        plt.imshow(image)
        
    if title is not None:
        plt.title(title)

def show_Nimages(imgs,scale=1):

    N=len(imgs)
    fig = plt.figure(figsize=(25/scale, 16/scale))
    for i, img in enumerate(imgs):
        ax = fig.add_subplot(1, N, i + 1, xticks=[], yticks=[])
        show_image(img)
        
def print_pred(array_of_classes):
    xx = array_of_classes
    s1,s2 = xx.shape
    for i in range(s1):
        for j in range(s2):
            print('%.3f ' % xx[i,j],end='')
        print('')

def transform_image_ben(img,resize=True,crop=False,norm255=True,keras=False):  
    image=cv2.addWeighted( img,4, cv2.GaussianBlur( img , (0,0) ,  10) ,-4 ,128)
    
    # NOTE plt.imshow can accept both int (0-255) or float (0-1), but deep net requires (0-1)
    if norm255:
        return image/255
    elif keras:
        image = np.expand_dims(image, axis=0)
        return preprocess_input(image)[0]
    else:
        return image.astype(np.int16)
    
    return image

'''1. Rotate or Flip'''
aug1 = OneOf([
    Rotate(p=0.99, limit=160, border_mode=0,value=0), # value=black
    Flip(p=0.5)
    ],p=1)

'''2. Adjust Brightness or Contrast'''
aug2 = RandomBrightnessContrast(brightness_limit=0.45, contrast_limit=0.45,p=1)
h_min=np.round(IMG_SIZE[1]*0.72).astype(int)
h_max= np.round(IMG_SIZE[1]*0.9).astype(int)
print(h_min,h_max)

'''3. Random Crop and then Resize'''
#w2h_ratio = aspect ratio of cropping
aug3 = RandomSizedCrop((h_min, h_max),IMG_SIZE[1],IMG_SIZE[0], w2h_ratio=IMG_SIZE[0]/IMG_SIZE[1],p=1)

'''4. CutOut Augmentation'''
max_hole_size = int(IMG_SIZE[1]/10)
aug4 = Cutout(p=1,max_h_size=max_hole_size,max_w_size=max_hole_size,num_holes=8 )#default num_holes=8

'''5. SunFlare Augmentation'''
aug5 = RandomSunFlare(src_radius=max_hole_size,
                      num_flare_circles_lower=10,
                      num_flare_circles_upper=20,
                      p=1)#default flare_roi=(0,0,1,0.5),

'''6. Ultimate Augmentation -- combine everything'''
final_aug = Compose([
    aug1,aug2,aug3,aug4,aug5
],p=1)


img1 = albaugment(aug1,image1)
img2 = albaugment(aug1,image1)
print('Rotate or Flip')
show_Nimages([image1,img1,img2],scale=2)
# time.sleep(1)

img1 = albaugment(aug2,image1)
img2 = albaugment(aug2,image1)
img3 = albaugment(aug2,image1)
print('Brightness or Contrast')
show_Nimages([img3,img1,img2],scale=2)
# time.sleep(1)

img1 = albaugment(aug3,image1)
img2 = albaugment(aug3,image1)
img3 = albaugment(aug3,image1)
print('Rotate and Resize')
show_Nimages([img3,img1,img2],scale=2)
print(img1.shape,img2.shape)
# time.sleep(1)

img1 = albaugment(aug4,image1)
img2 = albaugment(aug4,image1)
img3 = albaugment(aug4,image1)
print('CutOut')
show_Nimages([img3,img1,img2],scale=2)
# time.sleep(1)

img1 = albaugment(aug5,image1)
img2 = albaugment(aug5,image1)
img3 = albaugment(aug5,image1)
print('Sun Flare')
show_Nimages([img3,img1,img2],scale=2)
# time.sleep(1)

img1 = albaugment(final_aug,image1)
img2 = albaugment(final_aug,image1)
img3 = albaugment(final_aug,image1)
print('All above combined')
show_Nimages([img3,img1,img2],scale=2)
print(img1.shape,img2.shape)
    
aug_list = [aug5, aug2, aug3, aug4, aug1, final_aug]
aug_name = ['SunFlare', 'brightness or contrast', 'crop and resized', 'CutOut', 'rotate or flip', 'Everything Combined']

idx=8
layer_name = 'relu' #'conv5_block16_concat'
for i in range(len(aug_list)):
    #"D:\IAAA_CMMD\manifest-1616439774456\png_test_2/{image_id}"
    path=f"D:\IAAA_CMMD\manifest-1616439774456\png_test_2/{df.iloc[idx]['Img_name']}"
    input_img = np.empty((1,512, 512, 3), dtype=np.uint8)
    input_img[0,:,:,:] = preprocess_image(path)
    aug_img = albaugment(aug_list[i],input_img[0,:,:,:])
    ben_img = transform_image_ben(aug_img)
    
    print('test pic no.%d -- augmentation: %s' % (i+1, aug_name[i]))
