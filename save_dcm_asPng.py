import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# I need to add the path to "save" so images can be saved to a predifined path
# We use this after saving all images to one folder
# Set the input and output directories
input_dir = r'D:\IAAA_CMMD\manifest-1616439774456\test_2'
output_dir = r'D:\IAAA_CMMD\manifest-1616439774456\png_test_2'

# Loop through all DICOM files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.dcm'):
        print(filename)
        # Load the DICOM file
        
        #image_2D_path = data_path + ids_ + '_'+str(i) + '.png'
        images = sitk.ReadImage(os.path.join(input_dir, filename))
        images_array = sitk.GetArrayFromImage(images).astype('float32')
        img = np.squeeze(images_array)
        copy_img = img.copy()
        min = np.min(copy_img)
        max = np.max(copy_img)
        png_filename = os.path.splitext(filename)[0] + '.png'

        copy_img1 = copy_img - np.min(copy_img)
        copy_img = copy_img1/np.max(copy_img1)
        copy_img *= 2**8-1
        copy_img = copy_img.astype(np.uint8)
        plt.imshow(copy_img, cmap='gray')
        plt.axis('off')
        
        plt.savefig(png_filename, bbox_inches='tight',pad_inches = 0)
        plt.clf()
        
