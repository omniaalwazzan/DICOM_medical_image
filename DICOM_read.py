
import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
import imageio

# ref: https://towardsdatascience.com/dealing-with-dicom-using-imageio-python-package-117f1212ab82

#Stacking 99 slices from one case
im = imageio.volread(r'C:\Users\omnia\Desktop\manifest-1542731172463\QIN-BREAST\QIN-BREAST-01-0001\09-01-1991-NA-BREAST PRONE-21963\2.000000-CTAC-02377')
im.meta

# The Attributes of the DICOM File
im.meta.keys()
im.meta['Modality']

# Access the Pixel Data attribute
im.meta['PixelData']


#Slicing the first 5 rows and first 5 columns
im[0:50, 0:50]

#Access the first slice
im[0,:,:]
#Show the image with gray color-map
plt.imshow(im, cmap='gray')
#Don't show tha axes
plt.axis('off')
#Add a title to the plot
plt.title('Axial Slice')
plt.show()


#Show the first slice.
plt.imshow(im[0,:,:], cmap='gray')
#Don't show the axis
plt.axis('off')
#Add a title
plt.title('Axial Slice')
plt.show()

# The shape of the stacked images in each plane
# (Axial, Coronal, and Sagittal, respectively)
n0, n1, n2 = im.shape
# Print the ouput
print("Number of Slices:\n\t", "Axial=", n0, "Slices\n\t",
                               "Coronal=", n1, "Slices\n\t",
                               "Sagittal=", n2, "Slices")

# The sampling of the stacked images in each plane
# (Axial, Coronal, and Sagittal, respectively)
d0, d1, d2 = im.meta['sampling'] # in mm
# Print the output
print("Sampling:\n\t", "Axial=", d0, "mm\n\t",
                               "Coronal=", d1, "mm\n\t",
                               "Sagittal=", d2, "mm")


# The aspect ratio along the axial plane
axial_asp = d1/d2
# The aspect ratio along the sagittal plane
sagittal_asp = d0/d1
# The aspect ratio along the coronal plane
coronal_asp = d0/d2
# Print the output
print("Pixel Aspect Ratio:\n\t", "Axial=", axial_asp, "\n\t",
                               "Coronal=", coronal_asp, "\n\t",
                               "Sagittal=", sagittal_asp)

print("Field of View:\n\t", "Axial=", n0*d0, "mm\n\t",
                            "Coronal=", n1*d1, "mm\n\t",
                            "Sagittal=", n2*d2, "mm")



# Add a slider that starts with 0 and ends at the number of
# slices along the axial plane, n0=99.
@widgets.interact(axial_slice=(0,n0-1))
# Define the function that shows the images of the specified slice number.
# It starts with the 10th slice. And you can scroll over any slice
# using the slider.
def axial_slicer(axial_slice=50):
  fig, ax = plt.subplots(1, 1, figsize=(8, 8))
  # Show the image of the specified slice number in 'gray' color-map
  # and axial aspect ratio
  ax.imshow(im[axial_slice,:,:], cmap='gray', aspect=axial_asp)
  # Don't show the axis
  ax.axis('off')
  

# Define a figure with 1 row and 3 columns of plots to show
# the images along the three planes
fig, ax = plt.subplots(1, 3, figsize=(8, 8))
# Axial Plane: show the 10th slice
ax[0].imshow(im[10,:,:], cmap='gray', aspect= axial_asp)
#ax[0].axis('off')
ax[0].set_title('Axial')

# Coronal Plane: show the slice 100
ax[1].imshow(im[:,100,:],cmap='gray', aspect= coronal_asp)
#ax[1].axis('off')
ax[1].set_title('Coronal')

# Sagittal Plane: show the slice 100
ax[2].imshow(im[:,:,100], cmap='gray', aspect= sagittal_asp)
#ax[2].axis('off')
ax[2].set_title('Sagittal')
plt.show()



# Add three sliders that start with 0 and ends at the number of slices
# along each plane.
# Axial:    n0=99   slice
# Corornal: n1=512  slice
# Sagittal: n2=512  slice
@widgets.interact(axial_slice=(0,n0-1), coronal_slice=(0,n1-1),\
                  sagittal_slice=(0,n2-1))
def slicer(axial_slice, coronal_slice, sagittal_slice=100):
  fig, ax = plt.subplots(1, 3, figsize=(12, 12))

  # Show the specfied slice on the axial plane with 'gray' color-map
  # and axial aspect ratio.
  ax[0].imshow(im[axial_slice,:,:], cmap='gray', aspect= axial_asp)
  ax[0].axis('off')
  ax[0].set_title('Axial')

  # Show the specified slice on the coronal plane with 'gray' color-map
  # and coronal aspect ratio.
  ax[1].imshow(im[:,coronal_slice,:],cmap='gray', aspect= coronal_asp)
  ax[1].axis('off')
  ax[1].set_title('Coronal')

  # Show the specified slice on the sagittal plane with 'gray' color-map
  # and sagittal aspect ratio.
  ax[2].imshow(im[:,:,sagittal_slice], cmap='gray', aspect= sagittal_asp)
  ax[2].axis('off')
  ax[2].set_title('Sagittal')
