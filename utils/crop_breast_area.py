import os
import cv2
import numpy as np
# Set the input and output directories
input_dir = r"D:\IAAA_CMMD\manifest-1616439774456\test_dcm\dicom"
output_dir = r"D:\IAAA_CMMD\manifest-1616439774456\test_dcm\dicom_out/"


# Loop through all image files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".png"):
        # Read the image file
        img_path = os.path.join(input_dir, filename)
        im = cv2.imread(img_path)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
        # threshold 
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        hh, ww = thresh.shape

        # make bottom 2 rows black where they are white the full width of the image
        thresh[hh-3:hh, 0:ww] = 0
        # get bounds of white pixels
        white = np.where(thresh==255)
        xmin, ymin, xmax, ymax = np.min(white[1]), np.min(white[0]), np.max(white[1]), np.max(white[0])
        print(xmin,xmax,ymin,ymax)
        # crop the image at the bounds adding back the two blackened rows at the bottom
        crop = im[ymin:ymax+3, xmin:xmax]
        # save resulting masked image
        #cv2.imwrite(os.path.join(output_dir, filename), thresh)
        cv2.imwrite(os.path.join(output_dir, filename), crop)
        # display result
        #cv2.imshow("thresh", thresh)
        #cv2.imshow("crop", crop)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
            
