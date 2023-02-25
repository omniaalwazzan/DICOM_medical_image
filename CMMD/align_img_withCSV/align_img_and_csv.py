

import csv
import os
import panda as pd
import shutil

# Path to the folder containing the images
image_folder_path = r"D:\IAAA_CMMD\manifest-1616439774456\test" 
# Path to the CSV file containing the IDs
met_csv = r"D:/IAAA_CMMD/manifest-1616439774456/test_2/meta_sample.csv"
# Path to the CSV file to write the new file names to
output_csv_path = r"D:\IAAA_CMMD\manifest-1616439774456\test_2\out_tes.csv"
destination_folder = r'D:\IAAA_CMMD\manifest-1616439774456\test_2'
# Open the CSV file and read the IDs

file_name = os.listdir(image_folder_path)

"""
this for loop will loop over the metadata csv file and compare ids in csv with folder name 
if ids match, then move images to one folder and write their names to a csv files  
"""

for k in range(5):
    with open(met_csv, "r") as metadata:
        # Create a CSV reader object
        metadata_reader = csv.reader(metadata)
        
        # Skip the header ids
        next(metadata_reader)
        column1 = [row[0] for row in metadata_reader]
        with open(output_csv_path, "a", newline="") as output_csv:
            csv_writer = csv.writer(output_csv)            
            #csv_writer.writerow([column1[k]])
            if file_name[k] in column1:
                print(file_name[k])
                path1 = image_folder_path +'/'+ file_name[k]
                path2 = path1+ '/'+ next(os.walk(path1))[1][0]
                path3 = path2+ '/'+ next(os.walk(path2))[1][0]
                img_name = os.listdir(path3)
                #print(img_name)
                for j in range (len(img_name)):
                    file_path = os.path.join(path3, img_name[j])
                    
                    print(file_path)
                    if os.path.isfile(file_path):
                        
                        file_id = os.path.splitext(file_path)[0]
                        new_name = os.path.basename(file_id) + '.dcm'
                        os.makedirs(destination_folder, exist_ok=True)
                        #new_file_name = f"{img_name[j]}_{new_name}"
                        new_file_name = f"{file_name[k]}_{new_name}" 
                        new_file_path = os.path.join(destination_folder, new_file_name)
                        shutil.copy(file_path, new_file_path)
                        csv_writer.writerow([file_name[k],new_file_name, file_id])

                
            
    # Read the IDs from the second column and store them in a list
    
"""
After moving all images from subfolders, merge the new out csv file containing new img names with the clinical data 
so the resulting csv table will have image name, and other features, by doing this, we will be able to make an easy dataloder 

"""

df1 = pd.read_csv(r"D:/IAAA_CMMD/manifest-1616439774456/test_2/meta_sample.csv")
# Path to the CSV file to write the new file names to
df2 = pd.read_csv(r"D:\IAAA_CMMD\manifest-1616439774456\test_2\out_tes.csv")

           
new = pd.merge(df1, df2, how="inner", on=["ID1"])
new.to_csv(r"D:\IAAA_CMMD\manifest-1616439774456\test_2\out_merg.csv")
