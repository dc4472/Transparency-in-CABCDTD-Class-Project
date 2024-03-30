###################################################
#Calculate the cropped area per 598 by 598 pixels #
#Calculate the full image size                    #
#Add pathology information to the results         #
###################################################

#load the modules
import os
import re
import pandas as pd
from PIL import Image
import numpy as np

###################################################
#Calculate the cropped area per 598 by 598 pixels #
###################################################

# Create an empty DataFrame
df_test = pd.DataFrame(columns=['name', 'file_path', 'area_percentage'])

# Replace 'path_to_folder' with the actual path to your folder containing images
folder_path = '/content/test_598_same'

for filename in os.listdir(folder_path):
    if filename.endswith(".png"):  # Check only PNG files, adjust if needed
        image_path = os.path.join(folder_path, filename)
        results, image, nonzero_area_percentage = process_image_and_extract_groups(image_path)
        if results and image and nonzero_area_percentage:
            df_test = df_test.append({'name': results, 'file_path': image_path, 'area_percentage': nonzero_area_percentage}, ignore_index=True)


#define the function
def process_image_and_extract_groups(image_path):
    # Define the regex pattern
    pattern = r'Mass-Training_P_(\d+)_(\w+)_(\w+)_' #for training dataset, make sure you pick the proper one
    pattern = r'Mass-Test_P_(\d+)_(\w+)_(\w+)_' #for training dataset, make sure you pick the proper one


    match = re.search(pattern, image_path)
    if match:
        number = match.group(1)
        left = match.group(2)
        cc = match.group(3)
        result = f'Mass-Test_P_{number}_{left}_{cc}'

        # Open the image
        image = Image.open(image_path)

        # Convert the grayscale image to a NumPy array
        image_array = np.array(image)

        # Calculate the non-zero area
        nonzero_pixels = np.count_nonzero(image_array)
        total_pixels = image_array.size
        nonzero_area_percentage = (nonzero_pixels / total_pixels) * 100

        return result, image, nonzero_area_percentage
    else:
        print(f"No match found for file: {image_path}")
        return None, None, None

print("Done")

#####################################################
# merge the area info in both train and test images #
# add the pathology information as well             #
#####################################################

# Merge the DataFrames using pd.concat()
merged_df = pd.concat([df, df_test]) #df and df_test contain area info for train and test cropped images, respectively.
sorted_merged_df = merged_df.sort_values(by='name')
sorted_merged_df = sorted_merged_df.reset_index(drop=True)

#read in the pathology information
pathology = pd.read_csv('/content/content/all_mass_pathology.csv')
sorted_pathology = pathology.sort_values(by='full_name')
sorted_pathology = sorted_pathology.reset_index(drop=True)

# creat a column pathology in sorted_merged_df
sorted_merged_df['pathology'] = None

# copy the pathology if meet the requirement
def copy_pathology(row):
    for name in sorted_pathology['full_name']:
        if name in row['name']:
            return sorted_pathology.loc[sorted_pathology['full_name'] == name, 'pathology'].values[0]
    return None

sorted_merged_df['pathology'] = sorted_merged_df.apply(copy_pathology, axis=1)

# save the output files
sorted_merged_df.to_csv('/content/content/598_percentage_all.csv')


#################################
# calculate the full image size #
# add pathology information     #
#################################

#put all full images in side the folder /content/full_train_needed
# Folder containing the images
folder_path = '/content/content/full_train_needed'

# Initialize lists to store image path, width, and height information
image_path_list = []
width_list = []
height_list = []

# Loop through each file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.png'):  # Check if the file is a PNG image
        # Construct the full path to the image
        image_path = os.path.join(folder_path, file_name)

        # Extract the part between 'full_' and '.png' in the filename
        image_name = file_name.split('full_')[1].split('.png')[0]

        # Open the image using PIL
        image = Image.open(image_path)

        # Get width and height
        width, height = image.size

        # Append image path, width, and height to lists
        image_path_list.append(image_name)
        width_list.append(width)
        height_list.append(height)

# Create a DataFrame from the lists
data = {'Subject_ID': image_path_list, 'Width': width_list, 'Height': height_list}
df = pd.DataFrame(data)


# creat a column pathology in df
df['pathology'] = None

# copy the pathology if meet the requirement
def copy_pathology(row):
    for name in sorted_pathology['full_name']:
        if name in row['Subject_ID']:
            return sorted_pathology.loc[sorted_pathology['full_name'] == name, 'pathology'].values[0]
    return None

df['pathology'] = df.apply(copy_pathology, axis=1)
df_sort = df.sort_values(by='Subject_ID')

#save the output
df_sort.to_csv('/content/metadata/heaght_width_FULL.csv')



















