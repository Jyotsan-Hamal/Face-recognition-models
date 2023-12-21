'''
This code rename the dataset folder for training the model
'''


import os

def rename_files(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for i, file_name in enumerate(files, start=1):
            # Generate the new file name
            new_file_name = f"image{i}.jpg"
            
            # Construct the full path of the file
            old_file_path = os.path.join(root, file_name)
            new_file_path = os.path.join(root, new_file_name)
            
            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {old_file_path} -> {new_file_path}")

# Replace 'dataset/' with the actual path to your dataset folder
dataset_path = 'dataset/'

rename_files(dataset_path)
