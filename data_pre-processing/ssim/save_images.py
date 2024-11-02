
import pandas as pd
import shutil
import os


def copy_files_from_csv(csv_path, source_dir, target_dir):
    
    df = pd.read_csv(csv_path)
    file_names = df.iloc[:, 0].tolist()
    # Copy each file
    for file_name in file_names:
        source_path = os.path.join(source_dir, file_name)
        target_path = os.path.join(target_dir, file_name)

        # Check if the file exists in the source directory
        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)
         
        else:
            print(f"File not found: {file_name}")


if __name__ ==  '__main__': 

    # Example usage
    csv_file_path = 'approved_images.csv'
    source_directory = '../../cropp_images/results'
    target_directory = 'final_images'
    copy_files_from_csv(csv_file_path, source_directory, target_directory)
