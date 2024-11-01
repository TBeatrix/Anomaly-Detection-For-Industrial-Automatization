import os
import glob
import shutil


def modify_label_content(line):
    parts = line.strip().split()
    modified_line = ' '.join(parts[1:-1]) + ' open_box ' + parts[-1]
    return modified_line
 
# Directories
if __name__ == "__main__":
    directory =  "Tekercs2_kicsomagolas_maradek"  #"Tekercs_kicsomagolas_20230525" # "Nyak_kicsomagolas_20230209"
    #"Tekercs2_kicsomagolas_20230605" #"Tekercs_kicsomagolas_20230605" # "Nyak_kicsomagolas_20230220"
    images_dir = f'images_{directory}'
    labels_dir = f'yolov5_obb/runs/detect/labels_images_{directory}/labels'
    result_images_dir = f'result/images/open_box_images_{directory}'
    result_labels_dir = f'result/labels/open_box_labels_{directory}'

    # Create new directories
    os.makedirs(result_images_dir, exist_ok=True)
    os.makedirs(result_labels_dir, exist_ok=True)

    # List files in both directories
    image_files = glob.glob(os.path.join(images_dir, '*.png'))
    label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
    # Copy images that has a matching label

    # Extract just the file names without extension from labels
    label_basenames = [os.path.splitext(os.path.basename(f))[0] for f in label_files]
    # Copy matching images
    for image_file in image_files:
        image_basename = os.path.splitext(os.path.basename(image_file))[0]
        if image_basename in label_basenames:
            shutil.copy(image_file, result_images_dir)

    # Modify label files and copy


    for label_file in label_files:
        with open(label_file, 'r') as file:
            lines = file.readlines()
        modified_lines = [modify_label_content(line) for line in lines]
        with open(os.path.join(result_labels_dir, os.path.basename(label_file)), 'w') as file:
            file.write('\n'.join(modified_lines))


