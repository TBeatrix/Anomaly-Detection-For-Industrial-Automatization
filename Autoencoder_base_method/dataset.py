from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import os
from torchvision.io import read_image
from torchvision.utils import save_image
import random
import numpy as np
from torch.utils.data import DataLoader, random_split
import torchvision.utils as vutils

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, base_transform=None, transform=None, final_transform = None):
      
        self.transform = transform
        self.final_transform = final_transform
        self.base_transform = base_transform
        self.image_paths = image_paths
        self.black_masks = self.get_mask()

    def __len__(self):
        return len(self.image_paths)

    # mask for the black border of the images
    def get_mask(self):
        masks = []
        for img in self.image_paths:
            image = cv2.imread(img)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            black_pixel_mask = np.any(image != 0, axis=-1).astype(int)  
            masks.append(black_pixel_mask)
        return masks

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
           
        augmented = self.base_transform(image = image, mask=self.black_masks[idx])
        base_image = augmented['image'] 
        mask = augmented['mask'] 
        
        #mask = mask[np.newaxis, ...] # Expand dimensions to match (256, 256, 1) (intead of to_tensor transorm) 
        if self.transform:                 
            augmented = self.transform(image=base_image)
            augmented_image = augmented['image']
        else:
            augmented_image = base_image
     
        x_image = self.final_transform(augmented_image)
        mask = self.final_transform(mask)
        y_image = self.final_transform(base_image)   
        
        return x_image, y_image, mask, self.image_paths[idx] 


# Sort the days into different folds with as same number of images as possible
def create_groups(img_dir,  num_of_folds = 5):
    days = {}
    for dir_name in img_dir:
        for dirpath, _, files in os.walk(dir_name):
            for f in files:
                day =f.split('-')[0]
                if day not in days.keys():
                    days[day]= []
                
                days[day].append(os.path.join(dirpath, f))

    print("Number of days:", len(days.keys()))
    num_of_images = 0
    for i, key in enumerate(days.keys()):
        num_images = len(days[key])
        print(f"day {i+1}. has {num_images} images" )
        num_of_images += num_images
    print("Number of all images:", num_of_images)
    print("Ideal number of images in one fold:", num_of_images/num_of_folds)
    sorted_days = sorted(days.items(), key=lambda x: len(x[1]), reverse=True)

    groups = {}
    group_sums = [0] * num_of_folds 

    # Assign each day to the group with the smallest total
    for day, images in sorted_days:
        smallest_group = group_sums.index(min(group_sums))
        if smallest_group not in groups.keys():
            groups[smallest_group] = []
        groups[smallest_group].extend(images)
        group_sums[smallest_group] += len(images)

    for group, days in groups.items():
        print(f"Group {group + 1}: Days {len(days)}")
    return groups


def return_fold(num_of_fold, groups, train_transform, base_transform, final_transform):
    # Train data
    train_image_paths = []
    for index, images in groups.items():
        if index == num_of_fold-1:
            test_image_paths = images
        else:
            train_image_paths.extend(images)
   
    #Shuffle the images!
    random.shuffle(train_image_paths) 
    random.shuffle(test_image_paths) 
    total_images = len(train_image_paths) + len(test_image_paths)
    print("Total images num", total_images)
    
    train_dataset = CustomImageDataset(train_image_paths, base_transform = base_transform, transform=train_transform, final_transform = final_transform) + CustomImageDataset(train_image_paths, base_transform = base_transform, transform=train_transform, final_transform = final_transform) + \
    CustomImageDataset(train_image_paths, base_transform = base_transform, transform=train_transform, final_transform = final_transform)   
    test_dataset = CustomImageDataset(test_image_paths, base_transform= base_transform, transform=None, final_transform = final_transform)

    return train_dataset, test_dataset, test_dataset

    
# Create random train-test split
def create_splits(img_dir, train_transform, base_transform, final_transform, train_size=0.7, val_size=0.15):
    image_paths = []
    for dir_name in img_dir:
        for dirpath, _, files in os.walk(dir_name):
            for file in files:
                if file.lower().endswith(('.png', '.jpg')):
                    image_paths.append(os.path.join(dirpath, file))

    random.shuffle(image_paths)  # Shuffle the images
    total_images = len(image_paths)
    print("Total images num", total_images) 
 
    train_end = int( train_size * total_images)
    val_end = train_end + int(val_size * total_images)
    print("train end", train_end)
    print("val end", val_end)
    train_files = image_paths[:train_end]
    val_files = image_paths[train_end:val_end]
    test_files = image_paths[val_end:]
    
    train_dataset = CustomImageDataset(train_files, base_transform = base_transform, transform=train_transform, final_transform = final_transform) + CustomImageDataset(train_files, base_transform = base_transform, transform=train_transform, final_transform = final_transform) + \
    CustomImageDataset(train_files, base_transform = base_transform, transform=train_transform, final_transform = final_transform)   
    val_dataset = CustomImageDataset(val_files, base_transform= base_transform, transform=None, final_transform = final_transform)
    test_dataset = CustomImageDataset(test_files, base_transform= base_transform, transform=None, final_transform = final_transform)

    return train_dataset, val_dataset, test_dataset
   

def create_data_loaders(config, train_dataset, val_dataset, test_dataset, config_logger):
   
    config_logger.info(f'Train dataset size: {len(train_dataset)}')
    config_logger.info(f'Validation dataset size: {len(val_dataset)}')
    config_logger.info(f'Test dataset size: {len(test_dataset)}')

    # Create a DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, prefetch_factor=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # Testing the acquisition
    first_batch_images, _, _, _ = next(iter(train_loader))  
    print(first_batch_images.shape)

    config_logger.info(f'Dataset: {config["dataset_path"]}')
    config_logger.info(f'Train batch num: {len(train_loader)}')
    config_logger.info(f'Validation batch num: {len(val_loader)}')
    config_logger.info(f'Test batch num: {len(test_loader)}')
    return val_loader, test_loader, train_loader


def save_original_test_images(test_images, result_dir, type_):    

    grid = vutils.make_grid(test_images, nrow=8)
    grid = grid.permute(1, 2, 0)  # Permute to HxWxC for plotting
    grid = grid.numpy() 
    plt.figure(figsize=(12, 12))  
    plt.imshow(grid)
    plt.axis('off')
    plt.savefig(f'{result_dir}/original_images/{type_}_image_grid.png')
    plt.close()