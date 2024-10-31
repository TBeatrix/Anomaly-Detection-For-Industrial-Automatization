import csv
from model_train import  ResNetAutoencoder,  save_reconstucted_test_images
from dataset import CustomImageDataset
import torch
import os
from torchvision.io import read_image
from torchvision.utils import save_image
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import piq
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
import matplotlib
import scipy.ndimage
from skimage.metrics import structural_similarity as ssim


class CustomImageDatasetForScores(Dataset):
        def __init__(self, img_paths, scores, transform, final_transform):
            self.img_paths = img_paths
            self.transform = transform
            self.final_transform = final_transform
            self.scores = scores

        def __len__(self):
            return len(self.img_paths)

        def __getitem__(self, idx):
            image = cv2.imread(self.img_paths[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.transform:
                augmented = self.transform(image=image)
           
                image = augmented['image']
                image = self.final_transform(image)
 
            return image, [], self.scores[idx], 

def load_and_eval_model( test_loader,  A_base_transforms, final_transform, config):
 
    torch.manual_seed(624)

     # Model init
    device = torch.device(f"cuda:{config['train_params']['device']}" if torch.cuda.is_available() else "cpu")
    autoencoder = ResNetAutoencoder(config["train_params"]["model_type"]).to(device)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    autoencoder.load_state_dict(torch.load(f"{config['result_dir']}/model_weights_new.pth"))
  
    # Set the model to evaluation mode
    autoencoder.eval()

    eval_test_dataset(autoencoder, test_loader, device, config, A_base_transforms, final_transform)
    
    for dir_name in config["eval_params"]["eval_datasets"]:
        eval_dataset_from_directory(dir_name,  A_base_transforms, final_transform, config, autoencoder, device)

    for score_type in ["SSIM", "MSE", "MS_SSIM", "Thresholded_SSIM", "Patched_SSIM", "Patched_MS_SSIM"] :
        save_diagrams( config, score_type,  threshold = 75, fold = False)



def eval_test_dataset(autoencoder, test_loader, device, config, A_base_transforms, final_transform):
    scores, score_names = get_anomaly_scores(autoencoder, test_loader, device)

    for score_dict, name in zip(scores, score_names):

        filename = f"{config['result_dir']}/{name}/test.csv"
        # Writing results to csv file
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Image name', 'Anomaly score'])  
            for key, value in score_dict.items():
                writer.writerow([key, value.item()])


    ssim_scores = scores[0]
    best_scores =  {k: ssim_scores[k] for k in list(ssim_scores)[:15]}
    worst_scores = {k: ssim_scores[k] for k in list(ssim_scores)[-15:]}

    best_images = CustomImageDatasetForScores(list(best_scores.keys()), list(best_scores.values()) , A_base_transforms, final_transform)
    best_images_loader = DataLoader(best_images, batch_size=16, shuffle=False)

    worst_images = CustomImageDatasetForScores(list(worst_scores.keys()), list(worst_scores.values()), A_base_transforms, final_transform)
    worst_images_loader = DataLoader(worst_images, batch_size=16, shuffle=False)

    images, _, scores = next(iter(best_images_loader))
    for i, image in enumerate(images):
        save_image(image, f'{config["result_dir"]}/best_scores/{i}_{scores[i]}.png')

    images, _, scores = next(iter(worst_images_loader))
    for i, image in enumerate(images):
        save_image(image, f'{config["result_dir"]}/worst_scores/{i}_{scores[i]}.png')


def eval_dataset_from_directory(dir_name, A_base_transforms, final_transform, config, autoencoder, device):
    image_paths = []
    for dirpath, _,files in os.walk(dir_name):
        for  file in files:
            if file.lower().endswith(('.png', '.jpg')):
                image_paths.append(os.path.join(dirpath, file))

    anomaly_images = CustomImageDataset(image_paths, A_base_transforms, None, final_transform)
    anomaly_images_loader = DataLoader(anomaly_images, batch_size=config["train_params"]["batch_size"], shuffle=False)
    
    scores, score_names = get_anomaly_scores(autoencoder, anomaly_images_loader, device)
    test_images, _, test_image_masks, _ = next(iter(anomaly_images_loader))
    save_reconstucted_test_images(autoencoder, test_images, test_image_masks,  config, f"{dir_name.split('/')[-2]}_{dir_name.split('/')[-1]}", device)

    for score_dict, name in zip(scores, score_names):
        filename = f"{config['result_dir']}/{name}/{dir_name.split('/')[-2]}_{dir_name.split('/')[-1]}.csv"
        # Writing to csv file
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Image name', 'Anomaly score'])  
            for key, value in score_dict.items():
                writer.writerow([key, value.item()])

    

def save_diagrams( config, criterion, threshold = 75, fold = False, num_folds=5):
    # Collect values
    scores = {}
    test_dataset_idx = 0
    # If there is only one fold - do not use cross validation
    if not fold:
        result_dir =  config["result_dir"]
        file_dir = f'{result_dir}/{criterion}'
        for dirpath, _, files in os.walk(file_dir):
            for i, f in enumerate(files):
                
                file_name = os.path.join(dirpath, f)
                filename_end = f.split('.')[-2].split("_")[-1]
                if filename_end == "test":
                    test_dataset_idx = i          
            
                # Load CSV 
                with open(file_name, mode='r') as file:
                    csv_reader = csv.reader(file)
                    next(csv_reader)
                    values = []
                    for row in csv_reader:
                        values.append(float(row[-1]))
                    scores[f.split('.')[0]] = values

    else:
       # aggregate the results of the different folds
        result_dir = "/".join(config['result_dir'].split('/')[:-1])
        scores_in_folds = {}
       
        for fold in range(num_folds):
            file_dir = f'{result_dir}/fold_{fold+1}/{criterion}'
            for i, (dirpath, _, files) in enumerate(os.walk(file_dir)):
                for f in files:
                    scores_in_folds[f.split('.')[0]] = [] 

        for fold in range(num_folds):
            
            file_dir = f'{result_dir}/fold_{fold+1}/{criterion}'
           
            test_dataset_idx = 0
            for dirpath, _, files in os.walk(file_dir):
                for i, f in enumerate(files):
                    if f.split('.')[-1] == "csv":
                        file_name = os.path.join(dirpath, f)
                        filename_end = f.split('.')[-2].split("_")[-1]
                        if filename_end == "test":
                            test_dataset_idx = i          
                    
                        # Load CSV 
                        with open(file_name, mode='r') as file:  
                            csv_reader = csv.reader(file)
                            next(csv_reader)
                            values = []
                            for row in csv_reader:
                                values.append(float(row[-1]))
                            scores_in_folds[f.split('.')[0]].append(values)


        
        for i, (name, lists) in enumerate(scores_in_folds.items()):
            if i == test_dataset_idx:
                average_list = [value for sublist in lists for value in sublist]
            else:    
                # Compute the average list for the current name
                average_list = [sum(values) / len(values) for values in zip(*lists)]
            # Add the new averaged list to the result dictionary
            scores[name] = average_list

    # Boxplot data
    data = [scores[key] for key in scores.keys()]
    labels = list(scores.keys())
    # Colormap 
    cmap = matplotlib.colormaps['viridis']  
    colors = cmap(np.linspace(0, 1, len(labels))) 
    plt.figure(figsize=(8, 6))
    plt.title(f"{criterion} boxplot ")
 
    box = plt.boxplot(data, patch_artist=True, labels=labels)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    for median in box['medians']:
        median.set_color('black')
               
    Q1 = np.percentile(data[test_dataset_idx], 25)  
    Q3 = np.percentile(data[test_dataset_idx], 75) 

    plt.axhline(y=Q1, color='gray', linestyle='--', linewidth=1)
    plt.axhline(y=Q3, color='gray', linestyle='--', linewidth=1)
    plt.ylabel('Anomaly score')
    plt.xlabel('Image groups')
    plt.grid(axis='y')
    plt.xticks(rotation=90) 

    plt.savefig(f"{result_dir}/{criterion}_boxplot.png")

    #set Threshold
    threshold = np.percentile(data[test_dataset_idx], threshold)

    counts_below = {}  
    counts_above = {}  

    for key, values in scores.items():
      
        below = sum(1 for value in values if value < threshold) 
        above = sum(1 for value in values if value >= threshold)
        
        total_count = len(values)
        counts_below[key] = round((below / total_count) * 100, 3) 
        counts_above[key] = round((above / total_count) * 100, 3)

    # Visualization with histogram 
    labels = list(scores.keys())
    below_percent = list(counts_below.values())
    above_percent = list(counts_above.values())
    x = np.arange(len(labels)) 
    plt.figure(figsize=(8, 6))

    for i in range(len(labels)):
        if i == test_dataset_idx:  # If it is the  validation dataset
            plt.bar(x[i], below_percent[i], label=f'Below {threshold}', color="lightblue")
            plt.bar(x[i], above_percent[i], bottom=below_percent[i], label=f'Above {threshold}', color="lightyellow")
        else:
            plt.bar(x[i], below_percent[i], label=f'Below {threshold}' if i == 0 else "", color="red")
            plt.bar(x[i], above_percent[i], bottom=below_percent[i], label=f'Above {threshold}' if i == 0 else "", color="lightgreen")
        plt.text(x[i], below_percent[i] + above_percent[i] / 2, f'{(above_percent[i]):.1f}%', ha='center', va='center', color='black', fontsize=10)
        
    plt.xticks(x, labels)
    plt.xlabel('Measurements')
    plt.ylabel('Percentage (%)')
    plt.title(f'Percentage of Values Above and Below {round(threshold,4)}')
    plt.xticks(rotation=90) 

    plt.savefig(f"{result_dir}/{criterion}_histogram.png")



def get_anomaly_scores(model, val_data, device):
    print("Anomaly scores ...")
    model.eval()
    scores_ssim = {}
    scores_mse = {}
    scores_ms_ssim = {}
    scores_thresholded_ssim = {}
    scores_patched_ssim = {}
    scores_patched_ms_ssim = {}

    with torch.no_grad(): 
        for data_x, data_y, masks, path in val_data:
            for i, (img, mask) in enumerate(zip(data_x, masks)): 
                img = img.unsqueeze(0).to(device)
                img, mask = img.to(device), mask.to(device)
                
                # Forward pass
                output = model(img)

                MSE_loss = torch.nn.MSELoss()
                # Compute different types of errors
                SSIM_loss = piq.SSIMLoss( data_range=1)

                loss_mse = MSE_loss(output *mask, img * mask)
                loss_ssim = 1- piq.ssim(output *mask, img * mask)
                ms_ssim = 1 - piq.multi_scale_ssim(output *mask, img * mask, data_range=1)

                # thresholded SSIM
                thresholded_ssim = compute_thresholded_ssim_score(output * mask, img * mask)

                # Patched - masked SSIM
                patched_ssim = compute_sliding_window_ssim(output * mask, img * mask, ssim_type ="SSIM")

                # Patched - masked - MS SSIM
                patched_ms_ssim = compute_sliding_window_ssim(output * mask, img * mask, ssim_type ="MS-SSIM")

                scores_ssim[path[i]] = loss_ssim
                scores_mse[path[i]] = loss_mse
                scores_ms_ssim[path[i]] = ms_ssim
                scores_thresholded_ssim[path[i]] = thresholded_ssim
                scores_patched_ssim[path[i]] = patched_ssim
                scores_patched_ms_ssim[path[i]] = patched_ms_ssim    

    score_names = ["SSIM", "MSE", "MS_SSIM", "Thresholded_SSIM", "Patched_SSIM", "Patched_MS_SSIM"]        
    score_dicts = [scores_ssim, scores_mse, scores_ms_ssim, scores_thresholded_ssim, scores_patched_ssim, scores_patched_ms_ssim ]
    for i, scores in enumerate(score_dicts):
        score = dict(sorted(scores.items(), key=lambda item: item[1]))
        score_dicts[i] = score
    
    return score_dicts, score_names



def compute_thresholded_ssim_score(output, img):
    # Calculate SSIM map
    ssim_score, ssim_map = ssim(img.squeeze().permute(1, 2, 0).cpu().detach().numpy(), output.squeeze().permute(1, 2, 0).cpu().detach().numpy(), data_range=1, full=True, multichannel=True, channel_axis=-1)
    # avrage out chanels
    ssim_map  = np.mean(ssim_map, axis = -1, keepdims=True)
    # Create nose mask
    smoothed_ssim_map = scipy.ndimage.median_filter(ssim_map, size=5)
    thresholded_ssim_map = np.copy(smoothed_ssim_map)
    thresholded_ssim_map[smoothed_ssim_map > 0.9 ] = 1.0  # Suppress small differences
    thresholded_ssim_map[smoothed_ssim_map < 0.1] = 0.0

     # Combine masks
    masked_ssim_map = (1 -thresholded_ssim_map)

    # SSIM score
    masked_ssim_score = np.mean(masked_ssim_map) 
    return masked_ssim_score


def compute_sliding_window_ssim(original_img, reconstructed_img, ssim_type= "SSIM", window_size=128, step_size=64):
   
    assert original_img.shape == reconstructed_img.shape
    
    # Calculate the number of sliding windows
    img_height = img_width = 256
    patches_per_row = (img_width - window_size) // step_size + 1
    patches_per_col = (img_height - window_size) // step_size + 1
    
    # Initialize a list to store SSIM scores for each patch
    ssim_scores = []

    # Loop over patches using a sliding window
    for row in range(0, patches_per_col * step_size, step_size):
        for col in range(0, patches_per_row * step_size, step_size):
            # Extract the patch from both images
            original_patch = original_img[:,:,row:row+window_size, col:col+window_size]     
            reconstructed_patch = reconstructed_img[:,:,row:row+window_size, col:col+window_size]
            if ssim_type == "SSIM":
                # Compute the SSIM for the current patch
                patch_ssim = compute_thresholded_ssim_score(original_patch, reconstructed_patch)
            elif ssim_type == "MS-SSIM":
                # I do not use thresholds here
                patch_ssim = 1 - piq.multi_scale_ssim(original_patch, reconstructed_patch, kernel_size=7,  data_range=1)
            else:
                print("Wrong SSIM type!")
            
            # Append the SSIM score
            ssim_scores.append(patch_ssim)

    anomaly_score = max(ssim_scores)
    
    return anomaly_score
