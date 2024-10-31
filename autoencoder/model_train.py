from models import *
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import os
from torchvision.io import read_image
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F
from skimage.metrics import structural_similarity as ssim
import piq
import scipy.ndimage
import random
import cv2
import matplotlib.pyplot as plt
import csv


class ResNetAutoencoder(nn.Module):
    def __init__(self, model_type= "resnet_50"):
        super(ResNetAutoencoder, self).__init__()
        if model_type == "resnet_50":
            self.encoder = ResNet50_Encoder()
            self.decoder = ResNet50_Decoder()
        elif model_type == "resnet_18":
            self.encoder = ResNet18_Encoder()
            self.decoder = ResNet18_Decoder()
        elif model_type == "densenet":
            self.encoder = DenseNet_Encoder()
            self.decoder = DenseNet_Decoder()
        else:
            print("Wrong model type")

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_model(model, config, train_data, val_data, test_images, test_image_masks, training_logger, device):
    if config["optimizer_name"] == "Adam":
         optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    else:
        "The Optimizer name is not valid!"
    val_losses = []
    train_losses = []
    training_logger.info("Training started ...")

    criterion_name = config["criterion"]
    if criterion_name == "MSE":
        criterion = torch.nn.MSELoss()
    elif criterion_name == "SSIM":
        criterion = piq.SSIMLoss( data_range=1)
    else:
        print("Invalid Criterion param!")
  
    for epoch in range(config["num_epochs"]):
        model.train()
        epoch_loss = 0
        for x_img, y_img, mask, _ in train_data :
            
            x_img = x_img.to(device)
            y_img = y_img.to(device)
            mask = mask.to(device)
            # Forward pass
            outputs = model(x_img)       
            loss = criterion(outputs * mask, y_img * mask)
            epoch_loss += loss
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        avg_train_loss = epoch_loss / len(train_data)
        avg_val_loss = eval_model(model, config, val_data, device, criterion)   
        val_losses.append(avg_val_loss.item())
        train_losses.append(avg_train_loss.item())

        training_logger.info(f'Epoch [{epoch+1}/{config["num_epochs"]}], Train Loss: {avg_train_loss.item():.8f}, Validation Loss: {avg_val_loss.item():.8f}')
        if ((epoch) % 50 ==0):
            save_reconstucted_test_images(model, test_images, test_image_masks, config, epoch, device)
    
    visualize_loss(config["num_epochs"], train_losses, val_losses, config["result_dir"])


def visualize_loss(num_epochs, train_losses, val_losses, result_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_epochs), train_losses, label='Training Loss', color='red')
    plt.plot(range(num_epochs), val_losses, label='Validation Loss', color='blue')
    plt.title('Training and Validation Losses Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    # # Save the plot 
    plt.savefig(f'{result_dir}/val_and_train_loss_plot')


def eval_model(model, config, val_data, device, criterion):
    #print("Evaluating ...")
    model.eval()
    epoch_loss = 0
    with torch.no_grad(): 
        for img_x, img_y, mask, _ in val_data: 
            img_x, img_y, mask = img_x.to(device), img_y.to(device), mask.to(device)             
            # Forward pass
            outputs = model(img_x)
            loss = criterion(outputs *mask, img_y * mask)
            epoch_loss += loss
    avg_val_loss = epoch_loss / len(val_data)
    return avg_val_loss



    

def save_reconstucted_test_images(model, test_images, test_image_masks, config, epoch = 0, device = 'cpu'):
    model.eval()
    with torch.no_grad(): 
        for data_id, (data, test_image_mask) in enumerate(zip(test_images, test_image_masks)):
           
            img_ = data.unsqueeze(0)  
            img, test_image_mask = img_.to(device), test_image_mask.to(device)
            output = model(img)    
            # apply mask
            output = output * test_image_mask
            img = img * test_image_mask

            figure, ax = plt.subplots(2,4)
            
            # Save original image   
            ax[0][0].set_title('Original image', fontsize=8) 
            original_image = data.numpy().transpose(1, 2, 0)
            original_image = (original_image * 255).astype(np.uint8)    
            ax[0][0].imshow(original_image)      
            ax[0][0].axis('off')

            # Save reconstructed image
            ax[0][1].set_title('Reconstructed image', fontsize=8)
            recon = output.squeeze().cpu().numpy().transpose(1, 2, 0)
            recon = (recon * 255).astype(np.uint8)    
            image = Image.fromarray(recon)
            ax[0][1].imshow(image)
            ax[0][1].axis('off')
        

            #Create different heatmaps
            # MAE
            MAE_difference = torch.abs(output - img)

            # Max
            ax[1][0].set_title('MAE - max', fontsize=8)
            difference_max = torch.max(MAE_difference, dim=1)[0].permute(1, 2, 0)
            difference_np = difference_max.cpu().detach().numpy()
            ax[1][0].imshow(difference_np, cmap='hot', interpolation='nearest')          
            ax[1][0].axis('off')

            # Mean
            ax[1][1].set_title('MAE - mean', fontsize=8)
            difference_mean = torch.mean(MAE_difference, dim=1).permute(1, 2, 0)
            difference_np = difference_mean.cpu().detach().numpy()
            ax[1][1].imshow(difference_np, cmap='hot', interpolation='nearest')          
            ax[1][1].axis('off')

            # Thresholded mean
            ax[1][2].set_title('MAE - thresholded mean', fontsize=8)
            threshold = torch.quantile(MAE_difference, 0.50) 
            difference = torch.where(MAE_difference > threshold, MAE_difference, torch.tensor(0.0))
            difference_mean = torch.mean(difference, dim=1).permute(1, 2, 0)
            difference_np = difference_mean.cpu().detach().numpy()
            ax[1][2].imshow(difference_np, cmap='hot', interpolation='nearest')          
            ax[1][2].axis('off')
           
            # SSIM
            # Simple SSIM
            original_image_np = img.squeeze().permute(1, 2, 0).cpu().detach().numpy()
            output_image_np = output.squeeze().permute(1, 2, 0).cpu().detach().numpy()
            ssim_score, ssim_map = ssim(original_image_np, output_image_np , data_range=1, full=True, multichannel=True, channel_axis=-1)
    
            # Normalize the SSIM map to [0, 1]
            ssim_min = ssim_map.min()
            ssim_max = ssim_map.max() 
            ssim_map = (ssim_map - ssim_min) / (ssim_max - ssim_min)
          
            # Avg over channel dim
            ax[0][2].set_title("SSIM", fontsize=8)
            ssim_map  = np.mean(ssim_map, axis = -1, keepdims=True)         
            ax[0][2].imshow(1-ssim_map, cmap='hot', interpolation='nearest')          
            ax[0][2].axis('off')    
    
            ax[0][3].set_title("Thresholded SSIM", fontsize=8)
            # Create noise masked SSIM heatmap    
            smoothed_ssim_map = scipy.ndimage.median_filter(ssim_map, size=5)
            # Apply thresholds
            thresholded_ssim_map = np.copy(smoothed_ssim_map)
            thresholded_ssim_map[smoothed_ssim_map > 0.85 ] = 1.0  
            thresholded_ssim_map[smoothed_ssim_map < 0.15] = 0.0
            masked_ssim_map = 1-thresholded_ssim_map
            ax[0][3].imshow(  masked_ssim_map, cmap='hot', interpolation='nearest')    


            ax[1][3].set_title("Blured, thresholded SSIM", fontsize=6)
            smoothed_ssim_map = scipy.ndimage.median_filter(ssim_map, size=9)
            # Apply thresholds
            thresholded_ssim_map = np.copy(smoothed_ssim_map)
            thresholded_ssim_map[smoothed_ssim_map > 0.75 ] = 1.0 
            thresholded_ssim_map[smoothed_ssim_map < 0.1] = 0.0
            masked_ssim_map = 1-thresholded_ssim_map
  
            ax[1][3].imshow(  masked_ssim_map, cmap='hot', interpolation='nearest')    

            # Save the image
            plt.savefig(f'{config["result_dir"]}/reconstructed_images/reconstructed_image_{epoch}_{data_id}.png', bbox_inches='tight', pad_inches=0)
            plt.close()