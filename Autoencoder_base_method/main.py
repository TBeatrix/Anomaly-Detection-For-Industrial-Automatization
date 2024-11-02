
# from clustering import cluster_hidden_representation
from utils import *
from load_config import Config
import torch
from model_train import *
from dataset import *
from augmentations import *
#from piq import ssim_loss
from model_eval import *

import random
import yaml

#Set random seed
torch.manual_seed(624)
random.seed(624) 

# Set config, logger and seed
config = yaml.safe_load(open("config.yaml"))
# Set folds
num_folds = config["train_params"]["num_of_folds"]
groups = create_groups(config["train_params"]["dataset_path"], num_folds)

# Create directories
result_dir = create_directories(config, num_folds)
config["result_dir"] = result_dir

# set logger
training_logger, config_logger = set_logger(result_dir)
train_config = config["train_params"]
train_config["result_dir"] = result_dir
eval_config = config["eval_params"]
cluster_config = config["clustering_params"]

# Save config file
with open(os.path.join(result_dir, "config.yaml"), "w") as file:
     yaml.safe_dump(config, file)


# Define augmentations
A_transforms_train, A_base_transforms, final_transform = set_augmentations(train_config)
# Set loss function
criterion_name = train_config["criterion"]
if criterion_name == "MSE": 
     criterion = torch.nn.MSELoss()
elif criterion_name == "SSIM":
     criterion = piq.SSIMLoss( data_range=1)
else:
     print("Invalid Criterion param!")

# Repeate for every fold
for i in range(num_folds):
     i = i+1
     config["result_dir"] = f"{result_dir}/fold_{i}"
     train_config["result_dir"] = f"{result_dir}/fold_{i}"
     print(f"Training fold: {i}")
     # Data acquisition
     # if there is only one fold --> Normal test-train-val split
     if num_folds == 1:
          train_dataset, val_dataset, test_dataset = create_splits(
            config["dataset_path"], A_transforms_train, A_base_transforms, final_transform)         
     # In case of cross-validation
     else:
         train_dataset, val_dataset, test_dataset = return_fold(i, groups, A_transforms_train, A_base_transforms, final_transform)
     
     val_loader, test_loader, train_loader = create_data_loaders(train_config, train_dataset, val_dataset, test_dataset, config_logger)


     # Training
     if train_config["need_train"]:
          # save test and train images from the first batch
          train_images_x, train_images_y, mask, _ = next(iter(train_loader))
          save_original_test_images(train_images_x, config['result_dir'], "train_batch")

          test_images, _, test_image_masks, _  = next(iter(test_loader))
          save_original_test_images(test_images, config['result_dir'], "test_batch")
            
          # Model init
          device = torch.device(f"cuda:{train_config['device']}" if torch.cuda.is_available() else "cpu")
          autoencoder = ResNetAutoencoder(train_config["model_type"]).to(device)
          
          train_model(autoencoder, train_config, train_loader, val_loader, test_images, test_image_masks, training_logger, device)
          test_loss = eval_model(autoencoder, train_config, test_loader, device, criterion)
          training_logger.info(f"Final loss on the test dataset: { test_loss.item()}")

          # Save the model state
          path = f"{config['result_dir']}/model_weights_new.pth"
          torch.save(autoencoder.state_dict(), path)

     if eval_config["need_eval"]:
          load_and_eval_model(test_loader, A_base_transforms, final_transform, config)

  
if eval_config["need_eval"]:
     print("Evaluate folds...")    
     for score_type in config["eval_params"]["score_types"]: 
          save_diagrams(train_config, score_type, threshold = 75, fold = True, num_folds=num_folds)