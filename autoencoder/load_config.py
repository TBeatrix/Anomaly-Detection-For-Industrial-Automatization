import json
import torch
import piq

class Config:

    def __init__(self):
        # Load the config file
        with open('config.json', 'r') as f:
            config = json.load(f)
            # Access specific parameters
            train_params = config["train_params"]
            self.need_train = train_params["need_train"]
            self.need_eval = train_params["need_eval"]
            self.eval_dir = train_params["eval_dir"]
            self.need_clustering = train_params["need_clustering"]
            self.model_type = train_params["encoder_model_type"]
            self.exp_name = train_params["exp_name"]
            criterion_name = train_params["criterion"]
            if criterion_name == "MSE": 
                 self.criterion = torch.nn.MSELoss()
            elif criterion_name == "SSIM":
                 self.criterion = piq.SSIMLoss( data_range=255)
            else:
                print("Invalid Criterion param!")

            self.optimizer_name = train_params["optimizer"]
            
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.device = device

            self.learning_rate = train_params["learning_rate"]
            self.batch_size = train_params["batch_size"]
            self.num_epochs = train_params["num_epochs"]

            dataset_params = config["dataset_params"]
            self.img_dir = dataset_params["dataset_path"]
            self.image_size = dataset_params["image_size"]
            print("Learning Rate:", self.learning_rate)
            print("Batch Size:", self.batch_size)
            print("Model type: ", self.model_type)