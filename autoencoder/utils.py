import logging
import glob
import os
import shutil

def set_logger(result_dir):
    config_logger = logging.getLogger('ConfigLogger')
    training_logger = logging.getLogger('TrainingLogger')

    config_logger.setLevel(logging.INFO)
    training_logger.setLevel(logging.INFO)
    # Format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Config data
    file_handler = logging.FileHandler(f'{result_dir}/exp_config.txt')
    file_handler.setFormatter(formatter)
    config_logger.addHandler(file_handler)

    # Train datai
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    training_file_handler = logging.FileHandler(f'{result_dir}/training_logs.txt')
    training_file_handler.setFormatter(formatter)
    training_logger.addHandler(training_file_handler)
    training_logger.addHandler(console_handler)

    return training_logger, config_logger

def create_directories(config, num_of_folds):
    if config["train_params"]["need_train"]:
        # Set file paths - if the filename exists, create a new with an incremented index
        files = glob.glob(f'train_results/{config["train_params"]["exp_name"]}_*')
        indexes = [0]
        for f in files:
            indexes.append(int(os.path.splitext(os.path.basename(f))[0].split('_')[-1]))
        next_index = max(indexes) + 1
        result_dir = f'train_results/{config["train_params"]["exp_name"]}_{next_index}'
    else:
        result_dir = f'train_results/{config["eval_params"]["eval_dir"]}'

    # Create additional directiries
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
        for i in range(num_of_folds):
            i = i+1
            os.mkdir(f'{result_dir}/fold_{i}')
            os.mkdir(f'{result_dir}/fold_{i}/original_images')
            os.mkdir(f'{result_dir}/fold_{i}/reconstructed_images')

        
            if config["eval_params"]["need_eval"] and not os.path.exists(f'{result_dir}/fold_{i}/best_scores'):
                os.mkdir(f'{result_dir}/fold_{i}/best_scores')
                os.mkdir(f'{result_dir}/fold_{i}/worst_scores')
                os.mkdir(f'{result_dir}/fold_{i}/anomaly_scores')
                for s_type in config["eval_params"]["score_types"]:
                    os.mkdir(f'{result_dir}/fold_{i}/{s_type}')
       

    return result_dir