# IMPORTS
import os
import sys
from pathlib import Path
import pickle
import cv2
import torch
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from models.common import DetectMultiBackendForClustering
from utils.datasets import  LoadImagesForClustering, LoadImages
from utils.general import ( check_img_size, increment_path)
from utils.torch_utils import select_device, time_sync
import torch.nn as nn
import matplotlib.pyplot as plt
import hdbscan
from umap import UMAP
from sklearn.neighbors import NearestNeighbors
import yaml
import matplotlib.patches as patches
import joblib
from sklearn.preprocessing import normalize

# Create file paths
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

@torch.no_grad()
def run():
    # Load params from config file
    params = yaml.safe_load(open("config/params.yaml"))
    # Saving directory
    save_dir = increment_path(Path("runs/clasters") / params["save_dir"], exist_ok=False)  # increment run
    print(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    #Save config file
    with open(os.path.join(save_dir, "config.yaml"), "w") as file:
        yaml.safe_dump(params, file)
 
    if params["saved_detection"] == []: # If a new dtection is needed
        base_model, imgsz = load_YOLO_model(params)
        dataset = load_data(params["image_file_path"], base_model.stride, imgsz)
        latent_space, filenames, anomaly_flag = detection_with_YOLO(params, base_model, dataset, imgsz)     
        latent_space = torch.stack(latent_space)
    
        latent_space = latent_space.cpu().numpy()
   
        if params["PCA_explained_variance"]:
            PCA_measure_explained_variance(params, latent_space, save_dir)

        # DIMENSION REDUCTION
        latent_space = dimension_reduction(params, latent_space, save_dir )

        # SCALING
        if params["scaling_type"] == "normalizing" :
            latent_space = normalize(latent_space, norm='l2')
        if params["scaling_type"] == "standard":
            scaler = StandardScaler()
            latent_space = scaler.fit_transform(latent_space)
            joblib.dump(scaler, os.path.join(save_dir,'scaler.pkl'))
        elif params["scaling_type"] == "robust":
            scaler = RobustScaler()
            latent_space = scaler.fit_transform(latent_space)
            joblib.dump(scaler, os.path.join(save_dir,'scaler.pkl')) 
        
        latent_space_reduced = latent_space
       

    # LOAD A SAVED REPRESENTATION FROM FILE        
    else: 
        file = params["saved_detection"]  
        latent_space_reduced = np.load(file)
        with open('filenames.txt', 'r') as f:
            filenames = [line.strip() for line in f]
        with open('anomaly_list.txt', 'r') as f:
            anomaly_flag = [line.strip() for line in f]
        anomaly_flag = list(map(lambda x: x == "True", anomaly_flag))


    # DBSCAN PARAMETER TEST WITH NEAREST NEIGHBORS
    if params["dbscan_eps_test"]:
        NearestNeigborsForDbscan(latent_space_reduced, save_dir)

    # CLASTERING
    clusterer = perform_clastering(params, latent_space_reduced)
    # Save the model  
    joblib.dump(clusterer, 'clasterer_model.joblib')
    labels = clusterer.labels_
   
    # COLLECTING OUTLIERS
    num_clusters = clustering_evaluation(labels, latent_space_reduced, filenames, clusterer.probabilities_)
    outliers_evaluation(anomaly_flag, labels)

    # VISUALIZE IMAGES IN THE CLASTERS
    visualize_images_in_clusters(num_clusters, labels, clusterer.outlier_scores_,   filenames, anomaly_flag, save_dir)

    # visualize the reprezentation in 2D
    visualization_in_2D(params, labels, anomaly_flag, latent_space_reduced, save_dir )


def load_YOLO_model(params):
    # Load model
    device = select_device(params["device"])
    base_model = DetectMultiBackendForClustering(params["weights_path"], device=device, YOLO_layers= params["YOLO_layers"])
    imgsz = check_img_size(params["image_size"], s=base_model.stride)  # check image size

    # Half
    if device.type != 'cpu':
        params["half"]== False  # half precision only supported by PyTorch on CUDA
    base_model.model.half() if params["half"] else base_model.model.float()

    return base_model, imgsz

def load_data(source, stride, imgsz ):
    if source == 'dataset/reels_sorted/normal':
        dataset = LoadImagesForClustering([source], img_size=imgsz, stride=stride, auto=True)
    
    else:
        image_paths = []
        isAnomaly = True
        anomalies = []
        for dir_name in source:
            isAnomaly = True 
            if dir_name.split("/")[-1] == 'normal' or dir_name.split("/")[-1] == 'fingers' or dir_name.split("/")[-1] == 'hands'    :
                isAnomaly = False
            for dirpath, dirnames, files in os.walk(dir_name):
                for file in files:
                    image_paths.append(os.path.join(dirpath, file))
                    anomalies.append(isAnomaly)      
        dataset = LoadImagesForClustering(image_paths, anomalies, img_size=imgsz, stride=stride, auto=True)
    return dataset


def detection_with_YOLO(params, base_model, dataset, imgsz):
   
    # RUN YOLO MODEL    
    base_model.warmup(imgsz=(1, 3, imgsz, imgsz), half=params["half"])  # warmup
    filenames = []
    anomaly_flag = []
    latent_space = []
    dt= [0.0, 0.0, 0.0]
    last = 0
    device = select_device(params["device"])
    for file, im, _, isAnomaly, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if params["half"] else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        pred = base_model(im) #shape: 1, 1024, 8, 8
                
        # Reshape features from [1, 1024, 20, 20] to  [1, 409600]  
        latent_space.append(pred)
        filenames.append(file)
        anomaly_flag.append(isAnomaly)
        t3 = time_sync()
        dt[1] += t3 - t2

    torch.save(latent_space, 'yolo_output_tensor.pt')
    with open('filenames.txt', 'w') as f:
        for file in filenames:
            f.write("%s\n" % file)

    with open('anomaly_list.txt', 'w') as f:
        for file in anomaly_flag:
            f.write("%s\n" % file)

    return latent_space, filenames, anomaly_flag


def PCA_measure_explained_variance(params, latent_space, save_dir):
    print("PCA explained variance computing...")
    pca = PCA(n_components=600, random_state=22)
    pca.fit(latent_space)

    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    # cumulative explained variance
    cumulative_variance = explained_variance.cumsum()
    # Plot
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance by PCA')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "PCA_explained_variance.png"))


def dimension_reduction(params, latent_space, save_dir ):
    if params["dim_reduction_type"] == "PCA":
            print("performing PCA...")
            pca = PCA(n_components=params["pca_params"]["n_components"])  
            latent_space_reduced = pca.fit_transform(latent_space)
           
            np.save(os.path.join(save_dir, 'pca.npy'), latent_space_reduced)


    elif params["dim_reduction_type"] == "UMAP_and_PCA" or params["dim_reduction_type"] == "UMAP":
        print("performing UMAP...")
        if params["dim_reduction_type"] == "UMAP_and_PCA" :
            pca = PCA(n_components=params["pca_params"]["n_components"])  
            latent_space = pca.fit_transform(latent_space)
            # Save PCA model
            joblib.dump(pca, os.path.join(save_dir,'pca.pkl'))

        umap_model = UMAP(n_components=params["umap_params"]["n_components"], 
                          n_neighbors= params["umap_params"]["n_neighbors"], 
                          min_dist=params["umap_params"]["min_dist"],
                          metric=params["umap_params"]["metric"],
                          set_op_mix_ratio=params["umap_params"]["set_op_mix_ratio"])
                     
        latent_space_reduced = umap_model.fit_transform(latent_space)
        #Save UMAP model
        np.save(os.path.join(save_dir, 'umap.npy'), latent_space_reduced)
        joblib.dump(umap_model, os.path.join(save_dir,'umap.pkl'))
    else:
        print("The selected dim reduction type is not supported yet!")
    return latent_space_reduced


def  NearestNeigborsForDbscan(latent_space_reduced, save_dir):
    neighbors = NearestNeighbors(n_neighbors=10)  
    neighbors_fit = neighbors.fit(latent_space_reduced)
    distances, indices = neighbors_fit.kneighbors(latent_space_reduced)

    distances = np.sort(distances[:, 9], axis=0)
    plt.plot(distances)
    plt.ylabel("10th Nearest Neighbor Distance")
    plt.xlabel("Data Points sorted by distance")
    plt.savefig(os.path.join(save_dir,"dbscan.png"))


def perform_clastering(params, latent_space_reduced):
    if params["clastering_method"] == "DBSCAN":
        print("performing DBSCAN...")
        return DBSCAN_clastering(params, latent_space_reduced)
        
    elif params["clastering_method"] == "HDBSCAN":
        print("performing HDBSCAN...") 
        return HDBSCAN_clastring(params, latent_space_reduced)
    else:
        print("The selected clastering method is not implemented yet!")    

def DBSCAN_clastering(params, latent_space_reduced):
    dbscan=DBSCAN(algorithm=params["dbscan_params"]["algorithm"], 
                      eps=params["dbscan_params"]["eps"], 
                      leaf_size=300,
                      metric=params["dbscan_params"]["metric"], 
                      metric_params=None,
                      min_samples=params["dbscan_params"]["min_samples"], 
                      n_jobs=None, p=None)
    dbscan.fit(latent_space_reduced)
    return dbscan


def HDBSCAN_clastring(params, latent_space_reduced):

   hdbscan_result = hdbscan.HDBSCAN(min_cluster_size=params["hdbscan_params"]["min_cluster_size"],
                            leaf_size=300,    
                            min_samples=params["hdbscan_params"]["min_samples"], 
                            allow_single_cluster=params["hdbscan_params"]["allow_single_cluster"], 
                            alpha=params["hdbscan_params"]["alpha"], 
                            metric=params["hdbscan_params"]["metric"],
                            cluster_selection_method=params["hdbscan_params"]["cluster_selection_method"],
                            prediction_data=False)
    hdbscan_result.fit(latent_space_reduced)
    hdbscan_result.generate_prediction_data()
    return hdbscan_result


def visualization_in_2D(params, labels, anomaly_flag, latent_space_reduced, save_dir):
    anomaly_flag = np.array(anomaly_flag)
    if params["dim_reduction_type"] == "PCA":
        pca = PCA(n_components=2)  # reduce dimensions to 2
        latent_space_reduced = pca.fit_transform(latent_space_reduced)

        plt.figure()
        anomaly = np.array([latent_space_reduced[i]  for i, anomaly in enumerate(anomaly_flag) if  anomaly ])
        
        plt.scatter(latent_space_reduced[:,0], latent_space_reduced[:,1], c = labels, cmap='viridis', marker='o', edgecolor='k', s=5)
        plt.scatter(anomaly[:,0], anomaly[:,1], c = 'red', marker='X', edgecolor='k', s=2)
        plt.title("Clustering Results in 2D")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.colorbar(label='Cluster number')
        plt.savefig(os.path.join(save_dir,"clastering_PCA_2D.png"))
    
    elif params["dim_reduction_type"] == "UMAP" or params["dim_reduction_type"] == "UMAP_and_PCA":
        umap_model = UMAP(n_components=2, n_neighbors= 20, 
                                 min_dist=0.5, set_op_mix_ratio=0.25)
        embedded_data = umap_model.fit_transform(latent_space_reduced)

        plt.figure()
       #Add highlight to anomalious samples 
        anomaly = np.array([embedded_data[i]  for i, anomaly in enumerate(anomaly_flag) if  anomaly ])
        
        plt.scatter(embedded_data[:,0], embedded_data[:,1], c = labels, cmap='viridis', marker='o', s=5)
        plt.colorbar(label='Cluster group')
        plt.scatter(anomaly[:,0], anomaly[:,1], c = 'r', marker='X', s=1)
       
        plt.xlabel("UMAP Component 1")
        plt.ylabel("UMAP Component 2")
        plt.title('Cluster Visualization in 2D using UMAP')   
        plt.savefig(os.path.join(save_dir,"2D_UMAP_diagram_with_anomalies.png"))




def clustering_evaluation(labels, data, filenames, probabilities):
    unique_clusters = np.unique(labels)
    num_clusters = len(unique_clusters[unique_clusters != -1])

    print(f"Number of clusters: {num_clusters}")
   
    outlier_indices = np.where(labels == -1)[0]
    outlier_images = [filenames[i] for i in outlier_indices]
    print(f"Number of outliers: {len(outlier_images)}")

    return num_clusters



def visualize_images_in_clusters(num_clusters, labels, outlier_scores, filenames, anomaly_flag, save_dir):

    for i in range(num_clusters+1):
        # Outliers
        indices = np.where(labels == i-1)[0]
        if len(indices) < 1000: 
            images = [filenames[i] for i in indices]
            anomalies = [anomaly_flag[i] for i in indices ]
            fig, axes = plt.subplots(nrows= (len(images) + 10) // 10, ncols=10, figsize=(20,60))
            axes = axes.flatten() 
            for ax, img, anomaly in zip(axes, images, anomalies):
                image = cv2.imread(img)
                if image is not None:
                    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # BGR -> RGB
                    if anomaly:         
                        rect = patches.Rectangle((0, 0), image.shape[1], image.shape[0], 
                                            linewidth=6, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)
                ax.axis('off')
                ax.set_title(round(outlier_scores[i], 3), fontsize=12)
            # Hide empty subplots
            for j in range(len(images), len(axes)):
                axes[j].axis('off')
            plt.savefig(os.path.join(save_dir, f"clusters_{i-1}.png"))
            plt.close(fig)


def outliers_evaluation(anomaly_flag, labels):
    print("Num of anomalies: ", sum(anomaly_flag))
    TP = sum(1 for b, n in zip(anomaly_flag, labels) if b and n == -1)
    FN = sum(1 for b, n in zip(anomaly_flag, labels) if b and n != -1)
    TN = sum(1 for b, n in zip(anomaly_flag, labels) if not b and n != -1)
    FP = sum(1 for b, n in zip(anomaly_flag, labels) if not b and n == -1)
    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    print("Accuracy of anomalies:", TP /( TP + FP +1))


if __name__ == "__main__":
    run()