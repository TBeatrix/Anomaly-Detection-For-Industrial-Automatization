import pickle
import os
from clastering import *
import hdbscan
from hdbscan.validity import validity_index 
import yaml
import torch
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import joblib

def run():
    # Create file paths
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0] 
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT)) 
    ROOT = Path(os.path.relpath(ROOT, Path.cwd())) 
       
    # Load params from config file
    params = yaml.safe_load(open("config/params.yaml"))

    # Saving directory
    save_dir = Path(params["train_dir"] +  "/predictions") # Select the same directory as the training
    save_dir.mkdir(parents=True, exist_ok=True)  

    base_model, imgsz = load_YOLO_model(params)
    normal_dataset = LoadImagesForClustering(params["normal_image_file_path"], img_size=imgsz, stride=base_model.stride, auto=True)
    anomaly_dataset = LoadImagesForClustering(params["anomaly_image_file_path"], img_size=imgsz, stride=base_model.stride, auto=True)

    # Load data - normal and anomaly and predict with YOLO
    latent_space_0, filenames_normal, _ = detection_with_YOLO(params, base_model, normal_dataset, imgsz)
    latent_space_1, filenames_anomaly, _ = detection_with_YOLO(params, base_model, anomaly_dataset, imgsz)
    latent_space = latent_space_0 + latent_space_1
   
    latent_space = torch.stack(latent_space)
    latent_space = latent_space.cpu().numpy()

    # DIMENSION REDUCTION
    new_data = predict_dim_reduction(params, latent_space, save_dir )
    
    if params["scaling_type"] == "normalizing":
        new_data = normalize(new_data, norm='l2')
    else:
        scaler = joblib.load(os.path.join(params["train_dir"],'scaler.pkl'))# load RobustScaler or StandardScaler()
        # Fit the scaler to your data and transform it
        new_data = scaler.transform(new_data)
       
    # Prediction
    labels, membership_strengths = predict_clusters( new_data, params)
    outlier_indices = np.where(labels == -1)[0]
    
    print(f"Number of outliers: {len(outlier_indices)}")
    # Evaluation
    TP, FP, TN, FN = [], [], [], []
    for i, normal in enumerate(normal_dataset):
        if labels[i] != -1:
            TN.append(filenames_normal[i])
        else:
            FP.append(filenames_normal[i])
    
    for i, anomaly in enumerate(anomaly_dataset):
        if labels[i + len(normal_dataset)] == -1:
            TP.append(filenames_anomaly[i])
        else:
            FN.append(filenames_anomaly[i])

    print(f"Anomalies detected as anomalies: {len(TP)}")
    print(f"Anomalies detected as normal: {len(FN)}")
    print(f"Normal images detected as anomalies: {len(FP)}")
    print(f"Normal images detected as normal: {len(TN)}")

    #Visualization of the images
    visualize_predictions(TP, FP, TN, FN, save_dir)
    

def predict_dim_reduction(params, latent_space, save_dir ):
    if params["dim_reduction_type"] == "PCA":
            print("performing PCA...")
            pca = joblib.load(os.path.join(params["train_dir"], 'pca.pkl'))# load PCA
            latent_space_reduced = pca.transform(latent_space)
          

    elif params["dim_reduction_type"] == "UMAP":
        print("performing UMAP...")
        pca =joblib.load(os.path.join(params["train_dir"], 'pca.pkl')) # loaded PCA
        latent_space_reduced = pca.transform(latent_space)
        umap_model = joblib.load(os.path.join(params["train_dir"], 'umap.pkl')) # loaded UMAP
        latent_space_reduced = umap_model.transform(latent_space_reduced)
        np.save(os.path.join(save_dir, 'umap.npy'), latent_space_reduced)

    else:
        print("The selected dim reduction type is not supported yet!")
    return latent_space_reduced


def visualize_predictions(TP, FP, TN, FN,  save_dir):
    name_list = ["TP", "FP", "TN", "FN"]
    for i, data in enumerate([TP, FP, TN, FN]):
       
        images = data  
        fig, axes = plt.subplots(nrows= (len(images) + 15) // 15, ncols=15, figsize=(20,40))
        axes = axes.flatten() 
        for ax, img in zip(axes, images):
            image = cv2.imread(img)
            if image is not None:
                ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # BGR -> RGB
            ax.axis('off')
        # Üres subplotok elrejtése
        for j in range(len(images), len(axes)):
            axes[j].axis('off')
        plt.savefig(os.path.join(save_dir, f"{name_list[i]}.png"))
        plt.close(fig)
        



def predict_clusters(data, params):
    # Load model
    loaded_model = joblib.load('clasterer_model.joblib')
    if params["clastering_method"] == "DBSCAN":
        print("predicting with DBSCAN...")
        return "Not implemented yet"    
    elif params["clastering_method"] == "HDBSCAN":
        print("predicting with HDBSCAN...") 
        labels, membership_strengths = hdbscan.approximate_predict(loaded_model, data)
        return labels, membership_strengths
    else:
        print("The selected clastering method is not implemented yet!") 

if __name__ == "__main__":
    run()  