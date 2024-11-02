import argparse

from tqdm import tqdm

import SAA as SegmentAnyAnomaly
from datasets import *
from utils.csv_utils import *
from utils.eval_utils import *
from utils.metrics import *
from utils.training_utils import *


def eval(
        # model-related
        model,
        train_data: DataLoader,
        test_data: DataLoader,

        # visual-related
        resolution,
        is_vis,

        # experimental parameters
        dataset,
        class_name,
        cal_pro,
        img_dir,
        k_shot,
        experiment_indx,
        device: str
):
    similarity_maps = []
    scores = []
    test_imgs = []
    gt_list = []
    gt_mask_list = []
    names = []
    print("STOP")
  
    for (data, mask, label, name, img_type) in tqdm(test_data):

        for d, n, l, m in zip(data, name, label, mask):
            d = d.numpy()
            l = l.numpy()
            m = m.numpy()
            m[m > 0] = 1

            test_imgs += [d]
            names += [n]
            gt_list += [l]
            gt_mask_list += [m]

            score, appendix = model(d)
            scores += [score]

            similarity_map = appendix['similarity_map']
            similarity_maps.append(similarity_map)

    test_imgs, scores, gt_mask_list = specify_resolution(
        test_imgs, scores, gt_mask_list,
        resolution=(resolution, resolution)
    )
    _, similarity_maps, _ = specify_resolution(
        test_imgs, similarity_maps, gt_mask_list,
        resolution=(resolution, resolution)
    )

    scores = normalize(scores)
    similarity_maps = normalize(similarity_maps)
    
    np_scores = np.array(scores)
    img_scores = np_scores.reshape(np_scores.shape[0], -1).max(axis=1)


    if dataset in ['visa_challenge']:
        result_dict = {'i_roc': 0, 'p_roc': 0, 'p_pro': 0,
                       'i_f1': 0, 'i_thresh': 0, 'p_f1': 0, 'p_thresh': 0,
                       'r_f1': 0}
    else:
        gt_list = np.stack(gt_list, axis=0)
        result_dict = metric_cal(np.array(scores), gt_list, gt_mask_list, cal_pro=cal_pro)

    if is_vis:
        plot_sample_cv2(
            names,
            test_imgs,
            {'SAA_plus': scores, 'Saliency': similarity_maps},
            gt_mask_list,
            save_folder=img_dir
        )

    all_scores_mean = {}
    all_scores_max = {}
    for name, score in zip(names, scores):
      
        score_mean = np.mean(score)
        score_max = np.max(score)
        defect_type = name.split('-')[1]
        if defect_type not in all_scores_mean:
            all_scores_mean[defect_type] = []
            all_scores_max[defect_type] = []
        product_name = name.split("-")[0]
        with open(f"{img_dir}/{product_name}_{defect_type}_results.txt", 'a') as file:
            file.write(f"{name}; {score_max};\n")
        all_scores_mean[defect_type].append(score_mean)
        all_scores_max[defect_type].append(score_max)

    for i, all_scores in enumerate([all_scores_mean, all_scores_max]):
     
        data = [all_scores[key] for key in all_scores.keys()]
        labels = list(all_scores.keys())
        print(len(data), len(labels))
        cmap = matplotlib.colormaps['plasma']
        colors = cmap(np.linspace(0, 1, len(labels))) 
        normal_data_index =  list(all_scores.keys()).index("normal")
        # Boxplot 
        plt.figure(figsize=(8, 6))
        box = plt.boxplot(data, patch_artist=True, labels=labels)

        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        Q1 = np.percentile(data[normal_data_index], 25) 
        Q3 = np.percentile(data[normal_data_index], 75) 

        plt.axhline(y=Q1, color='gray', linestyle='--', linewidth=1)
        plt.axhline(y=Q3, color='gray', linestyle='--', linewidth=1)

        for median in box['medians']:
            median.set_color('black')
        plt.grid(axis='y')

        plt.xticks(rotation=90)  
        plt.savefig(f"{img_dir}/{product_name}_boxplot_{i}.png")



    compute_anomaly_scores(similarity_maps)
    return result_dict


def compute_anomaly_scores(similarity_maps):
    print(len(similarity_maps))
    print(similarity_maps.shape)


def main(args):
    kwargs = vars(args)

    # prepare the experiment dir
    model_dir, img_dir, logger_dir, model_name, csv_path = get_dir_from_args(**kwargs)

    logger.info('==========running parameters=============')
    for k, v in kwargs.items():
        logger.info(f'{k}: {v}')
    logger.info('=========================================')

    # give some random seeds
    seeds = [111, 333, 999, 1111, 3333, 9999]
    kwargs['seed'] = seeds[kwargs['experiment_indx']]
    setup_seed(kwargs['seed'])

    if kwargs['use_cpu'] == 0:
        device = f"cuda:0"
    else:
        device = f"cpu"

    kwargs['device'] = device

    # get the train dataloader
    if kwargs['k_shot'] > 0:
        train_dataloader, train_dataset_inst = get_dataloader_from_args(phase='train', perturbed=False, **kwargs)
    else:
        train_dataloader, train_dataset_inst = None, None

    # get the test dataloader
    test_dataloader, test_dataset_inst = get_dataloader_from_args(phase='test', perturbed=False, **kwargs)

    # get the model
    model = SegmentAnyAnomaly.Model(
        dino_config_file=kwargs['dino_config_file'],
        dino_checkpoint=kwargs['dino_checkpoint'],
        sam_checkpoint=kwargs['sam_checkpoint'],
        box_threshold=kwargs['box_threshold'],
        text_threshold=kwargs['text_threshold'],
        out_size=kwargs['eval_resolution'],
        device=kwargs['device'],
    )

    general_prompts = SegmentAnyAnomaly.build_general_prompts(kwargs['class_name'])
    manual_prompts = SegmentAnyAnomaly.manul_prompts[kwargs['dataset']][kwargs['class_name']]
    print("manual_prompts",  manual_prompts)
    print("general prompts", general_prompts)
    textual_prompts = general_prompts + manual_prompts

    model.set_ensemble_text_prompts(textual_prompts, verbose=False)

    property_text_prompts = SegmentAnyAnomaly.property_prompts[kwargs['dataset']][kwargs['class_name']]
    model.set_property_text_prompts(property_text_prompts, verbose=False)
    print("property prompts", property_text_prompts)
    model = model.to(device)

    metrics = eval(
        # model-related parameters
        model=model,
        train_data=train_dataloader,
        test_data=test_dataloader,

        # visual-related parameters
        resolution=kwargs['eval_resolution'],
        is_vis=True,

        # experimental parameters
        dataset=kwargs['dataset'],
        class_name=kwargs['class_name'],
        cal_pro=kwargs['cal_pro'],
        img_dir=img_dir,
        k_shot=kwargs['k_shot'],
        experiment_indx=kwargs['experiment_indx'],
        device=device
    )

    logger.info(f"\n")

    for k, v in metrics.items():
        logger.info(f"{kwargs['class_name']}======={k}: {v:.2f}")

    save_metric(metrics, dataset_classes[kwargs['dataset']], kwargs['class_name'],
                kwargs['dataset'], csv_path)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_args():
    parser = argparse.ArgumentParser(description='Anomaly detection')
    # data related parameters
    parser.add_argument('--dataset', type=str, default='mvtec',
                        choices=['mvtec', 'visa_challenge', 'visa_public', 'ksdd2', 'mtd', 'products'])
    parser.add_argument('--class-name', type=str, default='metal_nut')
    parser.add_argument('--k-shot', type=int, default=0) # no effect... just set it to 0.

    # experiment related parameters
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--vis', type=str2bool, choices=[True, False], default=True)
    parser.add_argument("--root-dir", type=str, default="./result")
    parser.add_argument("--cal-pro", type=str2bool, default=False)
    parser.add_argument("--experiment_indx", type=int, default=0) # no effect... just set it to 0.
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--use-cpu", type=int, default=0)

    # method related parameters
    parser.add_argument('--eval-resolution', type=int, default=400)
    parser.add_argument("--dino_config_file", type=str,
                        default='GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
                        help="path to config file")
    parser.add_argument(
        "--dino_checkpoint", type=str, default='weights/groundingdino_swint_ogc.pth', help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, default='weights/sam_vit_h_4b8939.pth', help="path to checkpoint file"
    )

    parser.add_argument("--box_threshold", type=float, default=0.1, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.1, help="text threshold")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    import os

    args = get_args()
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu_id}"
    main(args)


def save_diagrams( result_die, criterion, threshold = 75):
    # Collect values
 
    scores = {}
    test_dataset_idx = 0
   
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

   

    # Boxplothoz
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
               
    Q1 = np.percentile(data[test_dataset_idx], 25)  # Első kvartilis
    Q3 = np.percentile(data[test_dataset_idx], 75)  # Harmadik kvartilis

    plt.axhline(y=Q1, color='gray', linestyle='--', linewidth=1)
    plt.axhline(y=Q3, color='gray', linestyle='--', linewidth=1)
    plt.ylabel('Anomaly score')
    plt.xlabel('Image groups')
    plt.grid(axis='y')
    plt.xticks(rotation=90) 

    plt.savefig(f"{result_dir}/{criterion}_boxplot.png")


    #Thresholds
    threshold = np.percentile(data[test_dataset_idx], threshold)

    counts_below = {}  
    counts_above = {}  

    for key, values in scores.items():
        # Számlálás threshold alapján
        below = sum(1 for value in values if value < threshold) 
        above = sum(1 for value in values if value >= threshold)
        
        total_count = len(values)
        counts_below[key] = round((below / total_count) * 100, 3) 
        counts_above[key] = round((above / total_count) * 100, 3)

 # Adatok ábrázolása halmozott oszlopdiagramon
    labels = list(scores.keys())
    below_percent = list(counts_below.values())
    above_percent = list(counts_above.values())

    x = np.arange(len(labels))  # X tengely pozíciók

    plt.figure(figsize=(8, 6))

    # Halmozott oszlopok rajzolása
    for i in range(len(labels)):
        if i == test_dataset_idx:  # Ha ez a hatodik mérés (6. oszlop)
            plt.bar(x[i], below_percent[i], label=f'Below {threshold}', color="lightblue")
            plt.bar(x[i], above_percent[i], bottom=below_percent[i], label=f'Above {threshold}', color="lightyellow")
        else:
            plt.bar(x[i], below_percent[i], label=f'Below {threshold}' if i == 0 else "", color="red")
            plt.bar(x[i], above_percent[i], bottom=below_percent[i], label=f'Above {threshold}' if i == 0 else "", color="lightgreen")
        plt.text(x[i], below_percent[i] + above_percent[i] / 2, f'{(above_percent[i]):.1f}%', ha='center', va='center', color='black', fontsize=10)
        

    # Címek és tengelyek beállítása
    plt.xticks(x, labels)
    plt.xlabel('Measurements')
    plt.ylabel('Percentage (%)')
    plt.title(f'Percentage of Values Above and Below {round(threshold,4)}')
    plt.xticks(rotation=90) 

    # Megjelenítés
    plt.savefig(f"{result_dir}/{criterion}_histogram.png")



    # Az értékek (min, max, avg) kiszámítása minden méréshez
    min_values = [min(scores[key]) for key in scores.keys()]
    max_values = [max(scores[key]) for key in scores.keys()]
    avg_values = [np.mean(scores[key]) for key in scores.keys()]

    labels = list(scores.keys())

    # Diagram létrehozása
    plt.figure(figsize=(10, 6))

    # Számegyenes ábrázolása az intervallumokkal
    for i, label in enumerate(labels):
        # Intervallum ábrázolása (min-max)
        plt.plot([min_values[i], max_values[i]], [i, i], color='gray', marker='|', markersize=10)
        
        # Átlag értékek ábrázolása
        plt.plot(avg_values[i], i, 'ro', label='Avg' if i == 0 else "")  # 'ro' piros körrel ábrázolva

    # Tengely beállítások
    plt.yticks(np.arange(len(labels)), labels)  # Az Y tengely címkézése a mérésekkel
    plt.xlabel('Values')
    plt.title('Min, Max, and Avg Intervals for Different Measurements')

    # Megjelenítés
    plt.grid(True)
    plt.savefig(f"{result_dir}/{criterion}_line_diagram.png")
