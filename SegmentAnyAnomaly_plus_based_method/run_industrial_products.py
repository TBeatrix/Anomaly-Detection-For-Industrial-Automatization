import os
from datasets import dataset_classes
from multiprocessing import Pool

if __name__ == '__main__':

    pool = Pool(processes=1)

    dataset_list = ['products']
    gpu_indx = 0

    for dataset in dataset_list:
        print(dataset)
        classes = dataset_classes[dataset]
        for cls in classes[:]:
            print(cls)
            sh_method = f'python3 eval_SAA.py ' \
                        f'--dataset {dataset} ' \
                        f'--class-name {cls} ' \
                        f'--batch-size {1} ' \
                        f'--root-dir ./result_reels_big12 ' \
                        f'--cal-pro False ' \
                        f'--gpu-id {gpu_indx} ' \

            print(sh_method)
            pool.apply_async(os.system, (sh_method,))

    pool.close()
    pool.join()