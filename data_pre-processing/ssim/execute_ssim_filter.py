import csv
import logging
import time
import sys
from multiprocessing import cpu_count, Pool
from pathlib import Path
import cv2
import numpy as np
from skimage.metrics import structural_similarity
from worker import process_ssim_per_core
    
def get_chunk(img_paths, chunk_size):
    for i in range(0, len(img_paths), chunk_size):
        yield img_paths[i: i+ chunk_size]


def ssim_filter_consecutive_duplicates_multi_thread(images_path, save_path_approved, save_path_filtered, ssim_threshold):
        """
        Consequtivity is ensured only for chunks. Border elements in chunks are automatically kept as approved samples.
        """
        imgs = sorted(images_path)
        procs = 10 
        procIDs = list(range(0, procs))

        num_images_per_process = len(imgs) / float(procs)
        num_images_per_process = int(np.ceil(num_images_per_process))
        chunked_paths = list(get_chunk(imgs, num_images_per_process))

        # construct  payloads
        payloads = []
        for (i, image_paths) in enumerate(chunked_paths):
            out_path = save_path_approved / Path(f"approved_{i}.csv")

            data = {
                "id": i,
                "input_paths": image_paths,
                "output_path": out_path,
                "ssim_threshold": ssim_threshold
            }

            payloads.append(data)
        logger.info(f"Processing: {len(imgs)} pictures.")
        chunk_sizes = [len(chunk["input_paths"]) for chunk in payloads]
        logger.info(f"Processing: {chunk_sizes} pictures concurrently.")

        exec_start = time.time()

        # distribute processes
        logger.info(f"Launching pool using {procs} processes...")
        pool = Pool(processes=procs)
     
        approved_file_list = pool.map(process_ssim_per_core, payloads)
        logger.info(f"Approved images count: {sum([len(chunk) for chunk in approved_file_list])}")

        with open(save_path_approved / Path("approved_images.csv"), mode="w", newline='') as approved_images_csv:
            logger.info(f"Saving results into {save_path_approved}")        
            csv_writer = csv.writer(approved_images_csv, delimiter='\n')
            csv_writer.writerows(approved_file_list)

        logger.info(f"Waiting for processes to finish...")
        pool.close()
        pool.join()
        exec_end = time.time()
        logger.info(f"Pre-filtering took {round(exec_end - exec_start)} seconds")
        logger.info(f"Processes finished.")


if __name__ ==  '__main__': 
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    images_path = Path("reshaped_images")
    logger.info(f"Collecting pictures from {images_path}...")   
    images_path = list(images_path.glob('**/*.png'))


    image1 = cv2.imread(str(images_path[1]))
    print(image1)

    # consequtive SSIM filter
    ssim_filter_consecutive_duplicates_multi_thread(images_path, 
                                                    save_path_approved=Path("result"),
                                                    save_path_filtered=Path("result/"),
                                                    ssim_threshold=0.9)

