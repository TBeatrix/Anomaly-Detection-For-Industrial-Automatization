import logging
import sys
from pathlib import Path
import csv
import logging
import time
import sys
from multiprocessing import cpu_count, Pool
from pathlib import Path
import cv2
import numpy as np
from skimage.metrics import structural_similarity

def process_ssim_per_core(payload):
    approved_file_list = []
    
    for i in range(len(payload["input_paths"])):
        
        if i+1 < len(payload["input_paths"]):
            image1 = cv2.imread(str(payload["input_paths"][i]))
            
           
            image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            image2 = cv2.imread(str(payload["input_paths"][i+1]))
            image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            score = structural_similarity(image1_gray, image2_gray)
            
            if score < payload["ssim_threshold"]:
                approved_file_list.append(str(payload["input_paths"][i+1].name))
            else:
                pass
    
    return approved_file_list
