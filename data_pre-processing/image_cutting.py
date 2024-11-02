import numpy as np
import pandas as pd
import glob
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2
import math 
import re



def calc_bounding_box_angle(picture_label_coord):
    y1 = float(picture_label_coord[1])
    y2 = float(picture_label_coord[3])
    x1 = float(picture_label_coord[0])
    x2 = float(picture_label_coord[2])
    side1 = x2 - x1
    side2 = y2 - y1
    if side2 == 0 or side1 == 0:
        return 0
    angle_rad = math.atan2(side1, side2)
    angle_deg = math.degrees(angle_rad)
   
    return(angle_deg)




def base_pic_name_extract(s):
    patterns = ['_colored_depth', '_color']
    regex_pattern = '|'.join(map(re.escape, patterns))  
    s = re.sub(regex_pattern, '', s)
    return s


def rotate_image(image, angle, x1, y1, x3, y3):
    bcx = x1 + (x3-x1)/2
    bcy = y1 + (y3-y1)/2
    rx = image.shape[0]/2 + (bcx - image.shape[0]/2)
    ry = image.shape[1]/2 + (bcy - image.shape[1]/2)
    bounding_box_center = (rx, ry)
    rot_mat = cv2.getRotationMatrix2D(bounding_box_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def rotate_bounding_box(angle, x1, y1, x3, y3, x, y):
    #bounding box center point
    cx = x1 + (x3-x1)/2
    cy = y1 + (y3-y1)/2
    #radius of the circle
    r = round(math.sqrt(math.pow(x1-cx,2) + math.pow(y1-cy,2)))

    dy = round(y - cy)
    dx = round(x - cx)
 
    if (abs(dy)>r):
        alpha_rad = math.asin(1)
    else:
        alpha_rad = math.asin(abs(dy)/r)

    alpha = alpha_rad / 2 / math.pi * 360

    if (dy>=0) and (dx>=0):
        alpha = alpha + 0
    if (dy>=0) and (dx<0):
        alpha = 90 - alpha + 90
    if (dy<0) and (dx<0):
        alpha = alpha + 180
    if (dy<0) and (dx>=0):
        alpha = 90 - alpha + 270   


    beta = alpha + angle
    beta_rad = beta/360*2*math.pi
    y_new = cy + r * math.sin(beta_rad) 
    x_new = cx + r * math.cos(beta_rad) 
    return([x_new, y_new])

def extract_number(s):
    return int(s.split('_')[1])

# create cropped pictures based on the labeling
def shifting_rectangle_to_valid_image_points(im_rgb, xmin, xmax, ymin, ymax, pheight, pwidth):
    #check if manipulation needed
    i_x_max = im_rgb.shape[0]
    i_y_max = im_rgb.shape[1]


    if xmin < 0:
        xmax -= xmin 
        xmin = 0
    if ymin < 0:
        ymax -= ymin 
        ymin = 0
    if xmax > i_x_max:
        xmin -= (xmax - i_x_max)  
        xmax = i_x_max
    if ymax > i_y_max:
        ymin -= (ymax - i_y_max)  
        ymax = i_y_max

    return(xmin, xmax, ymin, ymax)


if __name__ == "__main__":

    directory = ""
    image_file_path = f"result_images_and_labels_with_openboxes/images/open_box_images_{directory}"
    label_file_path = f"result_images_and_labels_with_openboxes/labels/open_box_labels_{directory}"

    target_path = "cropp_images/results"
    black_background = True
    
    # load images 
    path_data =  [(path, name) for path, _, files in os.walk(image_file_path) for name in files]

    df_images = pd.DataFrame(path_data, columns=['image_path', 'image_files'])

    df_images['base_pic_id'] = df_images.apply(lambda row: (row['image_files'])[0:-4], axis=1)
    df_images['base_pic_id'] =  df_images.apply(lambda row: base_pic_name_extract(row['base_pic_id']), axis=1) 

    #Load labels
    path_data = [(path, name) for path, _, files in os.walk(label_file_path) for name in files]
    df_labels = pd.DataFrame(path_data, columns=['label_path', 'label_files'])
    
    df_labels['base_pic_id'] = df_labels.apply(lambda row: (row['label_files'])[0:-4], axis=1)
    df_labels['base_pic_id'] =  df_labels.apply(lambda row: base_pic_name_extract(row['base_pic_id']), axis=1)     

    print("Df label test", df_labels.head(2))
    print("Number of labels", len(df_labels))
    print("Number of images", len(df_images))

    
    df_final = pd.merge(df_labels, df_images, on='base_pic_id')
    df_final = df_final.drop_duplicates(subset=['base_pic_id'], keep='last')
    df_final = df_final.reset_index(drop=True)

    print("Length of final df: ", len(df_final))

    columns = np.append(df_final.columns.values, "picture_label_coord")
    df = pd.DataFrame(columns=columns)
    # creates a ScandirIterator aliased as files
    for i in range(len(df_final)):
        filename_with_path = df_final['label_path'].iloc[i] + "/" + df_final['label_files'].iloc[i]
        with open(filename_with_path, encoding='utf8') as f:

            for line in f:
                cur_row = line.split(sep = ' ')
                # if ((eval(cur_row[8])[0] == 'open_box') or (eval(cur_row[8])[0] == 'box')):   # For extracting only the openboxes
                box_coordinates = pd.DataFrame([[cur_row[0:9]]], columns= ["picture_label_coord" ])
                row_to_add = pd.concat([df_final.iloc[[i]].reset_index(drop=True), box_coordinates.reset_index(drop=True)], axis=1) 
                df = pd.concat([df, row_to_add], ignore_index=True)               

    print("Df with duplications where there are more than 1 open box", len(df))
    # Sort the DataFrame using the custom key
    df_final = df.loc[df['base_pic_id'].map(extract_number).argsort()]
    df_final = df_final.reset_index()
    df_final['bounding_box_angle'] = df_final.apply(lambda row: calc_bounding_box_angle(row['picture_label_coord']), axis=1)
    
    df_final['rot_bbox_coordinates'] = df_final.apply(lambda row: [], axis=1)
    if black_background:
        epsilon = 5 #epsilon: all directon take an eplison larger crop value for the pictures
    else:
        epsilon = 15


    for i in range(len(df_final)):
        picture_w_path = df_final['image_path'].iloc[i] + '/' + df_final['image_files'].iloc[i]
        image = cv2.imread(picture_w_path)
        angle = 360 - df_final['bounding_box_angle'].iloc[i]
        angle_orig = df_final['bounding_box_angle'].iloc[i]

        #project picture to a larger black image mask, to avoid data loss at the rotation of large bounding boxes
        # Create a larger black image mask
        l_img = np.zeros((2*image.shape[0], 2*image.shape[1], 3), dtype=np.uint8)

        # Calculate offsets to center `image` in `l_img`
        x_offset = (l_img.shape[1] - image.shape[1]) // 2
        y_offset = (l_img.shape[0] - image.shape[0]) // 2

        # Place `image` at the calculated offset within `l_img`
        l_img[y_offset:y_offset+image.shape[0], x_offset:x_offset+image.shape[1]] = image
        #original bounding box shift to the new projetion  
        x1, y1, x2, y2, x3, y3, x4, y4 = [ round(float(df_final['picture_label_coord'].iloc[i][j]) + image.shape[(j % 2)==0] / 2) for j in range(8) ]

        original_coords = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        rotated_coords = [rotate_bounding_box(angle_orig, x1, y1, x3, y3, *coord) for coord in original_coords]
        new_coords = [round(coord) for coords in rotated_coords for coord in coords]
       
        df_final.at[i, 'rot_bbox_coordinates'] = new_coords.copy()
        rotated_image = rotate_image(l_img, angle, x1, y1, x3, y3)
        im_rgb = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)  
      
        #area_coordinated
        y_coords = [round(float(df_final.at[i, 'rot_bbox_coordinates'][j])) for j in range(0, 7, 2)]
        x_coords = [round(float(df_final.at[i, 'rot_bbox_coordinates'][j])) for j in range(1, 8, 2)]

        # Min and max values
        xmin, xmax = min(x_coords), max(x_coords)
        ymin, ymax = min(y_coords), max(y_coords)

        # Width and height 
        pheight = xmax - xmin
        pwidth =  ymax - ymin
        #width - height ratio
        delta = abs(pheight - pwidth) / 2

        if not black_background:
            if pheight > pwidth:
                ymin, ymax = round(ymin - delta), round(ymax + delta)
            else:
                xmin, xmax = round(xmin - delta), round(xmax + delta)

        #manipulate the picture bounding box-es if necessary
        xmin, xmax, ymin, ymax = shifting_rectangle_to_valid_image_points(im_rgb, xmin, xmax, ymin, ymax, pheight, pwidth)
        test_picture_rgb = im_rgb[xmin-epsilon:xmax+epsilon,ymin-epsilon:ymax+epsilon,:].copy()

        #Create black borders
        if black_background:
            if pheight > pwidth:
                delta = (pheight - pwidth) // 2
                test_picture_rgb = cv2.copyMakeBorder(test_picture_rgb, 0, 0, delta, delta, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            else:
                delta = (pwidth - pheight) // 2
                test_picture_rgb = cv2.copyMakeBorder(test_picture_rgb, delta, delta, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
        filename = df_final['base_pic_id'].iloc[i] + '_cropped_' + str(i) + '.png'
        df_final.at[i, "filename"] = filename
        target_path_with_filename = os.path.join(target_path, filename)
        im_rgb = cv2.cvtColor(test_picture_rgb, cv2.COLOR_BGR2RGB)
        result = cv2.imwrite(target_path_with_filename, im_rgb)    
        if (i % 500 == 0 ):
            print(result)