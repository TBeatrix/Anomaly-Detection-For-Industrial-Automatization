1. Run SSIM filter for whole images

2. Run YOLOv5 openbox detector on the result images 
 
3. Collect images and labels, which has openbox
    run select_images_with_openbox.py

4. Cropp images with image_cutting.py

needed labels and original images - they are in result_images_and_labels_with_openboxes now.

5. SSIM_filter for the openboxes
    ssim_filter/reshape.py --> rashape all images to the same size and collect them in reshaped images

    ssim_filter/execute_ssim_filter.py --> create csv with the approved images

    ssim_filter/result/save_images.py  --> Collect the original (not reshaped) images in a directory based on the csv. 