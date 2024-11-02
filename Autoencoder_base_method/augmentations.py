import albumentations as A
import os
import torch
import torchvision.transforms as transforms

def set_augmentations(train_config):

    A_transforms_train = A.Compose([
                            A.CoarseDropout(max_holes=6, max_height= 30, max_width=30, min_holes=2, p=0.7),
                            A.OneOf([
                                A.GaussNoise(
                                var_limit=(10.0, 100.0),
                                mean=0,  
                                per_channel=False,  
                                
                                always_apply=False,  
                                p=0.5,  
                                ),
                                A.GlassBlur(
                                    sigma=0.0005,  
                                    max_delta=1,  
                                    iterations=1, 
                                    mode="fast",  #['fast', 'exact']
                                    always_apply=False,
                                    p=0.3,  
                                    ),
                                A.RandomGravel(
                                gravel_roi=(0.0, 0.0, 1.0, 1.0),  
                                number_of_patches=4, 
                                always_apply=False, 
                                p=0.3, 
                            ) ], p=0.7 ),
                            A.PixelDropout(
                                dropout_prob=0.01, 
                                per_channel=False, 
                                drop_value=None, 
                                mask_drop_value=None,  
                                always_apply=False,  
                                p=0.5,
                                ),
                            A.OneOf([
                                A.ElasticTransform(alpha=1, sigma=10, p=0.2),
                                A.Sharpen()],
                                p=0.3),
                            A.OneOf([ 
                                A.MotionBlur(blur_limit=3, p=0.2),
                                A.Blur(blur_limit=3, p=0.2 )],
                                p=0.3),             
                            A.OneOf([
                                    A.OpticalDistortion(p=0.3),
                                    A.GridDistortion(p=0.1), ],
                                p=0.3)                           
                
                            ])
                            
    A_base_transforms = A.Compose([
                            A.Resize(train_config["image_size"],train_config["image_size"]),
                            A.RandomRotate90(p=0.5),
                            A.Flip(p = 0.5),
                            A.RandomBrightnessContrast(
                                brightness_limit=(-0.1, 0.1),  
                                contrast_limit=(-0.1, 0.1), 
                                brightness_by_max=True, 
                                always_apply=False, 
                                p=0.6,  
                                ),
                            A.HueSaturationValue(
                            hue_shift_limit=5,  
                            sat_shift_limit=10, 
                            val_shift_limit=5,  
                            always_apply=False, 
                            p=0.6, 
                            )
                        ])



    final_transform = transforms.Compose([     
        transforms.ToTensor()            
    ])

    # Save configurations
    save_augmentations_to_file(A_transforms_train, A_base_transforms, final_transform, train_config["result_dir"])

    return A_transforms_train, A_base_transforms, final_transform


def save_augmentations_to_file(A_transforms_train, A_base_transforms, final_transform, result_dir):
    #Save transforms to file
    for  transform in [A_transforms_train, A_base_transforms, final_transform]:
        with open(os.path.join(result_dir,f"transforms"), 'a') as file:
            file.write("Next transform:" + '\n')
            for t in transform.transforms:
                file.write(str(t) + '\n')
