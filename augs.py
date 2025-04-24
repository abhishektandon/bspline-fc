from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa

import torch
from torchvision import transforms

import Augmentor
import os

torch.manual_seed(42)
ia.seed(0)


path = './data/bspline_fc_1_pID/train' # initial data to which augs are added (1 image per ID): run render.py with samples_per_id=1 to obtain this data
dest = './data/bspline_vpd/train/' # path to dir where raw visual prompts data is saved

os.makedirs(dest, exist_ok=True)

perspective_transformer = transforms.RandomPerspective(distortion_scale=0.6, p=1.0)
rotater = transforms.RandomRotation(degrees=(-20, 20))
affine_transfomer = transforms.RandomAffine(degrees=(-15, 15), translate=(0.1, 0.3), scale=(0.5, 0.75))
blurrer = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))
occlusion = transforms.RandomErasing(p=1.0)
tensor = transforms.ToTensor()
to_pil_image = transforms.ToPILImage()

brightness = transforms.ColorJitter(brightness=(0.5, 2.0))
edge_seq = iaa.Sequential(iaa.OneOf([
                    #  iaa.EdgeDetect(alpha=(0, 0.7)),
                     iaa.DirectedEdgeDetect(
                         alpha=(0, 0.7), direction=(0.0, 1.0)
                     ),
                 ]))
affine = iaa.Affine(
            # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            # rotate=(-45, 45),
            # shear=(-16, 16),
            # order=[0, 1],
            # cval=(0, 255),
            # mode=ia.ALL
        )
super_pixels = iaa.Superpixels(
                        p_replace=(0, 1.0),
                        n_segments=(20, 200)
                    )
blur = iaa.Sequential(iaa.OneOf([
                    # iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)),
                    # iaa.MedianBlur(k=(3, 11)),
                ]),)
sharpen = iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))
embros  = iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))
dropout = iaa.Sequential(iaa.OneOf([
                    # iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout(
                        (0.03, 0.15), size_percent=(0.02, 0.05),
                        per_channel=0.2
                    ),
                ]))
invert = iaa.Invert(0.05, per_channel=True) # invert color channels
add = iaa.Add((-10, 10), per_channel=0.5)
mul = iaa.Multiply((0.5, 1.5), per_channel=0.5)
lin_contrast = iaa.LinearContrast((0.5, 2.0), per_channel=0.5)
gray = iaa.Grayscale(alpha=(0.0, 1.0))
elastic_transform = iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
piecewise_affine = iaa.PiecewiseAffine(scale=(0.01, 0.05))
additive_gaussian = iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05*255), per_channel=0.5
                )
augs = {'pers': perspective_transformer, 
        'rot': rotater, 
        'affine_transformer': affine_transfomer, 
        'blur': blurrer, 
    #    'occlusion_tensor': occlusion, 
        'brightness': brightness, 
        'edge_iaa': edge_seq, 
        'translate_iaa': affine, 
        'super_pixels_iaa': super_pixels, 
        'blur_iaa': blur, 
        'sharpen_iaa': sharpen, 
    #    'embros_iaa': embros, 
        'dropout_iaa': dropout,
    #    'invert_iaa': invert,
    #    'add_iaa': add, 'mul_iaa': mul,'lin_contrast_iaa': lin_contrast, 
    #    'gray_iaa': gray, 
       'elastic_transform_iaa': elastic_transform, 
       'piecewise_affine_iaa': piecewise_affine,
    #    'additive_gaussian_iaa': additive_gaussian,
       }

ls = sorted(os.listdir(path), key=int)

global_counter = 1

print("processing...")
for i in ls:
    path_i = os.path.join(path, i)
    dest_i = os.path.join(dest, f'{global_counter}')
    os.makedirs(dest_i, exist_ok=True)
    ls_path_i = sorted(os.listdir(path_i), key=lambda x: int(x.split('.')[0]))
    
    counter = 1
    for j in ls_path_i:
             
        for k in augs.keys():
            path_j = os.path.join(path_i, j)
            dest_j = os.path.join(dest_i, f'{counter}.png')
            img = Image.open(path_j).resize((256, 256)).convert('RGB')

            if counter == 1:
                img.save(dest_j)
                counter = counter + 1
                dest_j = os.path.join(dest_i, f'{counter}.png')

            if k.endswith('iaa'): # # transforms that accept np array
                imgs = np.asarray(img).reshape(1, 256, 256, 3)
                out = augs[k](images=imgs)
                # print(out.shape)
                out = Image.fromarray(out.squeeze(0)).convert('RGB')
                out.save(f'{dest_j}')

            elif k.endswith('tensor'): # transforms that accept tensors
                out = augs[k](tensor(img))
                out = to_pil_image(out)
                out.save(f'{dest_j}')

            else: # transforms that accept pil imgs
                out = augs[k](img)
                out.save(f'{dest_j}')
            
            counter = counter + 1

    global_counter = global_counter + 1

print("done")

    
    
   
