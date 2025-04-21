import cv2
import numpy as np
import os
from tqdm import tqdm

path = './data/train/'
dest = './edge_maps/train'

def apply_self_quotient(image_path, kernel_size=8, sigma=30):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    smoothed_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    smoothed_image = smoothed_image.astype(np.float32) + 1e-8  
    quotient_image = np.divide(image.astype(np.float32), smoothed_image)
    quotient_image = cv2.normalize(quotient_image, None, 0, 255, cv2.NORM_MINMAX)
    quotient_image = np.uint8(quotient_image)
    
    return quotient_image

def apply_morphological_operations(image, kernel_size=3):

    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    morph_image = cv2.dilate(binary_image, kernel, iterations=1)
    
    inverted_image = cv2.bitwise_not(morph_image)
    
    return inverted_image

def invert_image(path):
    
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    inverted_image = cv2.bitwise_not(image)
    return inverted_image

def extract_edges_and_save_to_folder(path):

    ls_path = sorted(os.listdir(path), key=int)
    for i in tqdm(ls_path):
        sub_path = os.path.join(path, i)
        ls_sub_path = sorted(os.listdir(sub_path), key=lambda x: int(x.split('.')[0]))

        for j in ls_sub_path:
            img_path = os.path.join(sub_path, j)
            self_quotient = apply_self_quotient(img_path, kernel_size=15, sigma=30)
            dilated_self_quotient = apply_morphological_operations(self_quotient)
            dest_sub_path = os.path.join(dest, str(int(i)))
            os.makedirs(dest_sub_path, exist_ok=True)
            dest_path = os.path.join(dest_sub_path, j)
            cv2.imwrite(dest_path, dilated_self_quotient)

extract_edges_and_save_to_folder(path)



