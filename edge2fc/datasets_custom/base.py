from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import cv2
import numpy as np

class ImagePathDataset(Dataset):
    def __init__(self, image_paths, image_size=(256, 256), flip=False, to_normal=False):
        self.image_size = image_size
        self.image_paths = image_paths
        self._length = len(image_paths)
        self.flip = flip
        self.to_normal = to_normal # 是否归一化到[-1, 1]

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.0
        if index >= self._length:
            index = index - self._length
            p = 1.0

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        img_path = self.image_paths[index]
        image = None
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')

        image = transform(image)

        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)

        image_name = Path(img_path).stem
        return image, image_name
    

def apply_self_quotient(image_path, kernel_size=8, sigma=30):
    # Read image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image is loaded successfully
    if image is None:
        print("Error: Could not open or find the image.")
        return
    # Apply Gaussian blur to the image
    smoothed_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    # Prevent division by zero
    smoothed_image = smoothed_image.astype(np.float32) + 1e-8
    # Calculate the self-quotient image
    quotient_image = np.divide(image.astype(np.float32), smoothed_image)
    # Normalize the result to the range [0, 255]
    quotient_image = cv2.normalize(quotient_image, None, 0, 255, cv2.NORM_MINMAX)
    quotient_image = np.uint8(quotient_image)

    return quotient_image

def apply_morphological_operations(image, operation, kernel_size=3):
    # Convert the image to grayscale if it is not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    # Binarize the image using Otsu's thresholding method
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Create a kernel for morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    if operation == 'dilation':
        morph_image = cv2.dilate(binary_image, kernel, iterations=1)
    elif operation == 'erosion':
        morph_image = cv2.erode(binary_image, kernel, iterations=1)
    else:
        raise ValueError("Invalid operation. Use 'dilation' or 'erosion'.")
    
    inverted_image = cv2.bitwise_not(morph_image)

    # print(inverted_image.shape)
    inverted_image_rgb = cv2.cvtColor(inverted_image, cv2.COLOR_BGR2RGB)
    
    pil_image = Image.fromarray(inverted_image_rgb)
    
    return pil_image


class ExtractEdgesTransform():

    def __init__(self):
        self.apply_self_quotient = apply_self_quotient
        self.apply_morphological_operations = apply_morphological_operations
    
    def __call__(self, image_path):
        self_quotient = self.apply_self_quotient(image_path, kernel_size=15, sigma=30)
        dilated_self_quotient = self.apply_morphological_operations(self_quotient, 'dilation')
        
        return dilated_self_quotient


class ImagePathDatasetWithContext(Dataset):
    def __init__(self, image_paths, image_size=(256, 256), context = False, flip=False, to_normal=False):
        self.image_size = image_size
        self.image_paths = image_paths
        self._length = len(image_paths)
        self.flip = flip
        self.to_normal = to_normal # 是否归一化到[-1, 1]
        self.context = context

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.0
        if index >= self._length:
            index = index - self._length
            p = 1.0

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        if self.context: 
            context_transform = transforms.Compose(
                [ExtractEdgesTransform(), # takes img path as input and returns the edge
                transforms.RandomHorizontalFlip(p=p),
                transforms.Resize(self.image_size),
                transforms.ToTensor()]
                )

        img_path = self.image_paths[index]
        image = None
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')

        image = transform(image)

        if self.context: 
            context_img = context_transform(img_path)

        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)

            if self.context:
                context_img = (context_img - 0.5) * 2.
                context_img.clamp_(-1., 1.)

        # print("context:", self.context)
        image_name = Path(img_path).stem
        if self.context:
            return (image, image_name), context_img
        else:
            return image, image_name

