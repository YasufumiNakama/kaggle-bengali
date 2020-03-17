# ====================================================
# Flags for train
# ====================================================

DEBUG = False
USE_APEX = True
#SHOW_AUG = False
#SHOW_PLOT = False
Cyclic = False
ID = 'image_id'
target_cols = ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']
FOLD = 0
TRAIN_FOLDS = './folds.csv'
#MODEL = 'resnet50'
MODEL = 'efficientnet-b2'
#MODEL = 'densenet121'
ROOT = '../input/bengaliai-cv19/'
n_grapheme = 168
n_vowel = 11
n_consonant = 7
n_origin = 1295
n_total = n_grapheme + n_vowel + n_consonant + n_origin

train_params = {
    'n_splits': 5,
    'n_epochs': 300,
    'lr': 1e-3,
    'base_lr': 1e-4,
    'max_lr': 5e-4, # 3e-3
    'step_factor': 150, # n_epochs/2
    'train_batch_size': 512, #256, 512, 768, 1024
    'test_batch_size': 512, #256, 512, 768, 1024
    'accumulation_steps': 1,
}

N_JOBS = 6
ALPHA = 0.4
USE_MISH = False
USE_CUTMIX = True
USE_MIXUP = True
# ====================================================
# Library
# ====================================================

import sys
sys.path.append('../input/pytorch-pretrained-models/repository/pretrained-models.pytorch-master')

import gc
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
from PIL import Image
import numpy as np
import pandas as pd
import scipy as sp

import sklearn.metrics

#from fastprogress import master_bar, progress_bar
from functools import partial

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

import pretrainedmodels

from albumentations import (
    Compose, HorizontalFlip, IAAAdditiveGaussianNoise, Normalize, OneOf,
    RandomBrightness, RandomContrast, Resize, VerticalFlip, Rotate, ShiftScaleRotate,
    RandomBrightnessContrast, OpticalDistortion, GridDistortion, ElasticTransform, Cutout,
    IAAPerspective, IAAAffine,
)
from albumentations.pytorch import ToTensorV2, ToTensor

if USE_APEX:
    from apex import amp

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# ====================================================
# Utils
# ====================================================

@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f'[{name}] start')
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')

    
def init_logger(log_file='train.log'):
    from logging import getLogger, DEBUG, FileHandler,  Formatter,  StreamHandler
    
    log_format = '%(asctime)s %(levelname)s %(message)s'
    
    stream_handler = StreamHandler()
    stream_handler.setLevel(DEBUG)
    stream_handler.setFormatter(Formatter(log_format))
    
    file_handler = FileHandler(log_file)
    file_handler.setFormatter(Formatter(log_format))
    
    logger = getLogger('Bengali')
    logger.setLevel(DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    
    return logger

LOG_FILE = 'bengali-train.log'
LOGGER = init_logger(LOG_FILE)


def seed_torch(seed=777):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

SEED = 777
seed_torch(SEED)


from torch.optim.lr_scheduler import _LRScheduler

class CyclicLR(_LRScheduler):
    def __init__(self, optimizer, base_lr, max_lr, step_size, gamma=0.99, mode='triangular', last_epoch=-1):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.gamma = gamma
        self.mode = mode
        assert mode in ['triangular', 'triangular2', 'exp_range']
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        new_lr = []
        # make sure that the length of base_lrs doesn't change. Dont care about the actual value
        for base_lr in self.base_lrs:
            cycle = np.floor(1 + self.last_epoch / (2 * self.step_size))
            x = np.abs(float(self.last_epoch) / self.step_size - 2 * cycle + 1)
            if self.mode == 'triangular':
                lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))
            elif self.mode == 'triangular2':
                lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) / float(2 ** (cycle - 1))
            elif self.mode == 'exp_range':
                lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * (self.gamma ** (self.last_epoch))
            new_lr.append(lr)
        return new_lr

# =================================================================
# RandomAugMix
# https://www.kaggle.com/haqishen/augmix-based-on-albumentations
# =================================================================

import albumentations
from PIL import Image, ImageOps, ImageEnhance
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.augmentations import functional as F


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
    Returns:
    An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
    Returns:
    A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)


def shear_y(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)


def translate_x(pil_img, level):
    level = int_parameter(sample_level(level), pil_img.size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)


def translate_y(pil_img, level):
    level = int_parameter(sample_level(level), pil_img.size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


augmentations = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y
]

augmentations_all = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, color, contrast, brightness, sharpness
]

def normalize(image):
    """Normalize input image channel-wise to zero mean and unit variance."""
    return image - 127

def apply_op(image, op, severity):
    #   image = np.clip(image, 0, 255)
    pil_img = Image.fromarray(np.uint8(image))  # Convert to PIL.Image
    pil_img = op(pil_img, severity)
    return np.asarray(pil_img)

def augment_and_mix(image, severity=3, width=3, depth=-1, alpha=1.):
    """Perform AugMix augmentations and compute mixture.
    Args:
    image: Raw input image as float32 np.ndarray of shape (h, w, c)
    severity: Severity of underlying augmentation operators (between 1 to 10).
    width: Width of augmentation chain
    depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
      from [1, 3]
    alpha: Probability coefficient for Beta and Dirichlet distributions.
    Returns:
    mixed: Augmented and mixed image.
    """
    ws = np.float32(
      np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))

    mix = np.zeros_like(image).astype(np.float32)
    for i in range(width):
        image_aug = image.copy()
        depth = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(augmentations)
            image_aug = apply_op(image_aug, op, severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * image_aug
#         mix += ws[i] * normalize(image_aug)

    mixed = (1 - m) * image + m * mix
#     mixed = (1 - m) * normalize(image) + m * mix
    return mixed


class RandomAugMix(ImageOnlyTransform):

    def __init__(self, severity=3, width=3, depth=-1, alpha=1., always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.severity = severity
        self.width = width
        self.depth = depth
        self.alpha = alpha

    def apply(self, image, **params):
        image = augment_and_mix(
            image,
            self.severity,
            self.width,
            self.depth,
            self.alpha
        )
        return image


from albumentations.core.transforms_interface import DualTransform


class GridMask(DualTransform):
    """GridMask augmentation for image classification and object detection.
    
    Author: Qishen Ha
    Email: haqishen@gmail.com
    2020/01/29

    Args:
        num_grid (int): number of grid in a row or column.
        fill_value (int, float, lisf of int, list of float): value for dropped pixels.
        rotate ((int, int) or int): range from which a random angle is picked. If rotate is a single int
            an angle is picked from (-rotate, rotate). Default: (-90, 90)
        mode (int):
            0 - cropout a quarter of the square of each grid (left top)
            1 - reserve a quarter of the square of each grid (left top)
            2 - cropout 2 quarter of the square of each grid (left top & right bottom)

    Targets:
        image, mask

    Image types:
        uint8, float32

    Reference:
    |  https://arxiv.org/abs/2001.04086
    |  https://github.com/akuxcw/GridMask
    """

    def __init__(self, num_grid=3, fill_value=0, rotate=0, mode=0, always_apply=False, p=0.5):
        super(GridMask, self).__init__(always_apply, p)
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)
        if isinstance(rotate, int):
            rotate = (-rotate, rotate)
        self.num_grid = num_grid
        self.fill_value = fill_value
        self.rotate = rotate
        self.mode = mode
        self.masks = None
        self.rand_h_max = []
        self.rand_w_max = []

    def init_masks(self, height, width):
        if self.masks is None:
            self.masks = []
            n_masks = self.num_grid[1] - self.num_grid[0] + 1
            for n, n_g in enumerate(range(self.num_grid[0], self.num_grid[1] + 1, 1)):
                grid_h = height / n_g
                grid_w = width / n_g
                this_mask = np.ones((int((n_g + 1) * grid_h), int((n_g + 1) * grid_w))).astype(np.uint8)
                for i in range(n_g + 1):
                    for j in range(n_g + 1):
                        this_mask[
                             int(i * grid_h) : int(i * grid_h + grid_h / 2),
                             int(j * grid_w) : int(j * grid_w + grid_w / 2)
                        ] = self.fill_value
                        if self.mode == 2:
                            this_mask[
                                 int(i * grid_h + grid_h / 2) : int(i * grid_h + grid_h),
                                 int(j * grid_w + grid_w / 2) : int(j * grid_w + grid_w)
                            ] = self.fill_value
                
                if self.mode == 1:
                    this_mask = 1 - this_mask

                self.masks.append(this_mask)
                self.rand_h_max.append(grid_h)
                self.rand_w_max.append(grid_w)

    def apply(self, image, mask, rand_h, rand_w, angle, **params):
        h, w = image.shape[:2]
        mask = F.rotate(mask, angle) if self.rotate[1] > 0 else mask
        mask = mask[:,:,np.newaxis] if image.ndim == 3 else mask
        image *= mask[rand_h:rand_h+h, rand_w:rand_w+w].astype(image.dtype)
        return image

    def get_params_dependent_on_targets(self, params):
        img = params['image']
        height, width = img.shape[:2]
        self.init_masks(height, width)

        mid = np.random.randint(len(self.masks))
        mask = self.masks[mid]
        rand_h = np.random.randint(self.rand_h_max[mid])
        rand_w = np.random.randint(self.rand_w_max[mid])
        angle = np.random.randint(self.rotate[0], self.rotate[1]) if self.rotate[1] > 0 else 0

        return {'mask': mask, 'rand_h': rand_h, 'rand_w': rand_w, 'angle': angle}

    @property
    def targets_as_params(self):
        return ['image']

    def get_transform_init_args_names(self):
        return ('num_grid', 'fill_value', 'rotate', 'mode')


class RandomMorph(ImageOnlyTransform):

    def __init__(self, _min=2, _max=6, element_shape=cv2.MORPH_ELLIPSE, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self._min = _min
        self._max = _max
        self.element_shape = element_shape

    def apply(self, image, **params):
        arr = np.random.randint(self._min, self._max, 2)
        kernel = cv2.getStructuringElement(self.element_shape, tuple(arr))

        if random.random() > 0.5:
            # make it thinner
            image = cv2.erode(image, kernel, iterations=1)
        else:
            # make it thicker
            image = cv2.dilate(image, kernel, iterations=1)

        return image


# =================================================================
# Dataset
# =================================================================
HEIGHT = 137
WIDTH = 236
IMG_RESIZE = 128 #224 128


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def crop_resize(img0, size=IMG_RESIZE, pad=16):
    #crop a box around pixels large than the threshold
    #some images contain line at the sides
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax,xmin:xmax]
    #remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    #make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img,(size,size))


class GraphemeDataset(Dataset):
    def __init__(self, df, label, transform=None):
        self.df = df
        self.label = label
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        label1 = self.label.grapheme_root.values[idx]
        label2 = self.label.vowel_diacritic.values[idx]
        label3 = self.label.consonant_diacritic.values[idx]
        label4 = self.label.grapheme.values[idx]
        file_path = f'../input/bengali-images/{self.df.image_id.values[idx]}.png'
        image = cv2.imread(file_path)
        image = Image.fromarray(np.uint8(image)).convert("L")
        image = np.asarray(image)
        image = cv2.resize(image, (IMG_RESIZE, IMG_RESIZE)).astype(np.float32)
        #image = cv2.resize(image, (HEIGHT, WIDTH)).astype(np.float32)
        #image = cv2.resize(image, (130, 224)).astype(np.float32)

        if self.transform:
            res = self.transform(image=image)
            #image = res['image'].astype(np.float32)
            image = res['image']
        else:
            #image = image.astype(np.float32)
            image = image
        
        image /= 255
        image = image[np.newaxis, :, :]
        image = 1 - image
        image = np.repeat(image, 3, 0)  # 1ch to 3ch
        
        return torch.tensor(image), label1, label2, label3, label4


# =================================================================
# Model weight
# =================================================================

PRETRAINED_DIR = Path('../input/pytorch-pretrained-models')

# =================================================================
# transforms
# =================================================================

def get_transforms(*, data):
    assert data in ('train', 'valid')
    
    if data == 'train':
        return Compose([
            #Resize(256, 256),
            #Rotate(limit=20, p=0.5),
            ShiftScaleRotate(rotate_limit=30, p=0.5),
            RandomMorph(p=0.5),
            #RandomAugMix(severity=3, width=3, alpha=1., p=0.7),
            GridDistortion(distort_limit=0.3, p=0.5),
            #IAAAffine(shear=20, mode='constant'),
            Cutout(p=0.5, max_h_size=12, max_w_size=12, num_holes=6),
            #Cutout(p=0.5, max_h_size=16, max_w_size=16, num_holes=8),
            #OneOf([
            #    GridMask(num_grid=3, mode=0),
            #    GridMask(num_grid=3, mode=2),
            #], p=0.5),
            #ToTensorV2(),
        ])
    
    elif data == 'valid':
        return Compose([
            #Resize(256, 256),
            #ToTensorV2(),
        ])

    
# =================================================================
# Model 
# =================================================================

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    
    def forward(self, x):
        return (x*torch.tanh(F.softplus(x)))
    
    
def convert_relu_to_mish(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, Mish())
        else:
            convert_relu_to_mish(child)
    return model


class ClassifierModule(nn.Sequential):
    def __init__(self, n_features):
        super().__init__(
            nn.BatchNorm1d(n_features),
            nn.Dropout(0.5),
            nn.Linear(n_features, n_features),
            nn.PReLU(),
            nn.BatchNorm1d(n_features),
            nn.Dropout(0.2),
            nn.Linear(n_features, n_total),
        )
        

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        
    def forward(self, x):
        return x


class CustomResNet(nn.Module):
    def __init__(self, model_name='resnet50', weights_path=None):
        assert model_name in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
        super().__init__()
        
        self.net = pretrainedmodels.__dict__[model_name](pretrained='imagenet')
        #self.net = pretrainedmodels.__dict__[model_name](pretrained=None)
        #self.net.load_state_dict(torch.load(weights_path))
        if USE_MISH:
            self.net = convert_relu_to_mish(self.net)

        n_features = self.net.last_linear.in_features
        
        #self.net.avgpool = nn.AdaptiveAvgPool2d(1)
        self.net.avgpool = GeM()
        # Classifier
        self.net.last_linear = Classifier()
        # grapheme_root
        self.fc1 = nn.Linear(n_features, n_grapheme)
        # vowel_diacritic
        self.fc2 = nn.Linear(n_features, n_vowel)
        # consonant_diacritic
        self.fc3 = nn.Linear(n_features, n_consonant)
        
    def forward(self, x):
        x = self.net(x)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        return x1, x2, x3


class CustomSENet(nn.Module):
    def __init__(self, model_name='se_resnet50', weights_path=None):
        assert model_name in ('senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d')
        super().__init__()
        
        self.net = pretrainedmodels.__dict__[model_name](pretrained=None)
        self.net.load_state_dict(torch.load(weights_path))
        if USE_MISH:
            self.net = convert_relu_to_mish(self.net)

        n_features = self.net.last_linear.in_features
        
        self.net.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Classifier
        self.net.last_linear = Classifier()
        # grapheme_root
        self.fc1 = nn.Linear(n_features, n_grapheme)
        # vowel_diacritic
        self.fc2 = nn.Linear(n_features, n_vowel)
        # consonant_diacritic
        self.fc3 = nn.Linear(n_features, n_consonant)
        
    def forward(self, x):
        x = self.net(x)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        return x1, x2, x3


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool1 = GeM()
        self.avgpool2 = GeM()
        self.avgpool3 = GeM()
        #self.dropout1 = nn.Dropout(0.1)
        #self.dropout2 = nn.Dropout(0.1)
        #self.dropout3 = nn.Dropout(0.1)
        #self.fc = nn.Linear(512 * block.expansion, num_classes)
        # grapheme_root
        self.fc1 = nn.Linear(512 * block.expansion, n_grapheme)
        # vowel_diacritic
        self.fc2 = nn.Linear(512 * block.expansion, n_vowel)
        # consonant_diacritic
        self.fc3 = nn.Linear(512 * block.expansion, n_consonant)
        # grapheme
        self.fc4 = nn.Linear(512 * block.expansion * 3, n_origin)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #x = self.avgpool(x)
        #x = torch.flatten(x, 1)
        #x = self.fc(x)

        x1 = self.avgpool1(x)
        x2 = self.avgpool2(x)
        x3 = self.avgpool3(x)
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x3 = torch.flatten(x3, 1)
        #x1 = self.dropout1(x1)
        #x2 = self.dropout2(x2)
        #x3 = self.dropout3(x3)

        h_conc = torch.cat((x1, x2, x3), 1)
        x4 = self.fc4(h_conc)

        x1 = self.fc1(x1)
        x2 = self.fc2(x2)
        x3 = self.fc3(x3)

        return x1, x2, x3, x4

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


# =================================================================
# efficientnet-b2
# https://github.com/lukemelas/EfficientNet-PyTorch/tree/master/efficientnet_pytorch
# =================================================================
import re
import math
import collections
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo

########################################################################
############### HELPERS FUNCTIONS FOR MODEL ARCHITECTURE ###############
########################################################################


# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',
    'num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'drop_connect_rate', 'image_size'])

# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'stride', 'se_ratio'])

# Change namedtuple defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def round_filters(filters, global_params):
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """ Round number of filters based on depth multiplier. """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


def get_same_padding_conv2d(image_size=None):
    """ Chooses static padding if you have specified an image size, and dynamic padding otherwise.
        Static padding is necessary for ONNX exporting of models. """
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)


class Conv2dDynamicSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a dynamic image size """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv2dStaticSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a fixed image size"""

    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = image_size if type(image_size) == list else [image_size, image_size]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class Identity(nn.Module):
    def __init__(self, ):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


########################################################################
############## HELPERS FUNCTIONS FOR LOADING MODEL PARAMS ##############
########################################################################


def efficientnet_params(model_name):
    """ Map EfficientNet model name to parameter coefficients. """
    params_dict = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    }
    return params_dict[model_name]


class BlockDecoder(object):
    """ Block Decoder for readability, straight from the official TensorFlow repository """

    @staticmethod
    def _decode_block_string(block_string):
        """ Gets a block through a string notation of arguments. """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=[int(options['s'][0])])

    @staticmethod
    def _encode_block_string(block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """
        Decodes a list of string notations to specify blocks inside the network.
        :param string_list: a list of strings, each string is a notation of block
        :return: a list of BlockArgs namedtuples of block args
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """
        Encodes a list of BlockArgs to a list of strings.
        :param blocks_args: a list of BlockArgs namedtuples of block args
        :return: a list of strings, each string is a notation of block
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


def efficientnet(width_coefficient=None, depth_coefficient=None, dropout_rate=0.2,
                 drop_connect_rate=0.2, image_size=None, num_classes=1000):
    """ Creates a efficientnet model. """

    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        # data_format='channels_last',  # removed, this is always true in PyTorch
        num_classes=num_classes,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None,
        image_size=image_size,
    )

    return blocks_args, global_params


def get_model_params(model_name, override_params):
    """ Get the block args and global params for a given model """
    if model_name.startswith('efficientnet'):
        w, d, s, p = efficientnet_params(model_name)
        # note: all models have drop connect rate = 0.2
        blocks_args, global_params = efficientnet(
            width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' % model_name)
    if override_params:
        # ValueError will be raised here if override_params has fields not included in global_params.
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params


def load_pretrained_weights(model, model_name, load_fc=True, advprop=False):
    """ Loads pretrained weights, and downloads if loading for the first time. """
    # AutoAugment or Advprop (different preprocessing)
    efficientnet_pretrained_path = '../input/pytorch-pretrained-models/efficientnet-b2-8bb594d6.pth'
    state_dict = torch.load(efficientnet_pretrained_path)
    if load_fc:
        model.load_state_dict(state_dict, strict=False)
    else:
        state_dict.pop('_fc.weight')
        state_dict.pop('_fc.bias')
        res = model.load_state_dict(state_dict, strict=False)
        assert set(res.missing_keys) == set(['_fc.weight', '_fc.bias']), 'issue loading pretrained weights'
    print('Loaded pretrained weights for {}'.format(model_name))


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block
    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above
    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods
    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks
    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')
    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        #self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        #self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()

        self.avgpool1 = GeM()
        self.avgpool2 = GeM()
        self.avgpool3 = GeM()

        # grapheme_root
        self.fc1 = nn.Linear(out_channels, n_grapheme)
        # vowel_diacritic
        self.fc2 = nn.Linear(out_channels, n_vowel)
        # consonant_diacritic
        self.fc3 = nn.Linear(out_channels, n_consonant)
        # grapheme
        self.fc4 = nn.Linear(out_channels * 3, n_origin)

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)


    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        bs = inputs.size(0)
        # Convolution layers
        x = self.extract_features(inputs)

        # Pooling and final linear layer
        #x = self._avg_pooling(x)
        #x = x.view(bs, -1)
        x1 = self.avgpool1(x)
        x2 = self.avgpool2(x)
        x3 = self.avgpool3(x)
        x1 = x1.view(bs, -1)
        x2 = x2.view(bs, -1)
        x3 = x3.view(bs, -1)

        h_conc = torch.cat((x1, x2, x3), 1)
        x4 = self.fc4(h_conc)

        #x = self._dropout(x)
        #x = self._fc(x)
        x1 = self.fc1(x1)
        x2 = self.fc2(x2)
        x3 = self.fc3(x3)
        return x1, x2, x3, x4

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, advprop=False, num_classes=1000, in_channels=3):
        model = cls.from_name(model_name, override_params={'num_classes': num_classes})
        load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000), advprop=advprop)
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size = model._global_params.image_size)
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name. """
        valid_models = ['efficientnet-b'+str(i) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))


class Efficientnetb2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = EfficientNet.from_pretrained('efficientnet-b2')

    def forward(self, x):
        return self.net(x)


# =================================================================
# densenet
# =================================================================
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor
from torch.jit.annotations import List


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input):
        # type: (List[Tensor]) -> Tensor
        def closure(*inputs):
            return self.bn_function(*inputs)

        return cp.checkpoint(closure, input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (List[Tensor]) -> (Tensor)
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (Tensor) -> (Tensor)
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        self.avgpool1 = GeM()
        self.avgpool2 = GeM()
        self.avgpool3 = GeM()

        # Linear layer
        # self.classifier = nn.Linear(num_features, num_classes)
        # grapheme_root
        self.fc1 = nn.Linear(num_features, n_grapheme)
        # vowel_diacritic
        self.fc2 = nn.Linear(num_features, n_vowel)
        # consonant_diacritic
        self.fc3 = nn.Linear(num_features, n_consonant)
        # grapheme
        self.fc4 = nn.Linear(num_features * 3, n_origin)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)

        #out = F.adaptive_avg_pool2d(out, (1, 1))
        #out = torch.flatten(out, 1)
        x1 = self.avgpool1(x)
        x2 = self.avgpool2(x)
        x3 = self.avgpool3(x)
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x3 = torch.flatten(x3, 1)

        h_conc = torch.cat((x1, x2, x3), 1)
        x4 = self.fc4(h_conc)

        #out = self.classifier(out)
        x1 = self.fc1(x1)
        x2 = self.fc2(x2)
        x3 = self.fc3(x3)

        return x1, x2, x3, x4


def _load_state_dict(model, model_url, progress):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)


def _densenet(arch, growth_rate, block_config, num_init_features, pretrained, progress,
              **kwargs):
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model


def densenet121(pretrained=False, progress=True, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
                     **kwargs)


# =================================================================
# Prepare data
# =================================================================
from sklearn import preprocessing

LOGGER.debug(f'Fold: {FOLD}')
LOGGER.debug(f'Model: {MODEL}')
LOGGER.debug(f'Train params: {train_params}')


train = pd.read_csv(ROOT+'train.csv')
#le = preprocessing.LabelEncoder()
#le.fit(train['grapheme'])
#train['grapheme'] = le.transform(train['grapheme'])
balance_col = 'grapheme_root'
NUM_CLASS = train[balance_col].nunique()
class_map_df = pd.DataFrame({balance_col: train[balance_col].unique()}).reset_index()
class_map_df.columns = ['label', balance_col]
class_map = dict(class_map_df[[balance_col, 'label']].values)

from torch.utils.data.sampler import Sampler
# see torch/utils/data/sampler.py
class BalanceSampler(Sampler):
    def __init__(self, df, length):
        self.length = length
        group = []
        grapheme_gb = df.groupby([balance_col])
        for k, i in class_map.items():
            g = grapheme_gb.get_group(k).index
            group.append(list(g))
            assert(len(g)>0)
        self.group=group

    def __iter__(self):
        index = []
        n = 0
        is_loop = True
        while is_loop:
            num_class = NUM_CLASS
            c = np.arange(num_class)
            np.random.shuffle(c)
            for t in c:
                i = np.random.choice(self.group[t])
                index.append(i)
                n += 1
                if n == self.length:
                    is_loop = False
                    break
        return iter(index)

    def __len__(self):
        return self.length


with timer('Prepare train and valid sets'):
    with timer('  * prepare data'):
        train = pd.read_csv(ROOT+'train.csv')
        le = preprocessing.LabelEncoder()
        le.fit(train['grapheme'])
        train['grapheme'] = le.transform(train['grapheme'])
        weight_lst = []
        for c in ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']:
            count = pd.DataFrame(train[c].value_counts()).reset_index()
            count.columns = ['label', 'count']
            count = count.sort_values('label')
            count['sample_weight'] = 1 / (count['count']/sum(count['count']))
            count['sample_weight'] = count['sample_weight'] / count['sample_weight'].mean()
            if c=='grapheme_root':
                count['sample_weight'] = np.clip(count['sample_weight'], 1, 3)
            else:
                count['sample_weight'] = np.clip(count['sample_weight'], 1, 2)
            class_weight_dict = {}
            for k, v in zip(count['label'].values, count['sample_weight'].values):
                class_weight_dict[k] = v
            #weight_lst.append(class_weight_dict)
            LOGGER.debug(f'class_weight_dict: {class_weight_dict}')
            weight_lst.append(list(class_weight_dict.values()))
            train[f'{c}_weight'] = train[c].map(class_weight_dict).values
        if DEBUG:
            train = train.sample(n=3000, random_state=0).reset_index(drop=True)
    
    with timer('  * make folds'):
        folds = train.copy()
        train_labels = train[target_cols+['grapheme']].values
        kf = MultilabelStratifiedKFold(n_splits=5, random_state=777)
        #kf = MultilabelStratifiedKFold(n_splits=5, random_state=42)
        for fold, (train_index, val_index) in enumerate(kf.split(train.values, train_labels)):
            folds.loc[val_index, 'fold'] = int(fold)
        folds['fold'] = folds['fold'].astype(int)
        folds.to_csv('folds.csv', index=None)
    
    with timer('  * load folds csv'):
        folds = pd.read_csv(TRAIN_FOLDS)
        trn_idx = folds[folds['fold'] != FOLD].index
        val_idx = folds[folds['fold'] == FOLD].index
    
    with timer('  * define dataset'):
        train_dataset = GraphemeDataset(folds.loc[trn_idx].reset_index(drop=True), folds.loc[trn_idx][target_cols+['grapheme']], 
                                        transform=get_transforms(data='train'))
                                       #transform=None)
        valid_dataset = GraphemeDataset(folds.loc[val_idx].reset_index(drop=True), folds.loc[val_idx][target_cols+['grapheme']], 
                                        #transform=get_transforms(data='valid'))
                                       transform=None)
        
    with timer('  * define dataloader'):
        train_loader = DataLoader(train_dataset, sampler=BalanceSampler(folds.loc[trn_idx].reset_index(drop=True), len(trn_idx)),
                                  batch_size=train_params['train_batch_size'],
                                  num_workers=N_JOBS)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=train_params['test_batch_size'],
                                  shuffle=False, num_workers=N_JOBS)
        
LOGGER.debug(f'train size: {len(train_dataset)}, valid size: {len(valid_dataset)}')


# =================================================================
# Train
# =================================================================

import torch.nn.functional as F
from torch.nn.parameter import Parameter


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return (x*torch.tanh(F.softplus(x)))


def convert_relu_to_mish(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, Mish())
        else:
            convert_relu_to_mish(child)
    return model


def ohem_loss(cls_pred, cls_target):
    """
    https://www.kaggle.com/c/bengaliai-cv19/discussion/128637
    """
    batch_size = cls_pred.size(0) 
    ohem_cls_loss = F.cross_entropy(cls_pred, cls_target, reduction='none', ignore_index=-1)
    sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
    keep_num = min(sorted_ohem_loss.size()[0], int(batch_size*0.7))
    if keep_num < sorted_ohem_loss.size()[0]:
        keep_idx_cuda = idx[:keep_num]
        ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
    cls_loss = ohem_cls_loss.sum() / keep_num
    return cls_loss


def ohem_loss1(cls_pred, cls_target):
    #sample_weight = torch.tensor(list(weight_lst[0].values())).to(device)
    sample_weight = torch.tensor(weight_lst[0], dtype=torch.float32).to(device)
    batch_size = cls_pred.size(0) 
    ohem_cls_loss = F.cross_entropy(cls_pred, cls_target, weight=sample_weight, reduction='none', ignore_index=-1)
    sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
    keep_num = min(sorted_ohem_loss.size()[0], int(batch_size*0.7))
    if keep_num < sorted_ohem_loss.size()[0]:
        keep_idx_cuda = idx[:keep_num]
        ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
    cls_loss = ohem_cls_loss.sum() / keep_num
    return cls_loss


def ohem_loss2(cls_pred, cls_target):
    #sample_weight = torch.tensor(list(weight_lst[1].values())).to(device)
    sample_weight = torch.tensor(weight_lst[1], dtype=torch.float32).to(device)
    batch_size = cls_pred.size(0) 
    ohem_cls_loss = F.cross_entropy(cls_pred, cls_target, weight=sample_weight, reduction='none', ignore_index=-1)
    sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
    keep_num = min(sorted_ohem_loss.size()[0], int(batch_size*0.7))
    if keep_num < sorted_ohem_loss.size()[0]:
        keep_idx_cuda = idx[:keep_num]
        ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
    cls_loss = ohem_cls_loss.sum() / keep_num
    return cls_loss


def ohem_loss3(cls_pred, cls_target):
    #sample_weight = torch.tensor(list(weight_lst[2].values())).to(device)
    sample_weight = torch.tensor(weight_lst[2], dtype=torch.float32).to(device)
    batch_size = cls_pred.size(0) 
    ohem_cls_loss = F.cross_entropy(cls_pred, cls_target, weight=sample_weight, reduction='none', ignore_index=-1)
    sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
    keep_num = min(sorted_ohem_loss.size()[0], int(batch_size*0.7))
    if keep_num < sorted_ohem_loss.size()[0]:
        keep_idx_cuda = idx[:keep_num]
        ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
    cls_loss = ohem_cls_loss.sum() / keep_num
    return cls_loss


def rand_bbox(size, lam):

    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(data, targets1, targets2, targets3, targets4, alpha):

    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets1 = targets1[indices]
    shuffled_targets2 = targets2[indices]
    shuffled_targets3 = targets3[indices]
    shuffled_targets4 = targets4[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    targets = [targets1, shuffled_targets1, targets2, shuffled_targets2, targets3, shuffled_targets3, targets4, shuffled_targets4, lam]

    return data, targets


def cutmix_criterion(preds1, preds2, preds3, preds4, targets, criterion1, criterion2, criterion3, criterion4):
    targets1, targets2, targets3, targets4, targets5, targets6, targets7, targets8, lam = targets[0], targets[1], targets[2], targets[3], targets[4], targets[5], targets[6], targets[7], targets[8]
    # criterion = nn.CrossEntropyLoss(reduction='mean')
    # criterion1 = ohem_loss1
    # criterion2 = ohem_loss2
    # criterion3 = ohem_loss3
    #criterion1 = nn.CrossEntropyLoss(weight=torch.tensor(weight_lst[0], dtype=torch.float32).to(device), reduction='mean')
    #criterion2 = nn.CrossEntropyLoss(weight=torch.tensor(weight_lst[1], dtype=torch.float32).to(device), reduction='mean')
    #criterion3 = nn.CrossEntropyLoss(weight=torch.tensor(weight_lst[2], dtype=torch.float32).to(device), reduction='mean')
    #criterion4 = nn.CrossEntropyLoss(reduction='mean')
    return [ lam * criterion1(preds1, targets1) + (1 - lam) * criterion1(preds1, targets2), lam * criterion2(preds2, targets3) + (1 - lam) * criterion2(preds2, targets4), lam * criterion3(preds3, targets5) + (1 - lam) * criterion3(preds3, targets6), lam * criterion4(preds4, targets7) + (1 - lam) * criterion4(preds4, targets8) ]


def mixup(data, targets1, targets2, targets3, targets4, alpha):

    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets1 = targets1[indices]
    shuffled_targets2 = targets2[indices]
    shuffled_targets3 = targets3[indices]
    shuffled_targets4 = targets4[indices]

    lam = np.random.beta(alpha, alpha)
    data = data * lam + shuffled_data * (1 - lam)
    targets = [targets1, shuffled_targets1, targets2, shuffled_targets2, targets3, shuffled_targets3, targets4, shuffled_targets4, lam]

    return data, targets


def mixup_criterion(preds1, preds2, preds3, preds4, targets, criterion1, criterion2, criterion3, criterion4):
    targets1, targets2, targets3, targets4, targets5, targets6, targets7, targets8, lam = targets[0], targets[1], targets[2], targets[3], targets[4], targets[5], targets[6], targets[7], targets[8]
    # criterion = nn.CrossEntropyLoss(reduction='mean')
    # criterion1 = ohem_loss1
    # criterion2 = ohem_loss2
    # criterion3 = ohem_loss3
    #criterion1 = nn.CrossEntropyLoss(weight=torch.tensor(weight_lst[0], dtype=torch.float32).to(device), reduction='mean')
    #criterion2 = nn.CrossEntropyLoss(weight=torch.tensor(weight_lst[1], dtype=torch.float32).to(device), reduction='mean')
    #criterion3 = nn.CrossEntropyLoss(weight=torch.tensor(weight_lst[2], dtype=torch.float32).to(device), reduction='mean')
    #criterion4 = nn.CrossEntropyLoss(reduction='mean')
    return [ lam * criterion1(preds1, targets1) + (1 - lam) * criterion1(preds1, targets2), lam * criterion2(preds2, targets3) + (1 - lam) * criterion2(preds2, targets4), lam * criterion3(preds3, targets5) + (1 - lam) * criterion3(preds3, targets6) , lam * criterion4(preds4, targets7) + (1 - lam) * criterion4(preds4, targets8)]


with timer('Train model'):
    
    n_epochs = train_params['n_epochs']
    lr = train_params['lr']
    base_lr = train_params['base_lr']
    max_lr = train_params['max_lr']
    step_factor = train_params['step_factor']
    test_batch_size = train_params['test_batch_size']
    accumulation_steps = train_params['accumulation_steps']
    
    if MODEL=='resnet50':
        model = resnet50()
        resnet50_pretrained_path = PRETRAINED_DIR / 'resnet50-19c8e357.pth'
        pretrained_dict = torch.load(resnet50_pretrained_path)
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        #model = CustomResNet(model_name=MODEL)
    elif MODEL=='efficientnet-b2':
        model = Efficientnetb2()
    elif MODEL=='densenet121':
        model = densenet121()
        densenet121_pretrained_path = PRETRAINED_DIR / 'densenet121.pth'
        #densenet121_pretrained_path = '../input/densenet121/densenet121.pth'
        state_dict = torch.load(densenet121_pretrained_path)
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    else:
        LOGGER.debug(f'{MODEL} is not implemented')
    LOGGER.debug('===========================================================')
    LOGGER.debug(model)
    LOGGER.debug('===========================================================')
    if USE_MISH:
        model = convert_relu_to_mish(model)
    model.to(device)
    
    #optimizer = Adam(model.parameters(), lr=lr, amsgrad=False)
    #optimizer = SGD(model.parameters(), lr=lr, weight_decay=4e-5, momentum=0.9, nesterov=True)
    if Cyclic:
        optimizer = Adam(model.parameters(), lr=lr, amsgrad=False)
        scheduler = CyclicLR(optimizer,
                             base_lr=base_lr,
                             max_lr=max_lr,
                             step_size=len(train_loader) * step_factor)
    else:
        optimizer = Adam(model.parameters(), lr=max_lr, amsgrad=False)
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        #scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=4, verbose=True, eps=5e-5)
        #scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=4, verbose=True, eps=5e-5)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.75, patience=5, verbose=True, eps=1e-6)

    if USE_APEX:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)
    
    #criterion = nn.CrossEntropyLoss()
    #criterion = ohem_loss
    #criterion1 = ohem_loss1
    criterion1 = nn.CrossEntropyLoss(weight=torch.tensor(weight_lst[0], dtype=torch.float32).to(device), reduction='mean')
    criterion2 = nn.CrossEntropyLoss(weight=torch.tensor(weight_lst[1], dtype=torch.float32).to(device), reduction='mean')
    criterion3 = nn.CrossEntropyLoss(weight=torch.tensor(weight_lst[2], dtype=torch.float32).to(device), reduction='mean')
    criterion4 = nn.CrossEntropyLoss(reduction='mean')
    #loss_weight = [0.8, 0.1, 0.1]
    loss_weight = [0.5, 0.25, 0.25, 0.25]
    #loss_weight = [0.50, 0.25, 0.25]
    #loss_weight = [0.40, 0.30, 0.30]
    best_score = 0.
    best_loss = np.inf
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(n_epochs):
        
        #if epoch==130:
            #criterion1 = ohem_loss
            #criterion2 = ohem_loss
            #criterion3 = ohem_loss
            #criterion4 = ohem_loss

        start_time = time.time()

        model.train()
        avg_loss = 0.

        optimizer.zero_grad()

        for i, (images, labels1, labels2, labels3, labels4) in enumerate(train_loader):
            if isinstance(scheduler, CyclicLR):
                scheduler.step()

            images = images.to(device)
            labels1 = labels1.to(device)
            labels2 = labels2.to(device)
            labels3 = labels3.to(device)    
            labels4 = labels4.to(device)
            
            if epoch<160:
                p = 0.8 # 0.8
            elif epoch>=160 and epoch<200:
                p = 0.6 # 0.6
            elif epoch>=200 and epoch<240:
                p = 0.4 # 0.4
            elif epoch>=240 and epoch<280:
                p = 0.2 # 0.2
            else:
                p = 0.1 # 0.1

            r = np.random.rand(1)
            if r<p:
                if np.random.rand()<0.5:
                    USE_CUTMIX = True
                else:
                    USE_MIXUP = True
            else:
                USE_CUTMIX = False
                USE_MIXUP = False

            if USE_CUTMIX:
                images, targets = cutmix(images, labels1, labels2, labels3, labels4, alpha=ALPHA)
                y_preds1, y_preds2, y_preds3, y_preds4 = model(images)
                loss_list = cutmix_criterion(y_preds1, y_preds2, y_preds3, y_preds4, targets, criterion1, criterion2, criterion3, criterion4)
                loss = loss_weight[0]*loss_list[0]+loss_weight[1]*loss_list[1]+loss_weight[2]*loss_list[2]+loss_weight[3]*loss_list[3]
            elif USE_MIXUP:
                images, targets = mixup(images, labels1, labels2, labels3, labels4, alpha=ALPHA)
                y_preds1, y_preds2, y_preds3, y_preds4 = model(images)
                loss_list = mixup_criterion(y_preds1, y_preds2, y_preds3, y_preds4, targets, criterion1, criterion2, criterion3, criterion4)
                loss = loss_weight[0]*loss_list[0]+loss_weight[1]*loss_list[1]+loss_weight[2]*loss_list[2]+loss_weight[3]*loss_list[3]
            else:
                y_preds1, y_preds2, y_preds3, y_preds4 = model(images)
                loss1 = criterion1(y_preds1, labels1)
                loss2 = criterion2(y_preds2, labels2)
                loss3 = criterion3(y_preds3, labels3)
                loss4 = criterion4(y_preds4, labels4)
                #loss1 = ohem_loss1(y_preds1, labels1)
                #loss2 = ohem_loss2(y_preds2, labels2)
                #loss3 = ohem_loss3(y_preds3, labels3)
                loss = loss_weight[0]*loss1+loss_weight[1]*loss2+loss_weight[2]*loss3+loss_weight[3]*loss4

            if USE_APEX:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            if (i+1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            avg_loss += loss.item() / accumulation_steps / len(train_loader)
            
        model.eval()
        avg_val_loss = 0.
        preds1 = np.zeros((len(valid_dataset)))
        preds2 = np.zeros((len(valid_dataset)))
        preds3 = np.zeros((len(valid_dataset)))

        for i, (images, labels1, labels2, labels3, labels4) in enumerate(valid_loader):
            
            images = images.to(device)
            labels1 = labels1.to(device)
            labels2 = labels2.to(device)
            labels3 = labels3.to(device)  
            labels4 = labels4.to(device)
            
            with torch.no_grad():
                y_preds1, y_preds2, y_preds3, y_preds4 = model(images)
            
            preds1[i * test_batch_size: (i+1) * test_batch_size] = y_preds1.argmax(1).to('cpu').numpy()
            preds2[i * test_batch_size: (i+1) * test_batch_size] = y_preds2.argmax(1).to('cpu').numpy()
            preds3[i * test_batch_size: (i+1) * test_batch_size] = y_preds3.argmax(1).to('cpu').numpy()

            loss1 = criterion1(y_preds1, labels1)
            loss2 = criterion2(y_preds2, labels2)
            loss3 = criterion3(y_preds3, labels3)
            loss4 = criterion4(y_preds4, labels4)
            #loss1 = ohem_loss1(y_preds1, labels1)
            #loss2 = ohem_loss2(y_preds2, labels2)
            #loss3 = ohem_loss3(y_preds3, labels3)
            loss = loss_weight[0]*loss1+loss_weight[1]*loss2+loss_weight[2]*loss3+loss_weight[3]*loss4

            avg_val_loss += loss.item() / len(valid_loader)
            
        scores = []
        scores.append(sklearn.metrics.recall_score(folds.loc[val_idx]['grapheme_root'].values, preds1, average='macro'))
        scores.append(sklearn.metrics.recall_score(folds.loc[val_idx]['vowel_diacritic'].values, preds2, average='macro'))
        scores.append(sklearn.metrics.recall_score(folds.loc[val_idx]['consonant_diacritic'].values, preds3, average='macro'))
        final_score = np.average(scores, weights=[2,1,1])
        
        if not isinstance(scheduler, CyclicLR):
            #scheduler.step(final_score)
            scheduler.step(avg_val_loss)

        elapsed = time.time() - start_time

        LOGGER.debug(f'  Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  final_score: {final_score:.4f}  time: {elapsed:.0f}s')
        LOGGER.debug(f'scores: {scores}')

        if final_score>best_score:
            best_score = final_score
            LOGGER.debug(f'  Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save(model.state_dict(), f'{MODEL}_fold{FOLD}_best_score.pth')

        if avg_val_loss<best_loss:
            best_loss = avg_val_loss
            LOGGER.debug(f'  Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')
            torch.save(model.state_dict(), f'{MODEL}_fold{FOLD}_best_loss.pth')
