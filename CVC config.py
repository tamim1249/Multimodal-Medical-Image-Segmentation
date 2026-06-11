import os, cv2, math, time, random, argparse, warnings
from pathlib import Path
from typing import List
import numpy as np
import scipy.ndimage as ndi
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings('ignore')

class CFG:
    IMAGE_DIR = "/kaggle/input/datasets/tamimdiu/cvc-clinicdb/CVC-ClinicDB/images"
    MASK_DIR  = "/kaggle/input/datasets/tamimdiu/cvc-clinicdb/CVC-ClinicDB/masks"
    IMG_SIZE = 256
    BATCH_SIZE = 8
    EPOCHS = 30
    LR = 1e-4
    WEIGHT_DECAY = 1e-5
    BASE = 24
    DEPTH = 1
    HEADS = 4
    AMP = True
    NUM_WORKERS = 2
    SEED = 42
    CHECKPOINT = "best_cvc_model.pth"
    FIG_DIR = "/kaggle/working/cvc_figures"
    OUT_DIR = "/kaggle/working/cvc_outputs"
    THRESHOLD = 0.5
    MAX_TRAIN = 0   # 0 means all; set 100 for quick debug


def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def list_pairs(image_dir, mask_dir):
    exts = ('.png','.jpg','.jpeg','.bmp','.tif','.tiff')
    imgs = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(exts)])
    pairs = []
    for f in imgs:
        ip = os.path.join(image_dir, f); mp = os.path.join(mask_dir, f)
        if os.path.exists(mp): pairs.append((ip, mp, f))
    if not pairs: raise RuntimeError('No image-mask pairs found. Check paths.')
    return pairs


def make_boundary(mask, iterations=1):
    m = (mask > 0.5).astype(np.uint8)
    if m.sum() == 0: return np.zeros_like(m, dtype=np.float32)
    dil = ndi.binary_dilation(m, iterations=iterations)
    ero = ndi.binary_erosion(m, iterations=iterations)
    return (dil.astype(np.uint8) - ero.astype(np.uint8)).astype(np.float32)


def augment(img, mask, bnd):
    if random.random() < 0.5:
        img = cv2.flip(img, 1); mask = cv2.flip(mask, 1); bnd = cv2.flip(bnd, 1)
    if random.random() < 0.5:
        img = cv2.flip(img, 0); mask = cv2.flip(mask, 0); bnd = cv2.flip(bnd, 0)
    if random.random() < 0.35:
        angle = random.uniform(-25, 25)
        h,w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2,h/2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpAffine(mask, M, (w,h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        bnd = cv2.warpAffine(bnd, M, (w,h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    if random.random() < 0.45:
        alpha = random.uniform(0.8, 1.2); beta = random.uniform(-20, 20)
        img = np.clip(img.astype(np.float32)*alpha + beta, 0, 255).astype(np.uint8)
    if random.random() < 0.25:
        k = random.choice([3,5]); img = cv2.GaussianBlur(img, (k,k), 0)
    return img, mask, bnd

