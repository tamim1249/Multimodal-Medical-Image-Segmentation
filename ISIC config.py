"
import os, argparse, random, math, csv
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------- Config ----------------
class CFG:
    TRAIN_IMG_DIR = "/kaggle/input/datasets/trantoanthang/isic-2018/data/images/train"
    TRAIN_MASK_DIR = "/kaggle/input/datasets/trantoanthang/isic-2018/data/annotations/train"
    VAL_IMG_DIR = "/kaggle/input/datasets/trantoanthang/isic-2018/data/images/val"
    VAL_MASK_DIR = "/kaggle/input/datasets/trantoanthang/isic-2018/data/annotations/val"
    TEST_IMG_DIR = "/kaggle/input/datasets/trantoanthang/isic-2018/data/images/test"
    TEST_MASK_DIR = "/kaggle/input/datasets/trantoanthang/isic-2018/data/annotations/test"
    OUT_DIR = "/kaggle/working/isic_outputs"
    FIG_DIR = "/kaggle/working/isic_figures"
    IMG_SIZE = 224
    BATCH_SIZE = 8
    EPOCHS = 15
    LR = 1e-3
    BASE = 16
    NUM_WORKERS = 2
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# ---------------- Dataset ----------------
def collect_pairs(img_dir, mask_dir, max_items=None):
    img_dir, mask_dir = Path(img_dir), Path(mask_dir)
    imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg")))
    pairs = []
    for ip in imgs:
        mp = mask_dir / (ip.stem + "_segmentation.png")
        if mp.exists():
            pairs.append((str(ip), str(mp)))
    if max_items:
        pairs = pairs[:max_items]
    return pairs
