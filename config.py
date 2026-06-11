import os, glob, time, random, argparse, logging, warnings
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("BMT-HARNet++")

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Publication style
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

COLORS = {
    "wt":    "#2196F3",
    "tc":    "#FF9800",
    "et":    "#F44336",
    "train": "#1976D2",
    "val":   "#E53935",
}


# =============================================================================
# Config
# =============================================================================
class CFG:
    DATASET_ROOT = "/kaggle/input/datasets/awsaf49/brats20-dataset-training-validation"
    MODALITIES   = ["flair", "t1", "t1ce", "t2"]
    NUM_CLASSES  = 3

    PATCH_SIZE   = (64, 64, 64)
    BATCH_SIZE   = 1
    NUM_WORKERS  = 2
    VAL_SPLIT    = 0.10
    SEED         = 42

    P_ET     = 0.50
    P_TUMOR  = 0.30
    P_RANDOM = 0.20

    BASE_FILTERS        = 16
    TRANSFORMER_DEPTH   = 1
    NUM_HEADS           = 4
    DROPOUT             = 0.10
    USE_DEEP_SUPERVISION = False

    EPOCHS              = 150
    LR                  = 1e-4
    WEIGHT_DECAY        = 1e-5
    AMP                 = True
    GRAD_CLIP           = 1.0
    EARLY_STOP_PATIENCE = 20
    CHECKPOINT          = "best_model.pth"

    W_DICE      = 1.0
    W_BCE       = 0.7
    W_FTV       = 0.7
    W_BOUNDARY  = 0.3
    W_DEEP      = 0.25

    BCE_POS_WEIGHT   = [3.0, 5.0, 10.0]
    FTV_CLASS_WEIGHT = [1.0, 1.25, 2.0]

    SW_OVERLAP     = 0.5
    THRESHOLD      = 0.5
    MIN_CC_SIZE_WT = 200
    MIN_CC_SIZE_TC = 100
    MIN_CC_SIZE_ET = 30
    MAX_PATIENTS   = 0
