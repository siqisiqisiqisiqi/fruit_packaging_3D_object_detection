import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import ipdb
import torch
import numpy as np
import torchvision.transforms as transforms
from src.params import *


class StereoCustomDataset(object):
    def __init__(self, npoints):
        self.npoints = npoints