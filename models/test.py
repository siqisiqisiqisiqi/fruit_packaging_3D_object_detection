import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import numpy as np
from src.params import *

# g_type2class = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
# g_class2type = {g_type2class[t]: t for t in g_type2class}
# g_type2onehotclass = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}

# g_type_mean_size = {'Car': np.array([3.88311640418, 1.62856739989, 1.52563191462]),
#                     'Pedestrian': np.array([0.84422524, 0.66068622, 1.76255119]),
#                     'Cyclist': np.array([1.76282397, 0.59706367, 1.73698127])}

# g_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3))  # size clustrs
# for i in range(NUM_SIZE_CLUSTER):
#     g_mean_size_arr[i, :] = g_type_mean_size[g_class2type[i]]

print(g_type2class)
