import sys
import os
import time

from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import numpy as np
from configparser import ConfigParser

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)
# sys.path.append(ROOT_DIR)

# import numpy as np
# from scipy import signal
# from src.params import *
# import torch
# import torch.nn as nn



# g_type2class = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
# g_class2type = {g_type2class[t]: t for t in g_type2class}
# g_type2onehotclass = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}

# g_type_mean_size = {'Car': np.array([3.88311640418, 1.62856739989, 1.52563191462]),
#                     'Pedestrian': np.array([0.84422524, 0.66068622, 1.76255119]),
#                     'Cyclist': np.array([1.76282397, 0.59706367, 1.73698127])}

# g_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3))  # size clustrs
# for i in range(NUM_SIZE_CLUSTER):
#     g_mean_size_arr[i, :] = g_type_mean_size[g_class2type[i]]

# print(g_type2class)

# a = {'name': 'peach', 'centroid': {'x': 0.113432, 'y': 0.115572, 'z': 0.032}, 'dimensions': {
#     'length': 0.07, 'width': 0.07, 'height': 0.064}, 'rotations': {'x': 0.0, 'y': 0.0, 'z': 0.0}}
# center = a['centroid']
# box3d_center = np.array([center['x'], center['y'], center['z']])
# size_class = g_type2onehotclass[a['name']]
# standard_size = g_type_mean_size[a['name']]
# size = a['dimensions']
# box_size = np.array([size['length'], size['width'], size['height']])
# size_residual = standard_size - box_size
# angle = a['rotations']['z']
# angle_per_class = 2 * np.pi / float(NUM_HEADING_BIN)
# angle_class = angle // angle_per_class
# angle_residual = angle % angle_per_class
# onehot = np.array([1])
# b = {'box3d_center': box3d_center, 'size_class': size_class, 'size_residual': size_residual,
#      'angle_class': angle_class, 'angle_residual': angle_residual}
# print(b)

# m = nn.Conv1d(16, 33, 1, stride=1)
# input = torch.randn(20, 16, 50)
# output = m(input)
# print(output.shape)

# strtime = time.strftime('%Y%m%d-%H%M%S',time.localtime(time.time()))
# strtime = strtime[4:8]
# print(strtime)

# plt.plot([1, 2, 3], [1, 4, 9])
# # plt.show()
# plt.savefig('foo.jpg')
# plt.show()

with np.load('camera_params/E.npz') as X:
    mtx, dist, Mat, tvecs = [X[i] for i in ('mtx', 'dist', 'Mat', 'tvecs')]

print(tvecs)
# tvecs[0,0] = -1 * tvecs[0,0]
tvecs[1,0] = -1 * tvecs[1,0]
tvecs[2,0] = -1 * tvecs[2,0]

r = R.from_euler('y', -2.0, degrees=True)
R_euler = r.as_matrix()
Mat2 = Mat@R_euler

r = R.from_euler('x', -3.0, degrees=True)
R_euler = r.as_matrix()
Mat3 = Mat2@R_euler

r = R.from_euler('z', 2.5, degrees=True)
R_euler = r.as_matrix()
Mat4 = Mat3@R_euler

config_object = ConfigParser()
config_object.read(f"camera_params/SN36077403.conf")
intri_param = config_object["LEFT_CAM_HD"]

fx = float(intri_param["fx"])
fy = float(intri_param["fy"])
cx = float(intri_param["cx"])
cy = float(intri_param["cy"])
mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
print(tvecs.shape)

a = np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4]])
b = np.array([[1],[2],[3]])

np.savez('camera_params/Ext2.npz', mtx = mtx, dist = dist, Mat = Mat4, tvecs = tvecs)

# for i in range(3,-1,-1):
#     print(i)