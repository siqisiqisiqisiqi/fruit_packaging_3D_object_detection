import sys
import os
from glob import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
PARENT_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(ROOT_DIR)

import torch
import json
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from numpy.linalg import norm

from src.params import *


pc_path = os.path.join(PARENT_DIR, "datasets", "pointclouds")
label_path = os.path.join(PARENT_DIR, "datasets", "labels")


class StereoCustomDataset(Dataset):
    def __init__(self, pc_path: str, label_path: str, downsample=True):
        super().__init__()

        self.pc_path = pc_path
        self.label_path = label_path
        self.DS = downsample

        self.pc_list = glob(f"{pc_path}/*.ply")

    def downsample(self, pc_in_numpy, num_object_points):
        pc_num = len(pc_in_numpy)
        idx = np.random.randint(pc_num, size=num_object_points)
        downsample_pc = pc_in_numpy[idx, :]
        return downsample_pc

    def __len__(self):
        return len(self.pc_list)

    def __getitem__(self, index):
        pcd = o3d.io.read_point_cloud(self.pc_list[index])
        pc_in_numpy = np.asarray(pcd.points)
        centroid_point = np.sum(pc_in_numpy, 0) / len(pc_in_numpy)
        pc_name = self.pc_list[index].split("/")[-1].split("_")
        label_dir = f"{label_path}/{pc_name[0]}.json"

        with open(label_dir) as f:
            d = json.load(f)

        object_num = len(d['objects'])
        distance = []
        for i in range(object_num):
            center = d['objects'][i]['centroid']
            label_center = np.array([center['x'], center['y'], center['z']])
            distance.append(norm(label_center - centroid_point, 2))
        idx = np.argmin(distance)
        label = d['objects'][idx]
        if self.DS:
            pc_in_numpy = self.downsample(pc_in_numpy, NUM_OBJECT_POINT)
        return pc_in_numpy, label


if __name__ == "__main__":
    dataset = StereoCustomDataset(pc_path, label_path)
    train_features, train_labels = next(iter(dataset))
    # visualize the downsampled point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(train_features)
    o3d.visualization.draw_geometries([pcd])
    # split the dataset into train and test dataset
    train_size = int(0.8*len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
