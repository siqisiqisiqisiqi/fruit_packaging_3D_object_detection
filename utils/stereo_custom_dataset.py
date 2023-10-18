import sys
import os
import re
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
from torch.utils.data import DataLoader

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

    def convertlabelformat(self, label, label_dir):
        center = label['centroid']
        box3d_center = np.array([center['x'], center['y'], center['z']]) * 100
        # box3d_center = np.array([center['x'], center['y'], center['z']])
        size_class = np.array([g_type2onehotclass[label['name']]])
        standard_size = g_type_mean_size[label['name']]
        size = label['dimensions']
        box_size = np.array(
            [size['length'], size['width'], size['height']]) * 100
        # box_size = np.array([size['length'], size['width'], size['height']])
        size_residual = standard_size - box_size
        angle = label['rotations']['z']
        angle_per_class = 2 * np.pi / float(NUM_HEADING_BIN)
        angle_class = np.array([angle // angle_per_class])
        angle_residual = np.array([angle % angle_per_class])
        one_hot = np.array([1])
        x = label_dir.split("/")
        x[-2] = "images"
        a = re.findall(r'\d+', x[-1])
        num = a[0]
        x[-1] = f"Image_{num}.jpg"
        img_dir = "/".join(x)
        label2 = {'one_hot': one_hot, 'box3d_center': box3d_center, 'size_class': size_class, 'size_residual': size_residual,
                  'angle_class': angle_class, 'angle_residual': angle_residual}
        return label2, img_dir

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
        pc_in_numpy = pc_in_numpy * 100
        # pc_in_numpy = pc_in_numpy
        if self.DS:
            pc_in_numpy = self.downsample(pc_in_numpy, NUM_OBJECT_POINT)
        label2, img_dir = self.convertlabelformat(label, label_dir)
        return pc_in_numpy, label2, img_dir


if __name__ == "__main__":
    dataset = StereoCustomDataset(pc_path, label_path)
    # train_features, train_labels = next(iter(dataset))
    # print(train_labels)

    # # visualize the downsampled point cloud
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(train_features)
    # o3d.visualization.draw_geometries([pcd])

    # split the dataset into train and test dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size])
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    train_features, train_labels, img_dir = next(iter(train_dataloader))
    box3d_center_label = train_labels.get('box3d_center')
    size_class_label = train_labels.get('size_class')
    size_residual_label = train_labels.get('size_residual')
    heading_class_label = train_labels.get(
        'angle_class')  # torch.Size([32, 1])
    one_hot = train_labels.get(
        'one_hot')
    heading_residual_label = train_labels.get('angle_residual')
    features = train_features.permute(0, 2, 1)
    num_point = features.shape[2]
    xyz_sum = features.sum(2, keepdim=True)
    xyz_mean = xyz_sum / num_point
