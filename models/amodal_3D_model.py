import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init

from utils.model_util import PointNetLoss, parse_output_to_tensors
from utils.point_cloud_process import point_cloud_process
from utils.compute_box3d_iou import compute_box3d_iou
from src.params import *


class PointNetEstimation(nn.Module):
    def __init__(self, n_classes=1):
        '''v1 Amodal 3D Box Estimation Pointnet
        :param n_classes:3
        :param one_hot_vec:[bs,n_classes]
        '''
        super(PointNetEstimation, self).__init__()
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        self.n_classes = n_classes

        self.fc1 = nn.Linear(512 + 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3 + NUM_HEADING_BIN *
                             2 + NUM_SIZE_CLUSTER * 4)
        self.fcbn1 = nn.BatchNorm1d(512)
        self.fcbn2 = nn.BatchNorm1d(256)

    def forward(self, pts, one_hot_vec):  # bs,3,m
        '''
        :param pts: [bs,3,m]: x,y,z after InstanceSeg
        :return: box_pred: [bs,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4]
            including box centers, heading bin class scores and residual,
            and size cluster scores and residual
        '''
        bs = pts.size()[0]
        n_pts = pts.size()[2]

        out1 = F.relu(self.bn1(self.conv1(pts)))  # bs,128,n
        out2 = F.relu(self.bn2(self.conv2(out1)))  # bs,128,n
        out3 = F.relu(self.bn3(self.conv3(out2)))  # bs,256,n
        out4 = F.relu(self.bn4(self.conv4(out3)))  # bs,512,n
        global_feat = torch.max(out4, 2, keepdim=False)[0]  # bs,512

        expand_one_hot_vec = one_hot_vec.view(bs, -1)  # bs,3
        expand_global_feat = torch.cat(
            [global_feat, expand_one_hot_vec], 1)  # bs,515

        x = F.relu(self.fcbn1(self.fc1(expand_global_feat)))  # bs,512
        x = F.relu(self.fcbn2(self.fc2(x)))  # bs,256
        box_pred = self.fc3(x)  # bs,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4
        return box_pred


class STNxyz(nn.Module):
    def __init__(self, n_classes=1):
        super(STNxyz, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        # self.conv4 = torch.nn.Conv1d(256, 512, 1)
        self.fc1 = nn.Linear(256 + n_classes, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)

        init.zeros_(self.fc3.weight)
        init.zeros_(self.fc3.bias)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.fcbn1 = nn.BatchNorm1d(256)
        self.fcbn2 = nn.BatchNorm1d(128)

    def forward(self, pts, one_hot_vec):
        bs = pts.shape[0]
        x = F.relu(self.bn1(self.conv1(pts)))  # bs,128,n
        x = F.relu(self.bn2(self.conv2(x)))  # bs,128,n
        x = F.relu(self.bn3(self.conv3(x)))  # bs,256,n
        x = torch.max(x, 2)[0]  # bs,256
        expand_one_hot_vec = one_hot_vec.view(bs, -1)  # bs,3
        x = torch.cat([x, expand_one_hot_vec], 1)  # bs,259
        x = F.relu(self.fcbn1(self.fc1(x)))  # bs,256
        x = F.relu(self.fcbn2(self.fc2(x)))  # bs,128
        x = self.fc3(x)  # bs,
        return x


class Amodal3DModel(nn.Module):
    def __init__(self, n_classes=1, n_channel=3):
        super(Amodal3DModel, self).__init__()
        self.n_classes = n_classes
        self.n_channel = n_channel
        self.STN = STNxyz(n_classes=1)
        self.est = PointNetEstimation(n_classes=1)
        self.Loss = PointNetLoss()


    def forward(self, features, label_dicts):

        # point cloud after the instance segmentation
        point_cloud = features.permute(0,2,1)
        point_cloud = point_cloud[:, :self.n_channel, :]
        one_hot = label_dicts.get('one_hot') #torch.Size([32, 3])
        bs = point_cloud.shape[0]  # batch size

        # If not None, use to Compute Loss
        box3d_center_label = label_dicts.get('box3d_center')  # torch.Size([32, 3])
        size_class_label = label_dicts.get('size_class')  # torch.Size([32, 1])
        size_residual_label = label_dicts.get(
            'size_residual')  # torch.Size([32, 3])
        heading_class_label = label_dicts.get('angle_class')  # torch.Size([32, 1])
        heading_residual_label = label_dicts.get(
            'angle_residual')  # torch.Size([32, 1])

        # object_pts_xyz size (batchsize, number object point, 3)
        object_pts_xyz, mask_xyz_mean = point_cloud_process(point_cloud)

        # T-net
        object_pts_xyz = object_pts_xyz.device()
        center_delta = self.STN(object_pts_xyz, one_hot)  # (32,3)
        stage1_center = center_delta + mask_xyz_mean  # (32,3)

        if (np.isnan(stage1_center.cpu().detach().numpy()).any()):
            ipdb.set_trace()
        object_pts_xyz_new = object_pts_xyz - \
            center_delta.view(
                center_delta.shape[0], -1, 1).repeat(1, 1, object_pts_xyz.shape[-1])

        # 3D Box Estimation
        box_pred = self.est(object_pts_xyz_new, one_hot)
        center_boxnet, \
            heading_scores, heading_residual_normalized, heading_residual, \
            size_scores, size_residual_normalized, size_residual = \
            parse_output_to_tensors(box_pred)

        box3d_center = center_boxnet + stage1_center  # bs,3

        losses = self.Loss(box3d_center, box3d_center_label, stage1_center,
                        heading_scores, heading_residual_normalized,
                        heading_residual,
                        heading_class_label, heading_residual_label,
                        size_scores, size_residual_normalized,
                        size_residual,
                        size_class_label, size_residual_label)

        for key in losses.keys():
            losses[key] = losses[key] / bs

            with torch.no_grad():
                iou2ds, iou3ds = compute_box3d_iou(
                    box3d_center.detach().cpu().numpy(),
                    heading_scores.detach().cpu().numpy(),
                    heading_residual.detach().cpu().numpy(),
                    size_scores.detach().cpu().numpy(),
                    size_residual.detach().cpu().numpy(),
                    box3d_center_label.detach().cpu().numpy(),
                    heading_class_label.detach().cpu().numpy(),
                    heading_residual_label.detach().cpu().numpy(),
                    size_class_label.detach().cpu().numpy(),
                    size_residual_label.detach().cpu().numpy())
            metrics = {
                'iou2d': iou2ds.mean(),
                'iou3d': iou3ds.mean(),
                'iou3d_0.7': np.sum(iou3ds >= 0.7) / bs
            }
            return losses, metrics
