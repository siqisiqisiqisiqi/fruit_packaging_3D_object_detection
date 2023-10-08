import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
PARENT_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(ROOT_DIR)

import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.amodal_3D_model import Amodal3DModel
from utils.stereo_custom_dataset import StereoCustomDataset
from src.params import *

pc_path = os.path.join(PARENT_DIR, "datasets", "pointclouds")
label_path = os.path.join(PARENT_DIR, "datasets", "labels")
save_path = os.path.join(ROOT_DIR, "models")

dataset = StereoCustomDataset(pc_path, label_path)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size])

train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

# test the dataloader
# train_features, train_labels = next(iter(train_dataloader))

# select the device
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# print(device)


def test(model, loader):
    test_losses = {
        'total_loss': 0.0,
        'center_loss': 0.0,
        'heading_class_loss': 0.0,
        'size_class_loss': 0.0,
        'heading_residual_normalized_loss': 0.0,
        'size_residual_normalized_loss': 0.0,
        'stage1_center_loss': 0.0,
        'corners_loss': 0.0
    }

    test_metrics = {
        'iou2d': 0.0,
        'iou3d': 0.0,
        'iou3d_0.7': 0.0,
    }

    n_batches = 0
    for i, data_dicts in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
        n_batches += 1

    data_dicts_var = {key: value.cuda() for key, value in data_dicts.items()}

    model = model.eval()

    with torch.no_grad():
        losses, metrics = model(data_dicts_var)

    for key in test_losses.keys():
        if key in losses.keys():
            test_losses[key] += losses[key].detach().item()
    for key in test_metrics.keys():
        if key in metrics.keys():
            test_metrics[key] += metrics[key]

    for key in test_losses.keys():
        test_losses[key] /= n_batches
    for key in test_metrics.keys():
        test_metrics[key] /= n_batches

    return test_losses, test_metrics


def train():
    SEED = 1
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    model = Amodal3DModel()
    model.to(device)

    # define the optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=BASE_LR,
        betas=(0.9, 0.999), eps=1e-08,
        weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=LR_STEPS[0], gamma=GAMMA)

    for epoch in range(MAX_EPOCH):
        train_losses = {
            'total_loss': 0.0,
            'center_loss': 0.0,
            'heading_class_loss': 0.0,
            'size_class_loss': 0.0,
            'heading_residual_normalized_loss': 0.0,
            'size_residual_normalized_loss': 0.0,
            'stage1_center_loss': 0.0,
            'corners_loss': 0.0
        }
        train_metrics = {
            'iou2d': 0.0,
            'iou3d': 0.0,
            'iou3d_0.7': 0.0,
        }
        n_batches = 0
        for i, data_dicts in tqdm(enumerate(train_dataloader), total=len(train_dataloader), smoothing=0.9):
            n_batches += 1

            data_dicts_var = {key: value.cuda()
                              for key, value in data_dicts.items()}
            optimizer.zero_grad()
            model = model.train()
            losses, metrics = model(data_dicts_var)
            total_loss = losses['total_loss']
            total_loss.backward()
            optimizer.step()

            for key in train_losses.keys():
                if key in losses.keys():
                    train_losses[key] += losses[key].detach().item()
            for key in train_metrics.keys():
                if key in metrics.keys():
                    train_metrics[key] += metrics[key]

        for key in train_losses.keys():
            train_losses[key] /= n_batches
        for key in train_metrics.keys():
            train_metrics[key] /= n_batches

        test_losses, test_metrics = test(model,test_dataloader)
        scheduler.step()

        if scheduler.get_lr()[0] < MIN_LR:
            for param_group in optimizer.param_groups:
                param_group['lr'] = MIN_LR

        if test_metrics['iou3d_0.7'] >= best_iou3d_70:
            best_iou3d_70 = test_metrics['iou3d_0.7']
            best_epoch = epoch + 1
            if epoch > MAX_EPOCH / 5:
                savepath = f"{save_path}/epoch{epoch}.pth"
                state = {
                    'epoch': epoch + 1,
                    'train_iou3d_0.7': train_metrics['iou3d_0.7'],
                    'test_iou3d_0.7': test_metrics['iou3d_0.7'],
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)

if __name__ == "__main__":
    train()