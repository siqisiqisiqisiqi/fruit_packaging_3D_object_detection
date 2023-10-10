import numpy as np


def point_cloud_process(point_cloud):
    """point cloud coordinate transform and center calculation

    Parameters
    ----------
    point_cloud : [batchsize, point number, 3]
        The point cloud should be in the world coordinate system
    """
    num_point = point_cloud.shape[2]
    xyz_sum = point_cloud.sum(2, keepdim=True)
    xyz_mean = xyz_sum / num_point
    point_cloud_trans = point_cloud - xyz_mean.repeat(1, 1, num_point)
    xyz_mean = xyz_mean.squeeze()
    return point_cloud_trans, xyz_mean
