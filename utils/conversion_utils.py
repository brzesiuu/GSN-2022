from typing import List
from pathlib import Path

import numpy as np
import cv2 as cv

from utils import PosePath


def project_local_to_uv(points_3d: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    """
    Projects 3D points in a camera local coordinate system
    :param points_3d: Array of 3D points in shape of (N,3)
    :type points_3d: np.ndarray
    :param camera_matrix: 3x3 camera matrix
    :type camera_matrix: np.ndarray
    :return: Array of calculated projected 2D points
    :rtype: np.ndarray
    """
    points = points_3d / points_3d[:, 2].reshape((points_3d.shape[0], 1))
    uv = np.matmul(camera_matrix, points.T).T
    uv = uv[:, :2]
    return uv


def get_heatmaps(points_3d: np.ndarray, camera_matrix: np.ndarray, image_size: tuple, gaussian_kernel: int = 3) \
        -> List[np.ndarray]:
    """
    Generates heatmaps based on given 3D points and camera matrix corresponding to them
    :param points_3d: Array of 3D points in shape of (N,3)
    :type points_3d: np.ndarray
    :param camera_matrix: 3x3 camera matrix
    :type camera_matrix: np.ndarray
    :param image_size: 2D size of image
    :type image_size: tuple/list
    :param gaussian_kernel: Size of Gaussian kernel for heatmap generation
    :type gaussian_kernel: int
    :return: List of heatmaps
    :rtype: List[np.ndarray]
    """
    uv = project_local_to_uv(points_3d, camera_matrix)
    heatmaps = []
    for point_2d in uv:
        img = np.zeros(image_size[:2], dtype=np.float32)
        img[round(point_2d[1]), round(point_2d[0])] = 1
        img = cv.GaussianBlur(img, (gaussian_kernel, gaussian_kernel), 0)
        img /= img.max()
        heatmaps.append(img)
    return heatmaps
