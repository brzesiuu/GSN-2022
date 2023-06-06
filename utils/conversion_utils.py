import numpy as np
import cv2 as cv
import torch


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


def get_heatmaps(points_2d: np.ndarray, image_size: tuple, gaussian_kernel: int = 3) \
        -> np.ndarray:
    """
    Generates heatmaps based on given 3D points and camera matrix corresponding to them
    :param points_2d: Array of 3D points in shape of (N,3)
    :type points_2d: np.ndarray
    :param image_size: 2D/3D size of image
    :type image_size: tuple/list
    :param gaussian_kernel: Size of Gaussian kernel for heatmap generation
    :type gaussian_kernel: int
    :return: List of heatmaps
    :rtype: np.ndarray
    """
    if len(image_size) > 2:
        image_size = image_size[-2:]
    heatmaps = []
    for point_2d in points_2d:
        img = np.zeros(image_size[:2], dtype=np.float32)
        if point_2d[0] > image_size[0] - 1:
            point_2d[0] = image_size[0] - 1
        if point_2d[1] > image_size[1] - 1:
            point_2d[1] = image_size[1] - 1
        img[round(point_2d[1]), round(point_2d[0])] = 1
        img = cv.GaussianBlur(img, (gaussian_kernel, gaussian_kernel), 0)
        img /= img.max()
        heatmaps.append(img)
    return np.array(heatmaps)


def get_keypoints_from_heatmaps(heatmaps):
    indices = []
    if isinstance(heatmaps, torch.Tensor):
        heatmaps_tmp = heatmaps.detach().cpu().numpy()
    else:
        heatmaps_tmp = heatmaps.copy()

    if len(heatmaps.shape) == 3:
        return _get_keypoints_from_heatmaps_single_batch(heatmaps_tmp)

    for heatmap in heatmaps_tmp:
        indices.append(_get_keypoints_from_heatmaps_single_batch(heatmap))
    return np.array(indices)


def _get_keypoints_from_heatmaps_single_batch(heatmaps):
    indices = []
    for heatmap in heatmaps:
        heatmap_norm = heatmap / heatmap.sum()
        y = np.dot(heatmap_norm.sum(axis=0), np.linspace(0, heatmap.shape[0], heatmap.shape[0]))
        x = np.dot(heatmap_norm.sum(axis=1), np.linspace(0, heatmap.shape[1], heatmap.shape[1]))
        index = [x, y]
        indices.append(index)
    return np.array(indices)
