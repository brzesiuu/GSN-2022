import numpy as np


def PCKLoss(keypoints_calculated, keypoints_gt, relative_distance_threshold=None, threshold_joint_indices=None,
            distance_threshold=None):
    if relative_distance_threshold is None and distance_threshold is None:
        raise ValueError("One of the distance threshold must be specified!")
    if relative_distance_threshold is not None and distance_threshold is not None:
        raise ValueError("Only one distance threshold must be specified!")
    if relative_distance_threshold is not None and threshold_joint_indices is None:
        raise ValueError("If relative_distance_error should be used then threshold joint indices must be specified!")
    if relative_distance_threshold is not None:
        num_total = 0
        for kp_calc, kp_gt in zip(keypoints_calculated, keypoints_gt):
            threshold = relative_distance_threshold * np.linalg.norm(
                kp_gt[threshold_joint_indices[0]] - kp_gt[threshold_joint_indices[0]])
            dists = np.linalg.norm((kp_calc - kp_gt), axis=1)
            num_in_range = np.count_nonzero(dists.flatten() < threshold)
            num_total += num_in_range
        return num_in_range / keypoints_calculated.shape[0] * keypoints_calculated.shape[1] * 100
    else:
        diff = keypoints_gt - keypoints_calculated
        dists = np.linalg.norm(diff, axis=2)
        return np.count_nonzero(dists.flatten() < distance_threshold) / len(dists.flatten()) * 100
