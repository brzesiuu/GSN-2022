import functools

from utils import conversion_utils


def keypoints_2d(function):
    """A general decorator function"""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        result = function(*args, **kwargs)
        if 'keypoints_2d' in result and result['keypoints_2d'] is not None:
            return result
        if 'keypoints_3d_local' in result and 'camera_matrix' in result:
            keypoints_2d = conversion_utils.project_local_to_uv(result['keypoints_3d_local'],
                                                                result['camera_matrix'])
            if "scale" in result:
                keypoints_2d *= result['scale']
            result["keypoints_2d"] = keypoints_2d
            return result
        if 'heatmaps' in result:
            kp_2d = conversion_utils.get_keypoints_from_heatmaps(result['heatmaps'])
            if "heatmaps_scale" in result:
                kp_2d /= result["heatmaps_scale"]
            result['keypoints_2d'] = kp_2d
            return result
        else:
            raise ValueError('Could not find necessary data to calculate 2D keypoints!')

    return wrapper


def heatmaps(func=None):
    if func is None:
        return functools.partial(heatmaps)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        result = func(*args, **kwargs)
        if 'heatmaps' in result and result['heatmaps'] is not None:
            return result
        if 'keypoints_2d' in result:
            uv = result['keypoints_2d'].copy()
        elif 'keypoints_3d_local' in result and 'camera_matrix' in result:
            uv = conversion_utils.project_local_to_uv(result['keypoints_3d_local'],
                                                      result['camera_matrix']).copy()
        else:
            raise ValueError('Could not find necessary data to calculate 2D keypoints!')
        if "scale" in result:
            uv *= result['scale']
        heatmaps_shape = list(result['image'].shape)
        if "heatmaps_scale" in result:
            uv *= result['heatmaps_scale']
            heatmaps_scale = result['heatmaps_scale']

            heatmaps_shape[-2] = round(heatmaps_shape[-2] * heatmaps_scale)
            heatmaps_shape[-1] = round(heatmaps_shape[-1] * heatmaps_scale)
        result['heatmaps'] = conversion_utils.get_heatmaps(uv, heatmaps_shape, result['gaussian_kernel'])
        return result

    return wrapper
