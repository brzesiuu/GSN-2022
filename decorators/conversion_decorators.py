import functools

from utils import conversion_utils


def keypoints_2d(function):
    """A general decorator function"""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        result = function(*args, **kwargs)
        if 'keypoints_2d' in result:
            return result
        if 'keypoints_3d_local' in result and 'camera_matrix' in result:
            result['keypoints_2d'] = conversion_utils.project_local_to_uv(result['keypoints_3d_local'],
                                                                          result['camera_matrix'])
        elif 'heatmaps' in result:
            pass
        else:
            raise ValueError('Could not find necessary data to calculate 2D keypoints!')
        return result

    return wrapper


def heatmaps(func=None, *, gaussian_kernel=None):
    if func is None:
        return functools.partial(heatmaps, gaussian_kernel=gaussian_kernel)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        result = func(*args, **kwargs)
        if 'heatmaps' in result:
            return result
        if 'keypoints_2d' in result:
            uv = result['keypoints_2d']
        elif 'keypoints_3d_local' in result and 'camera_matrix' in result:
            uv = conversion_utils.project_local_to_uv(result['keypoints_3d_local'],
                                                      result['camera_matrix'])
        else:
            raise ValueError('Could not find necessary data to calculate 2D keypoints!')
        result['heatmaps'] = conversion_utils.get_heatmaps(uv, result['image'].shape, gaussian_kernel)
        return result

    return wrapper
