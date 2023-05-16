import cv2 as cv
import numpy as np
import colorsys


def visualize_keypoints(image, keypoints, keypoints_map, thickness=2):
    num_colors = sum([len(joint_group) for joint_group in keypoints_map])
    image_copy = image.copy()

    hue_values = np.linspace(0, 1, num_colors)
    hue_idx = 0
    for joint_group in keypoints_map:
        for idx in range(len(joint_group) - 1):
            red, green, blue = colorsys.hsv_to_rgb(hue_values[hue_idx], 1.0, 1.0)
            bgr = (int(blue * 255), int(green * 255), int(red * 255))
            x1, y1 = keypoints[joint_group[idx]]
            x2, y2 = keypoints[joint_group[idx + 1]]
            image_copy = cv.line(image_copy, (int(x1), int(y1)), (int(x2), int(y2)), bgr, thickness=thickness)
            hue_idx += 1
    return image_copy
