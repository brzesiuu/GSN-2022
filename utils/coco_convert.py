from pathlib import Path
import shutil

import numpy as np

from utils import file_utils

def main():
    data = file_utils.load_config('C:\\datasets\\coco\\2017\\annotations\\person_keypoints_train2017.json')
    image_path = 'C:\\datasets\\coco\\2017\\train2017'
    result_path = 'C:\\datasets\\coco\\2017_converted'

    annotations = data["annotations"]

    for annotation in annotations:
        kp = annotation["keypoints"]
        if np.count_nonzero(kp) < 51:
            continue
        new_id = str("{:012d}".format(int(annotation["image_id"])))
        old_image_path = str(Path(image_path).joinpath(new_id + ".jpg"))
        new_image_path = str(Path(result_path).joinpath("images", new_id + ".jpg"))
        new_annotation_path = str(Path(result_path).joinpath("annotations", new_id + ".json"))
        annotation["image_path"] = new_image_path
        file_utils.save_config(annotation, new_annotation_path)
        shutil.copy(old_image_path, new_image_path)
        print(5)



if __name__ == '__main__':
    main()