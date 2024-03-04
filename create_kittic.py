from imagecorruptions import corrupt, get_corruption_names
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
# image = np.asarray(Image.open('test_image.jpg'))
#image = np.ones((427, 640, 3), dtype=np.uint8)

# corrupted_image = corrupt(img, corruption_name='gaussian_blur', severity=1)

# for corruption in get_corruption_names('blur'):
#     tic = time.time()
#     for severity in range(5):
#         corrupted = corrupt(image, corruption_name=corruption, severity=severity+1)
#         plt.imshow(corrupted)
#         plt.show()
#     print(corruption, time.time() - tic)
import os
from tqdm import tqdm
# via number
path = '/ws/data/kitti-vo'
cpath = '/ws/data/kitti-c'
splits = '/ws/external/kitti-splits/eigen_zhou/val_files.txt'
side_map = {'l': 2, 'r': 3, '2': 2, '3': 3}

with open(splits, 'r') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        folder, frame_index, side = line.split()
        f_str = "{:010d}{}".format(int(frame_index), '.jpg')
        image_path = os.path.join(path, folder, "image_0{}/data".format(side_map[side]), f_str)
        with open(image_path, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')
        img = np.asarray(img)
        # processing corruptions
        for corruption in get_corruption_names('noise'):
            tic = time.time()
            for severity in range(5):
                corrupted = corrupt(img, corruption_name=corruption, severity=severity + 1)
                cfolder_path = os.path.join(cpath, corruption, str(severity), folder, "image_0{}/data".format(side_map[side]))
                os.makedirs(cfolder_path, exist_ok=True)
                cimage_path = os.path.join(cfolder_path, f_str)
                corrupted = Image.fromarray(corrupted, 'RGB')
                corrupted.save(cimage_path)




