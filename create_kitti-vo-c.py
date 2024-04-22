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
root = '/ws/data/kitti-vo'
croot = '/ws/data/kitti-voc'
grdimage_dir = 'raw_data'
left_color_camera_dir = 'image_02/data'
vel_dir = 'velodyne_points/data'
split = 'test'
# splits = '/ws/external/kitti-splits/eigen_zhou/val_files.txt'

file_name = []
txt_file_name = os.path.join(root, grdimage_dir, 'kitti_split', split + '_files.txt')
with open(txt_file_name, "r") as txt_f:
    lines = txt_f.readlines()
    for line in lines:
        line = line.strip()
        # check grb file exist
        grb_file_name = os.path.join(root, grdimage_dir, line[:38], left_color_camera_dir,
                                          line[38:].lower())
        if not os.path.exists(grb_file_name):
            # ignore frames with out velodyne
            print(grb_file_name + ' do not exist!!!')
            continue

        velodyne_file_name = os.path.join(root, grdimage_dir, line[:38], vel_dir,
                                          line[38:].lower().replace('.png', '.bin'))
        if not os.path.exists(velodyne_file_name):
            # ignore frames with out velodyne
            print(velodyne_file_name + ' do not exist!!!')
            continue

        file_name.append(line)

side_map = {'l': 2, 'r': 3, '2': 2, '3': 3}

for file in tqdm(file_name):
    folder, subfolder, img_name = file.split("/")
    image_path = os.path.join(root, grdimage_dir, folder, subfolder, 'image_02/data', img_name)
    with open(image_path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB')
    img = np.asarray(img)
    # processing corruptions
    for corruption in get_corruption_names('common'):
        tic = time.time()
        for severity in range(5):
            corrupted = corrupt(img, corruption_name=corruption, severity=severity+1)
            cfolder_path = os.path.join(croot, corruption, str(severity+1), folder, subfolder, 'image_02/data')
            os.makedirs(cfolder_path, exist_ok=True)
            cimage_path = os.path.join(cfolder_path, img_name)
            corrupted = Image.fromarray(corrupted, 'RGB')
            corrupted.save(cimage_path)



