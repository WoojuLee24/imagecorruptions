from imagecorruptions import corrupt, get_corruption_names
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import csv
import skimage.transform

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
path = '/ws/data2/DSEC/train'
cpath = '/ws/data2/DSEC/test-c'
splits = '/ws/external/kitti-splits/eigen_zhou/val_files.txt'
side_map = {'l': 2, 'r': 3, '2': 2, '3': 3}

# with open(splits, 'r') as f:
#     lines = f.readlines()
#     for line in tqdm(lines):
#         folder, frame_index, side = line.split()
#         f_str = "{:010d}{}".format(int(frame_index), '.jpg')
#         image_path = os.path.join(path, folder, "image_0{}/data".format(side_map[side]), f_str)
#         with open(image_path, 'rb') as f:
#             with Image.open(f) as img:
#                 img = img.convert('RGB')
#         img = np.asarray(img)

def _parse(value, function, fmt):
    """
    Parse a string into a value, and format a nice ValueError if it fails.
    Returns `function(value)`.
    Any `ValueError` raised is catched and a new `ValueError` is raised
    with message `fmt.format(e)`, where `e` is the caught `ValueError`.
    """
    try:
        return function(value)
    except ValueError as e:
        raise_from(ValueError(fmt.format(e)), None)

def _read_annotations(csv_reader, classes):
    result = {}
    for line, row in enumerate(csv_reader):
        line += 1

        try:
            img_file, x1, y1, x2, y2, class_name = row[:6]
        except ValueError:
            raise_from(ValueError(
                'line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)),
                None)

        if img_file not in result:
            result[img_file] = []

        # If a row contains only an image path, it's an image without annotations.
        if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
            continue

        x1 = _parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
        y1 = _parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
        x2 = _parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
        y2 = _parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

        # Check that the bounding box is valid.
        if x2 <= x1:
            raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
        if y2 <= y1:
            raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

        # check if the current class name is correctly present
        if class_name not in classes:
            raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

        result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
    return result


classes = {'person': 0, 'large_vehicle': 1, 'car': 2}
csv_path = "/ws/external/assets/labels_filtered_test.csv"
with open(csv_path, 'r', newline='') as file:
    image_data = _read_annotations(csv.reader(file, delimiter=','), classes)

# folders = sorted(os.listdir(path))

for k, v in tqdm(sorted(image_data.items())):
    file = k.split('/')
    image_name = file[-1].replace('.npz','.png')
    image_path = os.path.join(path, file[-3], 'images/left/rectified', image_name)
    # img_rgb = cv2.imread(img_file)
    # img_rgb = img_rgb.astype(np.float32) / 255.0
    with open(image_path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB')
            img = img.resize((640, 480))
    img = np.asarray(img)
    # img_resized = skimage.transform.resize(img, (480, 640))
    # processing corruptions
    for corruption in get_corruption_names('common'):
        tic = time.time()
        for severity in range(5):
            # severity = 2
            corrupted = corrupt(img, corruption_name=corruption, severity=severity + 1)
            cfolder_path = os.path.join(cpath, corruption, str(severity + 1), f'{file[-3]}/images/left/rectified/')
            os.makedirs(cfolder_path, exist_ok=True)
            cimage_path = os.path.join(cfolder_path, image_name)
            corrupted = Image.fromarray(corrupted, 'RGB')
            corrupted.save(cimage_path)
#
# for folder in folders:
#     images_path = os.path.join(path, f'{folder}/images/left/rectified')
#     images_list = sorted(os.listdir(images_path))
#     for image_name in images_list:
#         image_path = os.path.join(path, f'{folder}/images/left/rectified/{image_name}')
#         with open(image_path, 'rb') as f:
#             with Image.open(f) as img:
#                 img = img.convert('RGB')
#         img = np.asarray(img)
#         # processing corruptions
#         for corruption in get_corruption_names('noise'):
#             tic = time.time()
#             # for severity in range(5):
#             severity = 2
#             corrupted = corrupt(img, corruption_name=corruption, severity=severity + 1)
#             cfolder_path = os.path.join(cpath, corruption, str(severity+1), f'{folder}/images/left/rectified/')
#             os.makedirs(cfolder_path, exist_ok=True)
#             cimage_path = os.path.join(cfolder_path, image_name)
#             corrupted = Image.fromarray(corrupted, 'RGB')
#             corrupted.save(cimage_path)
#
