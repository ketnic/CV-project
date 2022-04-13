import numpy as np
import cv2
import re
import urllib.request
from PIL import Image, ImageOps


def read_image(img_path):
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    if re.match(regex, img_path) is not None:
        req = urllib.request.urlopen(img_path)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        return cv2.imdecode(arr, -1)  # 'Load it as it is'
    return cv2.imread(img_path)


def padding(img, expected_size):
    desired_size = expected_size
    delta_width = desired_size - img.size[0]
    delta_height = desired_size - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width -
               pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def resize_with_padding(image, expected_size):
    old_size = image.shape[:2]  # old_size is in (height, width) format

    ratio = float(expected_size) / max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size]) # new_size should be in (width, height) format

    image = cv2.resize(image, (new_size[1], new_size[0]))

    delta_w = expected_size - new_size[1]
    delta_h = expected_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def process_image(image, img_size):
    # image = resize_with_padding(image, img_size[0])
    image = cv2.resize(image, img_size, interpolation=cv2.INTER_AREA)
    return image


def extend_image(image):
    return np.expand_dims(image, axis=0)


def process_and_save_image(img_path, img_size, save_path=None):
    image = read_and_process_image(img_path=img_path, img_size=img_size)
    if save_path is None:
        cv2.imwrite(img_path, image)
    else:
        cv2.imwrite(save_path, image)


def read_and_process_image(img_path, img_size):
    image = read_image(img_path)
    image = process_image(image, img_size)
    return image
