import os
import time
import datetime
import cv2
from PIL import Image
from numba import jit
import numpy as np


@jit(nopython=True)
def gray_floyd_steinberg(image):
    # image: np.array of shape (height, width), dtype=float, 0.0-1.0
    # works in-place!
    h, w = image.shape
    for y in range(h):
        for x in range(w):
            old = image[y, x]
            new = np.round(old)
            image[y, x] = new
            error = old - new
            # precomputing the constants helps
            if x + 1 < w:
                image[y, x + 1] += error * 0.4375  # right, 7 / 16
            if (y + 1 < h) and (x + 1 < w):
                image[y + 1, x + 1] += error * 0.0625  # right, down, 1 / 16
            if y + 1 < h:
                image[y + 1, x] += error * 0.3125  # down, 5 / 16
            if (x - 1 >= 0) and (y + 1 < h):
                image[y + 1, x - 1] += error * 0.1875  # left, down, 3 / 16
    return image


def pil_to_np(pilimage):
    return np.array(pilimage) / 255


def np_to_pil(image):
    return Image.fromarray((image * 255).astype('uint8'))


if __name__ == '__main__':
    # D:\Dataset\halftoneImg\train\target
    # path1 = '../dataset/test/Kodak/target/'
    # path2 = '../dataset/test/Kodak/data/'
    # E:\graduate\data\Kodak E:\graduate\data\halftone\Kodak
    path1 = 'E://graduate/data/halftone/Kodak/target/'
    path2 = 'E://graduate/data/halftone/Kodak/data/'
    if not os.path.isdir(path2):
        os.makedirs(path2)
    start_time = time.time()
    for filename in os.listdir(path1):
        gray_img = Image.open(path1 + filename)
        print(filename)
        gray_img = np.array(gray_img) / 255
        halftone_img = gray_floyd_steinberg(gray_img)
        halftone_img = Image.fromarray((halftone_img * 255).astype(np.uint8))
        halftone_img.save(path2 + filename)
    print(datetime.timedelta(seconds=int(time.time() - start_time)))
