from PIL import Image
import numpy as np
import os
import cv2
import random


def copied_name(name):
    """
    生成备份图片名
    :param name: 原名
    :return: 备份名
    """
    parts = os.path.splitext(name)
    return parts[0] + '_copy' + parts[1]


def resize(dir):
    """
    变换大小
    :param dir:
    :param target_height_percent:
    :param target_width_percent:
    :param overlap:
    :return:
    """
    print('resizing image: ' + dir)
    im1 = cv2.imread(dir)
    img = cv2.resize(im1, (28, 28), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(dir, img)


def flip_up_down(dir, overlap, value1, value2):
    """
    上下翻转
    :param dir:
    :param overlap:
    :return:
    """
    img = cv2.imread(dir, 0)
    new_img = cv2.flip(img, 0)
    if not overlap:
        cv2.imwrite(dir, new_img)
    else:
        cv2.imwrite(copied_name(dir), new_img)


def flip_left_right(dir, overlap, value1, value2):
    """
    左右翻转
    :param dir:
    :param overlap:
    :return:
    """
    img = cv2.imread(dir, 0)
    new_img = cv2.flip(img, 1)
    if not overlap:
        cv2.imwrite(dir, new_img)
    else:
        cv2.imwrite(copied_name(dir), new_img)


def transpose_image(dir, overlap, value1, value2):
    """
    图片转置，对角线翻转
    :param dir:
    :param overlap:
    :return:
    """
    img = cv2.imread(dir, 0)
    new_img = cv2.flip(img, -1)
    if not overlap:
        cv2.imwrite(dir, new_img)
    else:
        cv2.imwrite(copied_name(dir), new_img)


def adjust_brightness_contrast(dir, overlap, alpha, beta):
    """
    调整亮度
    :param dir:
    :param overlap:
    :return:
    """
    img = cv2.imread(dir, 0)
    w = img.shape[1]
    h = img.shape[0]

    for xi in range(0, w):
        for xj in range(0, h):
            img[xj, xi] = img[xj, xi] * alpha + beta
            if img[xj, xi] > 255:
                img[xj, xi] = 255
            elif img[xj, xi] < 0:
                img[xj, xi] = 0
    if not overlap:
        cv2.imwrite(dir, img)
    else:
        cv2.imwrite(copied_name(dir), img)


def random_brightness_contrast(dir, overlap, max_alpha, max_beta):
    img = cv2.imread(dir, 0)
    w = img.shape[1]
    h = img.shape[0]

    alpha = random.uniform(0, max_alpha)
    beta = random.randint(0 - max_beta, max_beta)

    for xi in range(0, w):
        for xj in range(0, h):
            img[xj, xi] = img[xj, xi] * alpha + beta
            if img[xj, xi] > 255:
                img[xj, xi] = 255
            elif img[xj, xi] < 0:
                img[xj, xi] = 0
    if not overlap:
        cv2.imwrite(dir, img)
    else:
        cv2.imwrite(copied_name(dir), img)


def mean_filter(dir, overlap, height, value2):
    img = cv2.imread(dir, 0)
    new_img = cv2.blur(img, (height, height))
    if not overlap:
        cv2.imwrite(dir, new_img)
    else:
        cv2.imwrite(copied_name(dir), new_img)


def gaussian_blur(dir, overlap, height, value2):
    img = cv2.imread(dir, 0)
    new_img = cv2.GaussianBlur(img, (height, height), 0)
    if not overlap:
        cv2.imwrite(dir, new_img)
    else:
        cv2.imwrite(copied_name(dir), new_img)


def median_filter(dir, overlap, height, value2):
    print(overlap)
    img = cv2.imread(dir, 0)
    new_img = cv2.medianBlur(img, height, 0)
    if not overlap:
        cv2.imwrite(dir, new_img)
    else:
        cv2.imwrite(copied_name(dir), new_img)


def nl_denoise_gray(dir, overlap, h, value2):
    img = cv2.imread(dir, 0)
    new_img = cv2.fastNlMeansDenoising(img, None, h, 7, 21)
    if not overlap:
        cv2.imwrite(dir, new_img)
    else:
        cv2.imwrite(copied_name(dir), new_img)


def add_salt_pepper_noise(dir, overlap, percent, value2):
    img = cv2.imread(dir, 0)
    m = int(28 * 28 * percent)
    for a in range(m):
        i = int(np.random.random() * 28)
        j = int(np.random.random() * 28)

        if img.ndim == 2:
            img[j, i] = 255
        elif img.ndim == 3:
            img[j, i, 0] = 255
            img[j, i, 1] = 255
            img[j, i, 2] = 255

    for a in range(m):
        i = int(np.random.random() * 28)
        j = int(np.random.random() * 28)

        if img.ndim == 2:
            img[j, i] = 0
        elif img.ndim == 3:
            img[j, i, 0] = 0
            img[j, i, 1] = 0
            img[j, i, 2] = 0

    if not overlap:
        cv2.imwrite(dir, img)
    else:
        cv2.imwrite(copied_name(dir), img)


def equalize_hist(dir, overlap, value1, value2):
    img = cv2.imread(dir, 0)
    new_img = cv2.equalizeHist(img)
    if not overlap:
        cv2.imwrite(dir, new_img)
    else:
        cv2.imwrite(copied_name(dir), new_img)


def clahe(dir, overlap, value1, value2):
    img = cv2.imread(dir, 0)
    clahe = cv2.createCLAHE()
    new_img = clahe.apply(img)
    if not overlap:
        cv2.imwrite(dir, new_img)
    else:
        cv2.imwrite(copied_name(dir), new_img)


def erode(dir, overlap, kernel_value, value2):
    img = cv2.imread(dir, 0)
    kernel = np.ones((kernel_value, kernel_value), np.uint8)
    new_img = cv2.erode(img, kernel)
    if not overlap:
        cv2.imwrite(dir, new_img)
    else:
        cv2.imwrite(copied_name(dir), new_img)


def dilate(dir, overlap, kernel_value, value2):
    img = cv2.imread(dir, 0)
    kernel = np.ones((kernel_value, kernel_value), np.uint8)
    new_img = cv2.dilate(img, kernel)
    if not overlap:
        cv2.imwrite(dir, new_img)
    else:
        cv2.imwrite(copied_name(dir), new_img)


dir = '/Users/keenan/Documents/workspace/Cloud-Server/test-data/test1.jpg'
img = Image.open(dir)
print(img.format, img.size, img.mode)

adjust_brightness_contrast(dir, True, 1.5, 30)

dir2 = '/Users/keenan/Documents/workspace/Cloud-Server/test-data/test1_copy.jpg'
img2 = Image.open(dir2)
print(img2.format, img2.size, img2.mode)
