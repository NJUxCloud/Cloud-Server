import tensorflow as tf
from PIL import Image
import numpy as np
import os
import cv2


def save_image(dir, image):
    """
    保存图片
    :param dir: 路径
    :param image: 图片
    :return:
    """
    im = Image.fromarray(image)
    im.save(dir)


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
    raw_image = tf.gfile.FastGFile(name=dir, mode='rb').read()
    with tf.Session() as sess:
        img = tf.image.decode_jpeg(raw_image)
        img_data = tf.image.resize_images(img, [28, 28], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        new_img = np.asarray(img_data.eval(), dtype='uint8')
        save_image(dir=dir, image=new_img)
        sess.close()


def flip_up_down(dir, overlap, value1, value2):
    """
    上下翻转
    :param dir:
    :param overlap:
    :return:
    """
    raw_image = tf.gfile.FastGFile(name=dir, mode='rb').read()
    with tf.Session() as sess:
        img_data = tf.image.decode_png(raw_image)
        adjusted = tf.image.flip_up_down(image=img_data)
        new_img = np.asarray(adjusted.eval(), dtype='uint8')
        if not overlap:
            save_image(dir=dir, image=new_img)
        else:
            save_image(dir=copied_name(dir), image=new_img)

        sess.close()


def flip_left_right(dir, overlap, value1, value2):
    """
    左右翻转
    :param dir:
    :param overlap:
    :return:
    """
    raw_image = tf.gfile.FastGFile(name=dir, mode='rb').read()
    with tf.Session() as sess:
        img_data = tf.image.decode_png(raw_image)
        adjusted = tf.image.flip_left_right(image=img_data)
        new_img = np.asarray(adjusted.eval(), dtype='uint8')
        if not overlap:
            save_image(dir=dir, image=new_img)
        else:
            save_image(dir=copied_name(dir), image=new_img)

        sess.close()


def transpose_image(dir, overlap, value1, value2):
    """
    图片转置，对角线翻转
    :param dir:
    :param overlap:
    :return:
    """
    raw_image = tf.gfile.FastGFile(name=dir, mode='rb').read()
    with tf.Session() as sess:
        img_data = tf.image.decode_png(raw_image)
        adjusted = tf.image.transpose_image(image=img_data)
        new_img = np.asarray(adjusted.eval(), dtype='uint8')
        if not overlap:
            save_image(dir=dir, image=new_img)
        else:
            save_image(dir=copied_name(dir), image=new_img)

        sess.close()


def adjust_brightness(dir, overlap, delta, value2):
    """
    调整亮度
    :param dir:
    :param overlap:
    :return:
    """
    raw_image = tf.gfile.FastGFile(name=dir, mode='rb').read()
    with tf.Session() as sess:
        img_data = tf.image.decode_png(raw_image)
        adjusted = tf.image.adjust_brightness(image=img_data, delta=delta)
        new_img = np.asarray(adjusted.eval(), dtype='uint8')
        if not overlap:
            save_image(dir=dir, image=new_img)
        else:
            save_image(dir=copied_name(dir), image=new_img)

        sess.close()
    pass


def random_brightness(dir, overlap, max_delta, value2):
    raw_image = tf.gfile.FastGFile(name=dir, mode='rb').read()
    with tf.Session() as sess:
        img_data = tf.image.decode_png(raw_image)
        adjusted = tf.image.random_brightness(image=img_data, max_delta=max_delta)
        new_img = np.asarray(adjusted.eval(), dtype='uint8')
        if not overlap:
            save_image(dir=dir, image=new_img)
        else:
            save_image(dir=copied_name(dir), image=new_img)
        sess.close()


def adjust_contrast(dir, overlap, contrast_factor, value2) -> True:
    raw_image = tf.gfile.FastGFile(name=dir, mode='rb').read()
    with tf.Session() as sess:
        img_data = tf.image.decode_png(raw_image)
        adjusted = tf.image.adjust_contrast(img_data, contrast_factor=contrast_factor)
        new_img = np.asarray(adjusted.eval(), dtype='uint8')
        if not overlap:
            save_image(dir=dir, image=new_img)
        else:
            save_image(dir=copied_name(dir), image=new_img)
        sess.close()


def random_contrast(dir, overlap, lower, upper):
    raw_image = tf.gfile.FastGFile(name=dir, mode='rb').read()
    with tf.Session() as sess:
        img_data = tf.image.decode_png(raw_image)
        adjusted = tf.image.random_contrast(image=img_data, lower=lower, upper=upper)
        new_img = np.asarray(adjusted.eval(), dtype='uint8')
        if not overlap:
            save_image(dir=dir, image=new_img)
        else:
            save_image(dir=copied_name(dir), image=new_img)
        sess.close()


def adjust_hue(dir, overlap, delta, value2):
    raw_image = tf.gfile.FastGFile(name=dir, mode='rb').read()
    with tf.Session() as sess:
        img_data = tf.image.decode_png(raw_image)
        adjusted = tf.image.adjust_hue(image=img_data, delta=delta)
        new_img = np.asarray(adjusted.eval(), dtype='uint8')
        if not overlap:
            save_image(dir=dir, image=new_img)
        else:
            save_image(dir=copied_name(dir), image=new_img)
        sess.close()


def random_hue(dir, overlap, max_delta, value2):
    raw_image = tf.gfile.FastGFile(name=dir, mode='rb').read()
    with tf.Session() as sess:
        img_data = tf.image.decode_png(raw_image)
        adjusted = tf.image.random_hue(image=img_data, max_delta=max_delta)
        new_img = np.asarray(adjusted.eval(), dtype='uint8')
        if not overlap:
            save_image(dir=dir, image=new_img)
        else:
            save_image(dir=copied_name(dir), image=new_img)
        sess.close()


def adjust_saturation(dir, overlap, saturation_factor, value2) -> True:
    raw_image = tf.gfile.FastGFile(name=dir, mode='rb').read()
    with tf.Session() as sess:
        img_data = tf.image.decode_png(raw_image)
        adjusted = tf.image.adjust_saturation(image=img_data, saturation_factor=saturation_factor)
        new_img = np.asarray(adjusted.eval(), dtype='uint8')
        if not overlap:
            save_image(dir=dir, image=new_img)
        else:
            save_image(dir=copied_name(dir), image=new_img)
        sess.close()


def random_saturation(dir, overlap, lower, upper) -> True:
    raw_image = tf.gfile.FastGFile(name=dir, mode='rb').read()
    with tf.Session() as sess:
        img_data = tf.image.decode_png(raw_image)
        adjusted = tf.image.random_saturation(image=img_data, lower=lower, upper=upper)
        new_img = np.asarray(adjusted.eval(), dtype='uint8')
        if not overlap:
            save_image(dir=dir, image=new_img)
        else:
            save_image(dir=copied_name(dir), image=new_img)
        sess.close()


def standardize(dir, overlap, value1, value2) -> True:
    raw_image = tf.gfile.FastGFile(name=dir, mode='rb').read()
    with tf.Session() as sess:
        img_data = tf.image.decode_png(raw_image)
        adjusted = tf.image.per_image_standardization(image=img_data)
        new_img = np.asarray(adjusted.eval(), dtype='uint8')
        if not overlap:
            save_image(dir=dir, image=new_img)
        else:
            save_image(dir=copied_name(dir), image=new_img)
        sess.close()


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

#TODO 图片会不是RGB或RGBA
def nl_denoise_colored(dir, overlap, h, value2):
    img = cv2.imread(dir, 0)
    print(img.shape)
    new_img = cv2.fastNlMeansDenoisingColored(img, None, h, 7, 21)
    if overlap:
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
