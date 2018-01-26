from apps.preprocess.exceptions import WrongValueException
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import time

"""
add by wsw
    所有的预处理函数都在这里添加
    views 不用修改
    init是所有级联函数的最高层
    如果要写一个预处理函数:
    首先在其上一层的return中,添加预处理函数名和预处理中文显示名
    然后填写参数,每个参数要加入annotation(格式仿照4个具体resize
    格式为:("该参数在界面的中文显示","该参数的类型 e.g. str或float",该参数的下限,该函参数上限)
    后两个为界面提供输入的范围

    在扩号后面 接入 -> True/False
    标明这个函数是否是最终函数
"""


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
    return parts[0] + '_' + time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())) + '_copy' + parts[1]


def init() -> False:
    """
    初始化
    """
    return {
        'functions': [
            {
                'func': 'resize',
                'name': '图像缩放'
            },
            {
                'func': 'crop',
                'name': '图像裁剪'
            },
            {
                'func': 'flip',
                'name': '图像翻转'
            },
            {
                'func': 'adjust',
                'name': '图像调整'
            }
        ]
    }


def resize() -> False:
    """
    图像大小重构
    """
    return {
        'functions': [
            {
                'func': 'resize_nearest',
                'name': '邻域法'
            },
            {
                'func': 'resize_bicubic',
                'name': '双三次插值法'
            },
            {
                'func': 'resize_bilinear',
                'name': '双线性插值法'
            },
            {
                'func': 'resize_area',
                'name': '面积插值法'
            }
        ]
    }


def crop() -> False:
    """
    图像裁剪
    """
    return {
        'functions': [
            {
                'func': 'resize_image_with_crop_or_pad',
                'name': '裁剪填充'
            },
            {
                'func': 'random_crop',
                'name': '随机裁剪'
            }
        ]
    }


def flip() -> False:
    """
    图像翻转
    :return:
    """
    return {
        'functions': [
            {
                'func': 'flip_up_down',
                'name': '上下翻转'
            },
            {
                'func': 'flip_left_right',
                'name': '左右翻转'
            },
            {
                'func': 'transpose_image',
                'name': '对角线翻转'
            }
        ]
    }


def adjust() -> False:
    """
    图像调整
    :return:
    """
    return {
        'functions': [
            {
                'func': 'adjust_brightness',
                'name': '调整亮度'
            },
            {
                'func': 'random_brightness',
                'name': '随机调整亮度'
            },
            {
                'func': 'adjust_contrast',
                'name': '调整对比度'
            },
            {
                'func': 'random_contrast',
                'name': '随机调整对比度'
            },
            {
                'func': 'adjust_hue',
                'name': '调整色调'
            },
            {
                'func': 'random_hue',
                'name': '随机调整色调'
            },
            {
                'func': 'adjust_saturation',
                'name': '调整饱和度'
            },
            {
                'func': 'random_saturation',
                'name': '随机调整饱和度'
            },
            {
                'func': 'standardize',
                'name': '标准归一化'
            }
        ]
    }


def resize_nearest(dir: ("路径", "str", None, None), new_x: ("长度", "float", 150, 500),
                   new_y: ("宽度", "float", 150, 500), overlap: ("是否覆盖原数据", "bool", None, None)) -> True:
    """
    邻域法
    :param dir:
    :param new_x:
    :param new_y:
    :param overlap:
    """
    if new_x < 150 or new_x > 500 or new_y < 150 or new_y > 500:
        raise WrongValueException(message='参数数据大小不在规定范围150~500内！')

    raw_image = tf.gfile.FastGFile(name=dir, mode='rb').read()
    with tf.Session as sess:
        img_data = tf.image.decode_png(raw_image)
        resized = tf.image.resize_images(images=img_data, size=[new_x, new_y],
                                         method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        new_img = np.asarray(resized.eval(), dtype='uint8')
        if overlap:
            save_image(dir=dir, image=new_img)
        else:
            save_image(dir=copied_name(dir), image=new_img)

        sess.close()


def resize_bicubic(dir: ("路径", "str", None, None), new_x: ("长度", "float", 150, 500),
                   new_y: ("宽度", "float", 150, 500), overlap: ("是否覆盖原数据", "bool", None, None)) -> True:
    """双三次插值法"""
    if new_x < 150 or new_x > 500 or new_y < 150 or new_y > 500:
        raise WrongValueException(message='参数数据大小不在规定范围150~500内！')

    raw_image = tf.gfile.FastGFile(name=dir, mode='rb').read()
    with tf.Session as sess:
        img_data = tf.image.decode_png(raw_image)
        resized = tf.image.resize_images(images=img_data, size=[new_x, new_y],
                                         method=tf.image.ResizeMethod.BICUBIC)

        new_img = np.asarray(resized.eval(), dtype='uint8')
        if overlap:
            save_image(dir=dir, image=new_img)
        else:
            save_image(dir=copied_name(dir), image=new_img)

        sess.close()


def resize_bilinear(dir: ("路径", "str", None, None), new_x: ("长度", "float", 150, 500),
                    new_y: ("宽度", "float", 150, 500), overlap: ("是否覆盖原数据", "bool", None, None)) -> True:
    """
    双线性插值法
    :param dir:
    :param new_x:
    :param new_y:
    :param overlap:
    :return:
    """
    if new_x < 150 or new_x > 500 or new_y < 150 or new_y > 500:
        raise WrongValueException(message='参数数据大小不在规定范围150~500内！')

    raw_image = tf.gfile.FastGFile(name=dir, mode='rb').read()
    with tf.Session as sess:
        img_data = tf.image.decode_png(raw_image)
        resized = tf.image.resize_images(images=img_data, size=[new_x, new_y],
                                         method=tf.image.ResizeMethod.BILINEAR)

        new_img = np.asarray(resized.eval(), dtype='uint8')
        if overlap:
            save_image(dir=dir, image=new_img)
        else:
            save_image(dir=copied_name(dir), image=new_img)

        sess.close()


def resize_area(dir: ("路径", "str", None, None), new_x: ("长度", "float", 150, 500),
                new_y: ("宽度", "float", 150, 500), overlap: ("是否覆盖原数据", "bool", None, None)) -> True:
    """
    面积插值法
    :param dir:
    :param new_x:
    :param new_y:
    :param overlap:
    :return:
    """
    if new_x < 150 or new_x > 500 or new_y < 150 or new_y > 500:
        raise WrongValueException(message='参数数据大小不在规定范围150~500内！')

    raw_image = tf.gfile.FastGFile(name=dir, mode='rb').read()
    with tf.Session as sess:
        img_data = tf.image.decode_png(raw_image)
        resized = tf.image.resize_images(images=img_data, size=[new_x, new_y],
                                         method=tf.image.ResizeMethod.AREA)

        new_img = np.asarray(resized.eval(), dtype='uint8')
        if overlap:
            save_image(dir=dir, image=new_img)
        else:
            save_image(dir=copied_name(dir), image=new_img)

        sess.close()


def resize_image_with_crop_or_pad(dir: ("路径", "str", None, None), target_height_percent: ("目标百分比长度", "float", 0, 500),
                                  target_width_percent: ("目标百分比宽度", "float", 0, 500),
                                  overlap: ("是否覆盖原数据", "bool", None, None)) -> True:
    """
    裁剪或自动填充
    :param dir:
    :param target_height_percent:
    :param target_width_percent:
    :param overlap:
    :return:
    """
    pass


def random_crop(dir: ("路径", "str", None, None), target_height_percent: ("目标百分比长度", "float", 0, 100),
                target_width_percent: ("目标百分比宽度", "float", 0, 100),
                overlap: ("是否覆盖原数据", "bool", None, None)) -> True:
    """
    随即裁剪
    :param dir:
    :param target_height_percent:
    :param target_width_percent:
    :param overlap:
    :return:
    """
    pass


def flip_up_down(dir: ("路径", "str", None, None), overlap: ("是否覆盖原数据", "bool", None, None)) -> True:
    """
    上下翻转
    :param dir:
    :param overlap:
    :return:
    """
    pass


def flip_left_right(dir: ("路径", "str", None, None), overlap: ("是否覆盖原数据", "bool", None, None)) -> True:
    """
    左右翻转
    :param dir:
    :param overlap:
    :return:
    """
    pass


def transpose_image(dir: ("路径", "str", None, None), overlap: ("是否覆盖原数据", "bool", None, None)) -> True:
    """
    图片转置，对角线翻转
    :param dir:
    :param overlap:
    :return:
    """
    pass


def adjust_brightness(dir: ("路径", "str", None, None), overlap: ("是否覆盖原数据", "bool", None, None)) -> True:
    """
    调整亮度
    :param dir:
    :param overlap:
    :return:
    """
    pass


def random_brightness(dir: ("路径", "str", None, None), overlap: ("是否覆盖原数据", "bool", None, None)) -> True:
    pass


def adjust_contrast(dir: ("路径", "str", None, None), overlap: ("是否覆盖原数据", "bool", None, None)) -> True:
    pass


def random_contrast(dir: ("路径", "str", None, None), overlap: ("是否覆盖原数据", "bool", None, None)) -> True:
    pass


def adjust_hue(dir: ("路径", "str", None, None), overlap: ("是否覆盖原数据", "bool", None, None)) -> True:
    pass


def random_hue(dir: ("路径", "str", None, None), overlap: ("是否覆盖原数据", "bool", None, None)) -> True:
    pass


def adjust_saturation(dir: ("路径", "str", None, None), overlap: ("是否覆盖原数据", "bool", None, None)) -> True:
    pass


def random_saturation(dir: ("路径", "str", None, None), overlap: ("是否覆盖原数据", "bool", None, None)) -> True:
    pass


def standardize(dir: ("路径", "str", None, None), overlap: ("是否覆盖原数据", "bool", None, None)) -> True:
    pass
