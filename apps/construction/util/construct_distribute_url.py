# encoding: utf-8
# !/usr/bin/env python

import tensorflow as tf
import math
import os
import numpy as np
import genericpath
import json

# Define parameters
import time

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(flag_name="mode", default_value="train", docstring="train or predict")
flags.DEFINE_string(flag_name="ratio", default_value="0.8", docstring="train test ratio")
flags.DEFINE_string(flag_name="model_name", default_value="model", docstring="model name")
flags.DEFINE_string(flag_name="save_path", default_value="./train_model/", docstring="model path")
flags.DEFINE_string(flag_name="data_dir", default_value="data", docstring="Directory for storing input data")
flags.DEFINE_string(flag_name="filename", default_value="", docstring="specific target")
flags.DEFINE_string(flag_name="result", default_value="./result.txt", docstring="result place")

# For distributed
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

# hyper param
flags.DEFINE_integer(flag_name="iter", default_value=2000, docstring="iter number")

flags.DEFINE_string(flag_name="config", default_value="", docstring="train_config")


# 传入的训练配置参数

# change use
def is_already_save(save_path):
    return os.path.exists(save_path + ".meta")


def weight_variable(shape, init="norm", stddev_norm=0.1):
    """
    权重初始化
    :param shape:  权重形状
    :param init:  初始化方法  zero 全零     norm 正太分布    xavier Xavier
    :param stddev_norm: 标准差
    :return:
    """
    if init == "zero":
        var = tf.constant(0.0, shape=shape)
        return tf.Variable(var)
    elif init == "norm":
        var = tf.truncated_normal(shape=shape, stddev=stddev_norm)
        return tf.Variable(var)
    elif init == "xavier":
        fan_in = shape[1] * shape[2] * shape[3]
        var = tf.truncated_normal(shape=shape, stddev=stddev_norm * 1.0 / fan_in)
        return tf.Variable(var)
    else:
        assert False, init + "不是合法的初始化方法"


def bias_variable(shape, constant=0.1):
    """
    偏移初始化
    :param shape:
    :param constant:
    :return:
    """
    initial = tf.constant(constant, shape=shape)
    return tf.Variable(initial)


# construct part
def conv2d(inputs, filters, stride=(1, 1), padding='SAME', init="正太分布",
           isBias=False, bias_constant=0.1, stddev_norm=0.1):
    """
    卷基层
    :param inputs:  上一层输入
    :param filters:  卷积参数 [核高度, 核宽度, 输出层]
    :param stride:   步长
    :param padding:  对齐方式   SAME/VALID
    :param init:     初始化方式
    :param isBias:   是否使用偏移
    :param bias_constant:  偏移常量
    :param stddev_norm:    正太分布值(如果初始化方式不是全零会使用到
    :return:      下一层输入
    """

    filter_height = filters[0]
    filter_width = filters[1]
    output_channels = filters[2]
    input_channels = int(inputs.get_shape()[-1])
    weights_shape = [filter_height, filter_width, input_channels, output_channels]
    biases_shape = [output_channels]
    filters = weight_variable(weights_shape, init=init, stddev_norm=stddev_norm)

    if isBias:
        biases = bias_variable(biases_shape, bias_constant)
        return tf.nn.conv2d(inputs, filters, strides=[1, stride[0], stride[1], 1], padding=padding) + biases
    else:
        return tf.nn.conv2d(inputs, filters, strides=[1, stride[0], stride[1], 1], padding=padding)


def max_pool(x, kernel=(2, 2), stride=(2, 2), padding="SAME"):
    """
    最大池化池化层
    :param x:  上一层输入
    :param kernel:   卷积核大小 [高度, 宽度]
    :param stride:   步长大小 [高度, 宽度]
    :param padding:   对齐方式  SAME/VALID
    :return:
    """
    return tf.nn.max_pool(x, ksize=[1, kernel[0], kernel[1], 1], strides=[1, stride[0], stride[1], 1], padding=padding)


def active(x, active_func, param=None):
    """
    选择激活函数
    :param x:  上一层输入
    :param active_func:  激活函数  str类型
    :param param:   数组,按照不同函数的参数顺序
    :return:
    """
    if param is None:
        param = [0.2]
    if active_func == "relu":
        return tf.nn.relu(x)
    elif active_func == "sigmoid":
        return tf.nn.sigmoid(x)
    elif active_func == "leaky_relu":
        if param is None or len(param) == 0:
            return tf.maximum(x, 0.2 * x)
        return tf.maximum(x, param[0] * x)
    else:
        assert False, active_func + "不是合法的激活函数"


def normalize(inputs, var_epsilon=1e-3):
    """
    归一化 操作
    :param inputs: 上一层输入
    :param var_epsilon: epsilon 值
    :return: 归一化之后的结结果
    """
    scale = tf.Variable(tf.ones([int(inputs.get_shape()[-1])]))
    offset = tf.Variable(tf.zeros([int(inputs.get_shape()[-1])]))
    mean, var = tf.nn.moments(inputs, list(range(len(inputs.get_shape()) - 1)))
    return tf.nn.batch_normalization(inputs, mean, var, offset, scale, var_epsilon)


def connect_layer(x, hidden):
    """
    全连接层  只后面不能加卷基层 池化层 和 归一化层
    :param x:  上一层输入
    :param hidden:  全连接层神经元个数
    :return: 全连接之后的结果
    """
    shape = x.get_shape()
    if len(shape) != 2:
        re = tf.reshape(x, [-1, int(shape[3]) * int(shape[1]) * int(shape[2])])
        x = re
    shape = x.get_shape()
    w = weight_variable([int(shape[1]), hidden])
    b = bias_variable([hidden])
    return tf.matmul(x, w) + b


def save_image(image, path):
    with open(path, "wb") as file:
        file.write(image)


def get_model(path, name):
    """
    这个函数是为了找到训练数量最多的ckpt文件
    :param path:    ckpt文件所在目录
    :param name:    model 的名字
    :return:
    """
    maxstep = -1
    for filename in os.listdir(path):
        if filename.startswith(name + ".ckpt-") and filename.endswith(".meta"):
            str = filename[(len(name) + 6):][:-5]
            step = int(str)
            if step > maxstep:
                maxstep = step

    return maxstep


def cnn(net_config, x):
    '''
    构造cnn网络
    :param net_config: 网络配置
    :param x:     预处理后的输入
    :return:    构造得到的神经网络
    '''
    middle_layer = net_config["middle_layer"]
    hidden = [tf.reshape(x, [-1, 28, 28, 1])]  # 用于保存每一层调用对应方法的结果，作为下一层的输入
    # TODO 暂时先固定这个图像的大小
    # 加入初始的一个值
    # 每一层由好几个小层组成
    for every_inner in middle_layer:
        layer_name = every_inner["layer"]
        if layer_name == "conv":
            # 获取本层参数
            filters = every_inner["filter"]
            stride = [1, 1] if "stride" not in every_inner.keys() else every_inner["stride"]
            padding = "SAME" if "padding" not in every_inner.keys() else every_inner["padding"]
            init = "norm" if "init" not in every_inner.keys() else every_inner["init"]
            isBias = False if "isBias" not in every_inner.keys() else every_inner["isBias"] != "False"
            bias_constant = 0.1 if "bias_constant" not in every_inner.keys() else every_inner["bias_constant"]
            stddev_norm = 0.1 if "stddev_norm" not in every_inner.keys() else every_inner["stddev_norm"]
            # 生成新的一层
            hidden.append(conv2d(hidden[len(hidden) - 1], filters, stride, padding, init=init,
                                 isBias=isBias, bias_constant=bias_constant, stddev_norm=stddev_norm))
        elif layer_name == "pool":
            # 获取本层参数
            kernel = [2, 2] if "kernel" not in every_inner.keys() else every_inner["kernel"]
            stride = [2, 2] if "stride" not in every_inner.keys() else every_inner["stride"]
            padding = "SAME" if "padding" not in every_inner.keys() else every_inner["padding"]
            # 生成新的一层
            hidden.append(max_pool(hidden[len(hidden) - 1], kernel=kernel, stride=stride, padding=padding))
        elif layer_name == "active":
            active_func = "sigmoid" if "active_func" not in every_inner.keys() else every_inner["active_func"]
            param = [0.2] if "param" not in every_inner.keys() else every_inner["param"]
            hidden.append(active(hidden[len(hidden) - 1], active_func=active_func, param=param))
        elif layer_name == "connect":
            hide = 512 if "hidden" not in every_inner.keys() else every_inner["hidden"]
            hidden.append(connect_layer(hidden[len(hidden) - 1], hidden=hide))
        elif layer_name == "norm":
            epsilon = 1e-3 if "epsilon" not in every_inner.keys() else every_inner["epsilon"]
            hidden.append(normalize(hidden[len(hidden) - 1], epsilon))

    # 处理输出层
    # output_layer = net_config["output_layer"]
    last = hidden[len(hidden) - 1]
    shape = last.get_shape()
    if len(shape) != 2:
        re = tf.reshape(x, [-1, int(shape[3]) * int(shape[1]) * int(shape[2])])
        last = re
    shape = last.get_shape()

    # TODO 假设按照输出为10
    w = weight_variable([int(shape[1]), 10])
    b = bias_variable([10])
    y_conv = tf.matmul(last, w) + b
    return y_conv


# 网络类型
def get_net(net_type, net_config, x, keep_prob):
    '''
    根据神经网络类型和模型设置获取网络
    :param net_type:
    :param net_config:
    :param x:
    :param keep_prob:
    :return:
    '''
    if net_type == "CNN":
        return cnn(net_config, x)

    return cnn(net_config, x)


# 损失函数
def loss_function(name, logits, labels):
    """
    定义损失函数
    :param name:  损失函数名称 mse:均方误差    entropy 交叉熵
    :param logits:
    :param labels:
    :return:
    """

    # 二次距离 TODO 需要修改
    if name == "mse":
        return tf.reduce_mean(tf.square(labels - logits))
    elif name == "entropy":
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))


def optimizer_function(optimizer_name, learning_rate):
    """
    定义优化算法
    :param optimizer_name:
    :param learning_rate:
    :return:
    """
    if optimizer_name == "GradientDescentOptimizer":
        return tf.train.GradientDescentOptimizer(learning_rate)
    else:
        return tf.train.AdagradOptimizer(learning_rate)


# def net(x, keep_prob):
#     W_conv1 = weight_variable([5, 5, 1, 32])
#     b_conv1 = bias_variable([32])
#     x_image = tf.reshape(x, [-1, 28, 28, 1])
#
#     h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#     h_pool1 = max_pool_2x2(h_conv1)
#
#     W_conv2 = weight_variable([5, 5, 32, 64])
#     b_conv2 = bias_variable([64])
#
#     h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#     h_pool2 = max_pool_2x2(h_conv2)
#
#     W_fc1 = weight_variable([7 * 7 * 64, 1024])
#     b_fc1 = bias_variable([1024])
#
#     h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
#     h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#
#     # keep_prob = tf.placeholder(tf.float32)
#     h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#
#     W_fc2 = weight_variable([1024, 10])
#     b_fc2 = bias_variable([10])
#
#     y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
#     return y_conv


def byteify(input):
    if isinstance(input, dict):
        return {byteify(key): byteify(value) for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [byteify(element) for element in input]
    elif isinstance(input, np.unicode):
        return input.encode('utf-8')
    else:
        return input


def main(_):
    # 模型环境信息
    model_name = FLAGS.model_name
    pre = FLAGS.save_path
    save_path = FLAGS.save_path
    save_path = save_path + model_name + ".ckpt"
    result_path = FLAGS.result
    iter = FLAGS.iter

    config = FLAGS.config
    config = json.loads(config)
    config = byteify(config)
    iter = int(config["iter"])  # 迭代数
    learning_rate = float(config["learning_rate"])  # 学习率
    loss_name = config["loss_name"]  # 损失函数
    optimizer_name = config["optimizer_name"]  # 优化算法
    net_type = config["net_type"]  # 神经网络（传统还是cnn）

    net_config = config[u"net_config"]  # 神经网络参数


    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):

            from tensorflow.examples.tutorials.mnist import input_data
            mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
            # Create the model
            x = tf.placeholder(tf.float32, [None, 784])
            # Define loss and optimizer
            keep_prob = tf.placeholder(tf.float32)
            y_ = tf.placeholder(tf.float32, [None, 10])

            y_conv = get_net(net_type, net_config, x, keep_prob)
            loss = loss_function(loss_name, y_conv, y_)

            global_step = tf.Variable(0)
            # 训练操作
            train_op = tf.train.AdagradOptimizer(1e-4).minimize(
                loss, global_step=global_step)
            # 保存器
            saver = tf.train.Saver()
            # 统计操作
            summary_op = tf.summary.merge_all()
            # 初始化操作
            init_op = tf.global_variables_initializer()
            # 评价操作
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Create a "supervisor", which oversees the training process.
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir=pre,
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=60
                                 )

        with sv.managed_session(server.target) as sess:
            # Loop until the supervisor shuts down or 1000 steps have completed.
            step = 0
            duration = 0.0;
            # 这个加载是为restore之前训练的最新结果
            save_path = save_path + model_name + ".ckpt-" + str(get_model(FLAGS.save_path, FLAGS.model_name))
            if is_already_save(save_path):
                saver.restore(sess, save_path)
            while not sv.should_stop() and step < iter:
                batch = mnist.train.next_batch(50)
                step = sess.run(global_step)
                if step % 100 == 0:
                    step = sess.run(global_step)
                    train_accuracy = sess.run(accuracy, feed_dict={
                        x: batch[0], y_: batch[1], keep_prob: 1.0})
                    with open(result_path, 'a') as file:
                        result_line = 'step:%d,accuracy:%f,duration:%f\n' % (step, train_accuracy, float(duration))
                        file.write(result_line)
                start_time = time.time()
                sess.run(train_op, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
                duration = time.time() - start_time
                # Ask for all the services to stop.
            test_accuracy = sess.run(accuracy, feed_dict={
                x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
            with open(result_path, 'a') as file:
                result_line = 'final_accuracy:%f\n' % test_accuracy
                file.write(result_line+"\n")
        sv.stop()


if __name__ == "__main__":
    tf.app.run()
