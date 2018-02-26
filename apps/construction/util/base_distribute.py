# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import genericpath
import numpy as np
import os
import tensorflow as tf
from PIL import Image
from os.path import splitdrive

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string(flag_name="mode", default_value="predict", docstring="train or predict")
flags.DEFINE_string(flag_name="model_name", default_value="model", docstring="model name")
flags.DEFINE_string(flag_name="save_path", default_value="./model2/", docstring="model path")
flags.DEFINE_string(flag_name="data_dir", default_value="data", docstring="Directory for storing input data")
flags.DEFINE_string(flag_name="filename", default_value="", docstring="specific target")

# define distribute part
flags.DEFINE_string(flag_name="ps_hosts", default_value="", docstring="Comma-separated list of hostname:port pairs")
flags.DEFINE_string(flag_name="worker_hosts", default_value="", docstring="Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
flags.DEFINE_string(flag_name="job_name", default_value="", docstring="One of 'ps', 'worker'")
flags.DEFINE_integer(flag_name="task_index", default_value=0, docstring="Index of task within the job")

# 传入的训练配置参数
flags.DEFINE_string(flag_name="config", default_value="", docstring="train_config")

# Join two (or more) paths.
def join(path, *paths):
    if isinstance(path, bytes):
        sep = b'\\'
        seps = b'\\/'
        colon = b':'
    else:
        sep = '\\'
        seps = '\\/'
        colon = ':'
    try:
        if not paths:
            path[:0] + sep  # 23780: Ensure compatible data type even if p is null.
        result_drive, result_path = splitdrive(path)
        for p in paths:
            p_drive, p_path = splitdrive(p)
            if p_path and p_path[0] in seps:
                # Second path is absolute
                if p_drive or not result_drive:
                    result_drive = p_drive
                result_path = p_path
                continue
            elif p_drive and p_drive != result_drive:
                if p_drive.lower() != result_drive.lower():
                    # Different drives => ignore the first path entirely
                    result_drive = p_drive
                    result_path = p_path
                    continue
                # Same drive in different case
                result_drive = p_drive
            # Second path is relative to the first
            if result_path and result_path[-1] not in seps:
                result_path += sep
            result_path = result_path + p_path
        # add separator between UNC and non-absolute path
        if (result_path and result_path[0] not in seps and
                result_drive and result_drive[-1:] != colon):
            return result_drive + sep + result_path
        return result_drive + result_path
    except (TypeError, AttributeError, BytesWarning):
        genericpath._check_arg_types('join', path, *paths)
        raise


def is_already_save(save_path):
    return os.path.exists(save_path + ".meta")

def to_int_array(str):
    '''
    '1,1,1,1' 变成[1,1,1,1]
    字符串变成数组
    :param str:
    :return:
    '''
    arr = str.split(',')
    arr = np.array(arr).astype(np.int).tolist()
    return arr


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def active(type,features):
    '''
    选择激活函数
    :param type:
    :param features:
    :return:
    '''
    if(type=="ReLU函数"):
        return tf.nn.relu(features)
    elif(type=="Sigmoid函数"):
        return tf.nn.sigmoid(features)
    return tf.nn.relu(features)

def normalize(x,shift,scale,epsilon):
    '''
    归一化处理
    :param x:
    :param shift:
    :param scale:
    :param epsilon:
    :return:
    '''
    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], keep_dims=True)
    shift = to_int_array(shift)
    scale = to_int_array(scale)
    epsilon = float('e'+epsilon)
    return tf.nn.batch_normalization(x, batch_mean, batch_var, shift, scale, epsilon)

def connect_layer(x,w,b):
    return tf.nn.relu(tf.matmul(x, w) + b)

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

def loss_function(name,logits,labels):
    '''
    定义损失函数
    :param name:
    :param logits:
    :param labels:
    :return:
    '''
    if(name=="平方差函数"):
        return
    elif(name=="交叉熵函数"):
        return  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))

def optimizer_function(optimizer_name,learning_rate):
    '''
    定义优化算法
    :param optimizer_name:
    :param learning_rate:
    :return:
    '''
    if(optimizer_name=="GradientDescentOptimizer"):
        return tf.train.GradientDescentOptimizer(learning_rate)
    else:
        return tf.train.AdagradOptimizer(learning_rate)


def default_net(x, keep_prob):
    '''
    默认网络
    :param x:
    :param keep_prob:
    :return:
    '''
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv

def cnn(net_config,x):
    '''
    构造cnn网络
    :param net_config:
    :param x:
    :return:
    '''
    middle_layer=net_config["middle_layer"]
    hidden=[]   #用于保存每一层调用对应方法的结果，作为下一层的输入
    for every_layer in middle_layer:
        w=weight_variable(to_int_array(every_layer["W"]))
        b=bias_variable(to_int_array(every_layer["b"]))
        #加入初始的一个值
        hidden.append( tf.reshape(x, [-1, 28, 28, 1]))
        #每一层由好几个小层组成
        inner_layer=every_layer["inner_layer"]
        for every_inner in inner_layer:
            layer_name=every_inner["layer"]
            if(layer_name=="卷积层"):
                hidden.append(conv2d(hidden[len(hidden)-1], w)+b)
            elif(layer_name=="池化层"):
                hidden.append(max_pool_2x2(hidden[len(hidden)-1]))
            elif(layer_name=="激活层"):
                hidden.append(active(every_inner["激活函数"],hidden[len(hidden)-1]))
            elif(layer_name=="全连接层"):
                hidden.append(connect_layer(hidden[len(hidden)-1],w,b))
            elif(layer_name=="归一化层"):
                hidden.append(normalize(hidden[len(hidden)-1],every_inner["shift"],every_inner["scale"],every_inner["epsilon"]))

    #处理输出层
    output_layer=net_config["output_layer"]
    w = weight_variable(to_int_array(output_layer["W"]))
    b = bias_variable(to_int_array(output_layer["b"]))
    y_conv = tf.matmul(hidden[len(hidden)-1], w) + b
    return y_conv


def general_net(net_config,x):
    '''
    构造传统神经网络
    :param net_config:
    :param x:
    :return:
    '''
    #处理隐藏层
    hidden_layer=net_config["hidden_layer"]
    hidden = []  # 用于保存每一层调用对应方法的结果，作为下一层的输入
    for every_layer in hidden_layer:
        w = weight_variable(to_int_array(every_layer["W"]))
        b = bias_variable(to_int_array(every_layer["b"]))

    # 处理输出层
    output_layer = net_config["output_layer"]
    w = weight_variable(to_int_array(output_layer["W"]))
    b = bias_variable(to_int_array(output_layer["b"]))
    #todo:传统的如何处理？


def get_net(net_type,net_config,x,keep_prob):
    '''
    根据神经网络类型和模型设置获取网络
    :param net_type:
    :param net_config:
    :param x:
    :param keep_prob:
    :return:
    '''
    if(net_type=="CNN"):
        return cnn(net_config,x)
    elif(net_type=="传统神经网络"):
        return general_net(net_config,x)
    return default_net(x,keep_prob)

def train():
    """
    分布式训练部分
    :return:
    """
    # 模型环境信息
    model_name = FLAGS.model_name
    pre = FLAGS.save_path
    save_path = FLAGS.save_path
    save_path = join(save_path, model_name + ".ckpt")
    config=FLAGS.config
    iter = int(config["iter"])    # 迭代数
    learning_rate=float(config["learning_rate"])  # 学习率
    loss_name=config["loss_name"]   # 损失函数
    optimizer_name=config["optimizer_name"] # 优化算法
    net_type=config["net_type"] # 神经网络（传统还是cnn）
    net_config=config["net_config"]  # 神经网络参数

    # assign distribute information
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index,
                             )
    print("start")

    # 如果是参数服务器那么只负责更新参数
    if FLAGS.job_name == "ps":
        server.join()

    # 如果是woker那么运行网络并且计算梯度更新
    elif FLAGS.job_name == "worker":

        # Assigns ops to the local worker by default.
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
            y_conv = get_net(net_type,net_config,x, keep_prob)
            loss = loss_function(loss_name,y_conv,y_)

            global_step = tf.Variable(0)
            # 训练操作
            # 优化算法
            tf.train.GradientDescentOptimizer()
            train_op = optimizer_function(optimizer_name,learning_rate).minimize(
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

        # The supervisor takes care of session initialization, restoring from
        # a checkpoint, and closing when done or an error occurs.
        with sv.prepare_or_wait_for_session(server.target) as sess:
            # Loop until the supervisor shuts down or 1000 steps have completed.
            step = 0
            # 这个加载是为restore之前训练的最新结果
            save_path = join(save_path, model_name + ".ckpt-"
                             + str(get_model(FLAGS.save_path, FLAGS.model_name)))
            if is_already_save(save_path):
                saver.restore(sess, save_path)
            while not sv.should_stop() and step < iter:
                # Run a training step asynchronously.
                batch = mnist.train.next_batch(50)
                step = sess.run(global_step)
                if step % 100 == 0:
                    step = sess.run(global_step)
                    train_accuracy = accuracy.eval(feed_dict={
                        x: batch[0], y_: batch[1], keep_prob: 1.0})
                    print("step %d, training accuracy %g" % (step, train_accuracy))
                train_op.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
                # Ask for all the services to stop.
            print("test accuracy %g" % accuracy.eval(feed_dict={
                x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
        sv.stop()


def inference(filename):
    """
    推断部分 和非分布式一样,除了加载模型的部分
    :param filename:
    :return:
    """
    config = FLAGS.config
    net_type = config["net_type"]  # 神经网络（传统还是cnn）
    net_config = config["net_config"]  # 神经网络参数
    model_name = FLAGS.model_name
    save_path = FLAGS.save_path
    save_path = join(save_path, model_name + ".ckpt-"
                     + str(get_model(FLAGS.save_path, FLAGS.model_name)))
    if not is_already_save(save_path):
        print("no model please train a model first")
        return
    im = Image.open(filename)
    arr = np.asarray(im)
    arr = np.array(arr, dtype='float32')
    arr /= 255.0
    arr = arr.flatten()
    # print(np.array([arr]).shape)
    x = tf.placeholder(tf.float32, [None, 784])
    keep_prob = tf.placeholder(tf.float32)
    predict = get_net(net_type,net_config,x, keep_prob)
    result = tf.argmax(predict, 1)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, save_path)
        with tf.device('/cpu:0'):
            re = sess.run(result, feed_dict={x: np.array([arr]), keep_prob: 1.0})
            print(re)
            ans = re[0]
    print("the predict result is " + str(ans))


def main(_):
    if FLAGS.mode == 'train':
        train()
    else:
        inference(FLAGS.filename)


if __name__ == "__main__":
    tf.app.run()
