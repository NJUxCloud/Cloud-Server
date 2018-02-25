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
# 迭代数
flags.DEFINE_integer(flag_name="iter", default_value=2000, docstring="iter number")
# 学习率
flags.DEFINE_float(flag_name="learning_rate", default_value=0.1, docstring="learning rate")
# 损失函数
flags.DEFINE_string(flag_name="loss", default_value="平方差函数", docstring="loss function")


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


def net(x, keep_prob):
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
    iter = FLAGS.iter
    learning_rate=FLAGS.learning_rate

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
            y_conv = net(x, keep_prob)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

            global_step = tf.Variable(0)
            # 训练操作
            # 优化算法
            train_op = tf.train.AdagradOptimizer(learning_rate).minimize(
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
    predict = net(x, keep_prob)
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
