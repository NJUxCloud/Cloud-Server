# 界面可选项


def neural_network_type():
    return {
        'options': [
            '传统神经网络', 'CNN'
        ],
        'default': 'CNN'
    }


def loss_function():
    return {
        'options': [
            '平方差函数', '交叉熵函数'
        ],
        'default': '平方差函数'
    }


def optimizer():
    return {
        'options': [
            'Gradient Descent Optimizer',
            'Adadelta Optimizer',
            'Adagrad Optimizer',
            'Adam Optimizer'
        ],
        'default': 'Gradient Descent Optimizer'
    }


def param_init():
    return {
        'options': [
            '全零', '正态分布', 'Xavier'
        ],
        'default': '全零'
    }


def activation_method():
    return {
        'options': [
            'Sigmoid', 'ReLU'
        ],
        'default': 'ReLU'
    }


def padding_method():
    return {
        'options': [
            'SAME', 'VALID'
        ],
        'default': 'SAME'
    }
