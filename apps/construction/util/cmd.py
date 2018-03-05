def get_train_cmd(basedir, ps_host, worker_host, config, save_path, model_name, data_dir, result_path, ratio):
    """
    根据输入获取命令
    :param basedir:     construct_distribute_url.py 所在的文件夹
    :param ps_host:     字符串:127.0.0.1:22223
    :param worker_host:  字符串:127.0.0.1:22223
    :param config:      config配置字符串
    :param save_path:   模型保存路径
    :param model_name:  模型名称
    :param data_dir:    数据集所在路径
    :param result_path: json结果保存路径
    :param ratio:  训练集测试集比例
    :return: 【ps命令,worker】命令
    """
    if basedir != "" and basedir[-1] != '/':
        basedir += "/"
    if save_path != "" and save_path[-1] != '/':
        save_path += "/"

    host_str = 'python %sconstruct_distribute.py --mode=train --ps_hosts=%s --worker_hosts=%s  ' \
               '--job_name=ps --task_index=0 --config=\'%s\' --save_path=%s --model_name=%s' \
               ' --data_dir=%s --result=%s --ratio=%f' \
               % (basedir, ps_host, worker_host, config, save_path, model_name, data_dir, result_path, ratio)

    worker_str = 'python %sconstruct_distribute.py --mode=train --ps_hosts=%s --worker_hosts=%s ' \
                 '--job_name=worker --task_index=0 --config=\'%s\' --save_path=%s --model_name=%s' \
                 ' --data_dir=%s --result=%s --ratio=%f' \
                 % (basedir, ps_host, worker_host, config , save_path, model_name, data_dir, result_path, ratio)
    return [host_str, worker_str]


def get_inference_cmd(basedir, config, save_path, model_name, filename, result_path):
    """
    根据输入获取命令
    :param basedir:     construct_inference.py 所在的文件夹
    :param config:      config配置字符串
    :param save_path:   模型保存路径
    :param model_name:  模型名称
    :param filename:      需要测试的图像路径
    :param result_path: json结果保存路径
    :return: 【ps命令,worker】命令
    """
    if basedir != "" and basedir[-1] != '/':
        basedir += "/"
    if save_path != "" and save_path[-1] != '/':
        save_path += "/"

    inf_str = 'python %sconstruct_inference.py --config=\'%s\' --save_path=%s --model_name=%s ' \
              '--filename=%s --result=%s' \
              % (basedir, config, save_path, model_name, filename, result_path)
    return inf_str


def get_sample_train_cmd(ps_host, worker_host, config,  ratio):
    """
    根据输入获取命令
    :param ps_host:     字符串:127.0.0.1:22223
    :param worker_host:  字符串:127.0.0.1:22223
    :param config:      config配置字符串
    :param ratio:  训练集测试集比例
    :return: 【ps命令,worker】命令
    """
    host_str = 'python construct_distribute.py  --mode=train --ps_hosts=%s --worker_hosts=%s  ' \
               '--job_name=ps --task_index=0 --config=\'%s\' --ratio=%f' \
               % ( ps_host, worker_host, config, ratio)

    worker_str = 'python construct_distribute.py --mode=train --ps_hosts=%s --worker_hosts=%s ' \
                 '--job_name=worker --task_index=0 --config=\'%s\' --ratio=%f' \
                 % (ps_host, worker_host, config , ratio)
    return [host_str, worker_str]

def get_sameple_inference_cmd( config, filename):
    """
    根据输入获取命令
    :param basedir:     construct_inference.py 所在的文件夹
    :param config:      config配置字符串
    :param save_path:   模型保存路径
    :param model_name:  模型名称
    :param filename:      需要测试的图像路径
    :param result_path: json结果保存路径
    :return: 【ps命令,worker】命令
    """

    inf_str = 'python3 construct_inference.py --config=\'%s\' --filename=%s ' % (config,filename)
    return inf_str


def test():
    config = '{"iter":"1000","learning_rate":"0.01","loss_name":"entropy",' \
             '"optimizer_name":"GradientDescentOptimizer","net_type":"CNN","net_config":' \
             '{"middle_layer":[{"layer":"conv","filter":[2,2,10]},{"layer":"conv","filter":' \
             '[2,2,20]},{"layer":"pool"},{"layer":"norm"},{"layer":"active"},{"layer":"connect"},' \
             '{"layer":"connect"}],"output_layer":{}}}'
    a = get_train_cmd("./", "127.0.0.1:22223", "127.0.0.1:22224", config, './model6', "model", "./data",
                      "result.txt",0.8)
    print(a[0])
    print(a[1])
    print(get_inference_cmd("",config,"./model5","model","./test/test1.jpg","result.json"))

if __name__ == "__main__":
    test()
