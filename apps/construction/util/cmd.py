def get_train_cmd(basedir, ps_host, worker_host, config, save_path, model_name, data_dir, result_path):
    """
    根据输入获取命令
    :param basedir:     construct_distribute.py 所在的文件夹
    :param ps_host:     字符串:127.0.0.1:22223
    :param worker_host:  字符串:127.0.0.1:22223
    :param config:      config配置字符串
    :param save_path:   模型保存路径
    :param model_name:  模型名称
    :param data_dir:    数据集所在路径
    :param result_path: json结果保存路径
    :return: 【ps命令,worker】命令
    """
    if basedir != "" and basedir[-1] != '/':
        basedir += "/"
    if save_path != "" and save_path[-1] != '/':
        save_path += "/"

    host_str = 'python %sconstruct_distribute.py --ps_hosts=%s --worker_hosts=%s  ' \
               '--job_name=ps --task_index=0 --config=\'%s\' --save_path=%s --model_name=%s' \
               ' --data_dir=%s --result=%s' \
               % (basedir, ps_host, worker_host, config, save_path, model_name, data_dir, result_path)

    worker_str = 'python %sconstruct_distribute.py --ps_hosts=%s --worker_hosts=%s ' \
                 '--job_name=worker --task_index=0 --config=\'%s\' --save_path=%s --model_name=%s' \
                 ' --data_dir=%s --result=%s' \
                 % (basedir, ps_host, worker_host, config , save_path, model_name, data_dir, result_path)
    return [host_str, worker_str]


def get_inference_cmd(basedir, config, save_path, model_name, target, result_path):
    """
    根据输入获取命令
    :param basedir:     construct_inference.py 所在的文件夹
    :param config:      config配置字符串
    :param save_path:   模型保存路径
    :param model_name:  模型名称
    :param target:      需要测试的图像路径
    :param result_path: json结果保存路径
    :return: 【ps命令,worker】命令
    """
    if basedir != "" and basedir[-1] != '/':
        basedir += "/"
    if save_path != "" and save_path[-1] != '/':
        save_path += "/"

    inf_str = 'python %sconstruct_inference.py --config=\'%s\' --save_path=%s --model_name=%s ' \
              '--target=%s --result=%s' \
              % (basedir, config, save_path, model_name, target, result_path)
    return inf_str



def test():
    config = '{"iter":"1000","learning_rate":"0.01","loss_name":"entropy",' \
             '"optimizer_name":"GradientDescentOptimizer","net_type":"CNN","net_config":' \
             '{"middle_layer":[{"layer":"conv","filter":[2,2,10]},{"layer":"conv","filter":' \
             '[2,2,20]},{"layer":"pool"},{"layer":"norm"},{"layer":"active"},{"layer":"connect"},' \
             '{"layer":"connect"}],"output_layer":{}}}'
    a = get_train_cmd("./", "127.0.0.1:22223", "127.0.0.1:22224", config, './model6', "model", "./data", "result.txt")
    print(a[0])
    print(a[1])
    print(get_inference_cmd("",config,"./model5","model","./test/test1.jpg","result.json"))

if __name__ == "__main__":
    test()
