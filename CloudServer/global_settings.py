# 可换成本地的download目录，方便测试，到时候部署在服务器上，此项为空
LOCAL_STORAGE_PATH = '/root/'

# 取得kubernetes结果的命令
KUBERNETES_RESULT_ORDER = 'kubectl describe node k8s-node-3 > /home/info.txt'

# 本地模型结果保存路径
LOCAL_TRAIN_RESULT_PATH = 'train_result/train.txt'
# 本地kubernetes结果保存路径
LOCAL_KUBERNETES_RESULT_PATH = 'kubernetes/info.txt'

# 本地infer结果保存路径
LOCAL_INFER_RESULT_PATH = 'infer_result/infer.json'

# ps和wk的pod
PS = 'ff2da74a1e08'
WK = '4ea0dbd83dfc'

PSHOSTS = '10.1.30.4:23333'
WKHOSTS = '10.1.30.3:23333'

# 取得训练结果的命令
TRAIN_RESULT_ORDER = 'docker cp ' + WK + ':/notebooks/%s /root/%s'
