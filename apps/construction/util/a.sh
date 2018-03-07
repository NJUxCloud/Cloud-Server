#!/bin/sh

docker cp /root/NJUCloud/1/model/modelname ff2da74a1e08:/notebooks
docker cp /root/NJUCloud/1/model/modelname 4ea0dbd83dfc:/notebooks

docker exec -it ff2da74a1e08 /bin/bash
cd  modelname
nohup python construct_distribute_url.py --mode=train --ps_hosts=10.1.30.4:23333 --worker_hosts=10.1.30.3:23333 --job_name=ps --task_index=0 --config='{"iter":"1000","learning_rate":"0.01","loss_name":"entropy","optimizer_name":"GradientDescentOptimizer","net_type":"CNN","net_config":{"middle_layer":[{"layer":"conv","filter":[2,2,10]},{"layer":"conv","filter":[2,2,20]},{"layer":"pool"},{"layer":"norm"},{"layer":"active"},{"layer":"connect"},{"layer":"connect"}],"output_layer":{}}}'&
exit

docker exec -it 4ea0dbd83dfc /bin/bash
cd  modelname
nohup python construct_distribute_url.py --mode=train --ps_hosts=10.1.30.4:23333 --worker_hosts=10.1.30.3:23333 --job_name=worker --task_index=0 --config='{"iter":"1000","learning_rate":"0.01","loss_name":"entropy","optimizer_name":"GradientDescentOptimizer","net_type":"CNN","net_config":{"middle_layer":[{"layer":"conv","filter":[2,2,10]},{"layer":"conv","filter":[2,2,20]},{"layer":"pool"},{"layer":"norm"},{"layer":"active"},{"layer":"connect"},{"layer":"connect"}],"output_layer":{}}}'&
exit

docker cp 4ea0dbd83dfc:/notebooks/modelname/train_model /root/NJUCloud/1/model/modelname
docker cp a4f96096622d:/notebooks/modelname/result.txt /root/NJUCloud/1/model/modelname

