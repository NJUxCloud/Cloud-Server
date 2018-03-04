#!/bin/sh

docker cp /root/mnist 796ce2dbf073:/notebooks
docker cp /root/mnist f3f8c72b32b6:/notebooks

docker exec -it 796ce2dbf073 /bin/bash
cd  mnist
nohup python construct_distribute.py --mode=train --ps_hosts=10.1.30.2:23333 --worker_hosts=10.1.30.3:23333 --job_name=ps --task_index=0 --config='{"iter":"1000","learning_rate":"0.01","loss_name":"entropy","optimizer_name":"GradientDescentOptimizer","net_type":"CNN","net_config":{"middle_layer":[{"layer":"conv","filter":[2,2,10]},{"layer":"conv","filter":[2,2,20]},{"layer":"pool"},{"layer":"norm"},{"layer":"active"},{"layer":"connect"},{"layer":"connect"}],"output_layer":{}}}'&
exit

docker exec -it f3f8c72b32b6 /bin/bash
cd  mnist
nohup python construct_distribute.py --mode=train --ps_hosts=10.1.30.2:23333 --worker_hosts=10.1.30.3:23333 --job_name=worker --task_index=0 --config='{"iter":"1000","learning_rate":"0.01","loss_name":"entropy","optimizer_name":"GradientDescentOptimizer","net_type":"CNN","net_config":{"middle_layer":[{"layer":"conv","filter":[2,2,10]},{"layer":"conv","filter":[2,2,20]},{"layer":"pool"},{"layer":"norm"},{"layer":"active"},{"layer":"connect"},{"layer":"connect"}],"output_layer":{}}}'&
exit

docker cp f3f8c72b32b6:/notebooks/mnist/train_model /root/mnist