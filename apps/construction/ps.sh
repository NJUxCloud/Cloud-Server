#!/bin/sh

python base_distribute.py --mode=train --ps_hosts=10.1.30.2:23333 --worker_hosts=10.1.30.3:23333,10.1.30.4:23333 --job_name=ps --task_index=0 --config='