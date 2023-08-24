partition=$1
node_num=$2
job_name=GTA2CITY

mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
srun --mpi=pmi2 -p $partition -n1 --quotatype=auto --gres=gpu:$node_num --ntasks-per-node=1 --cpus-per-task=40  --job-name=${job_name} \
python -m torch.distributed.launch --nproc_per_node=$node_num --master_port 29051 main.py --config configs/GTA2Cityscapes/config.yaml 2>&1|tee log/train-$now.log &