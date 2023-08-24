partition=$1
node_num=$2
job_name=GTA2CITY_eval
ckpt_path=$3

mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
srun --mpi=pmi2 -p $partition -n1 --quotatype=auto --gres=gpu:$node_num --ntasks-per-node=1 --cpus-per-task=5  --job-name=${job_name} \
python -m torch.distributed.launch --nproc_per_node=$node_num --master_port 29051 main.py --config configs/GTA2Cityscapes/config.yaml --eval_only --load_path ${ckpt_path} 2>&1|tee log/train-$now.log &