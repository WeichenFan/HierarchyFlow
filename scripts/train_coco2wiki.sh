
partition=$1
node_num=$2

mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
srun --mpi=pmi2 -p $partition -n$node_num --quotatype=auto --gres=gpu:$node_num --ntasks-per-node=$node_num --cpus-per-task=5  --job-name=s1 \
python3.6 ../main.py --config ../config/COCO2Wiki/arch_3_30_120_360/seed/COCO2Wiki_k_8_b3_3_30_120_360_s1.yaml 2>&1|tee log/train-$now.log &

sleep 1s

mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
srun --mpi=pmi2 -p $partition -n$node_num --quotatype=auto --gres=gpu:$node_num --ntasks-per-node=$node_num --cpus-per-task=5  --job-name=s2 \
python3.6 ../main.py --config ../config/COCO2Wiki/arch_3_30_120_360/seed/COCO2Wiki_k_8_b3_3_30_120_360_s2.yaml 2>&1|tee log/train-$now.log &

sleep 1s

mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
srun --mpi=pmi2 -p $partition -n$node_num --quotatype=auto --gres=gpu:$node_num --ntasks-per-node=$node_num --cpus-per-task=5  --job-name=s3 \
python3.6 ../main.py --config ../config/COCO2Wiki/arch_3_30_120_360/seed/COCO2Wiki_k_8_b3_3_30_120_360_s3.yaml 2>&1|tee log/train-$now.log &

sleep 1s

mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
srun --mpi=pmi2 -p $partition -n$node_num --quotatype=auto --gres=gpu:$node_num --ntasks-per-node=$node_num --cpus-per-task=5  --job-name=s4 \
python3.6 ../main.py --config ../config/COCO2Wiki/arch_3_30_120_360/seed/COCO2Wiki_k_8_b3_3_30_120_360_s4.yaml 2>&1|tee log/train-$now.log &

sleep 1s


mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
srun --mpi=pmi2 -p $partition -n$node_num --quotatype=auto --gres=gpu:$node_num --ntasks-per-node=$node_num --cpus-per-task=5  --job-name=s5 \
python3.6 ../main.py --config ../config/COCO2Wiki/arch_3_30_120_360/seed/COCO2Wiki_k_8_b3_3_30_120_360_s5.yaml 2>&1|tee log/train-$now.log &

sleep 1s

mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
srun --mpi=pmi2 -p $partition -n$node_num --quotatype=auto --gres=gpu:$node_num --ntasks-per-node=$node_num --cpus-per-task=5  --job-name=s6 \
python3.6 ../main.py --config ../config/COCO2Wiki/arch_3_30_120_360/seed/COCO2Wiki_k_8_b3_3_30_120_360_s6.yaml 2>&1|tee log/train-$now.log &

sleep 1s

mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
srun --mpi=pmi2 -p $partition -n$node_num --quotatype=auto --gres=gpu:$node_num --ntasks-per-node=$node_num --cpus-per-task=5  --job-name=s7 \
python3.6 ../main.py --config ../config/COCO2Wiki/arch_3_30_120_360/seed/COCO2Wiki_k_8_b3_3_30_120_360_s7.yaml 2>&1|tee log/train-$now.log &

sleep 1s

mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
srun --mpi=pmi2 -p $partition -n$node_num --quotatype=auto --gres=gpu:$node_num --ntasks-per-node=$node_num --cpus-per-task=5  --job-name=s8 \
python3.6 ../main.py --config ../config/COCO2Wiki/arch_3_30_120_360/seed/COCO2Wiki_k_8_b3_3_30_120_360_s8.yaml 2>&1|tee log/train-$now.log &

sleep 1s

mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
srun --mpi=pmi2 -p $partition -n$node_num --quotatype=auto --gres=gpu:$node_num --ntasks-per-node=$node_num --cpus-per-task=5  --job-name=s9 \
python3.6 ../main.py --config ../config/COCO2Wiki/arch_3_30_120_360/seed/COCO2Wiki_k_8_b3_3_30_120_360_s9.yaml 2>&1|tee log/train-$now.log &

sleep 1s

mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
srun --mpi=pmi2 -p $partition -n$node_num --quotatype=auto --gres=gpu:$node_num --ntasks-per-node=$node_num --cpus-per-task=5  --job-name=s10 \
python3.6 ../main.py --config ../config/COCO2Wiki/arch_3_30_120_360/seed/COCO2Wiki_k_8_b3_3_30_120_360_s10.yaml 2>&1|tee log/train-$now.log &
