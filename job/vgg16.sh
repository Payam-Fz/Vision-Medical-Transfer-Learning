#!/bin/bash
# a file for job output, you can check job progress
#SBATCH --output=../out/job_logs/vgg16_frozen%j.out

# a file for errors
#SBATCH --error=../out/job_logs/vgg16_frozen%j.err

# select the node edith
#SBATCH --partition=edith
#SBATCH --nodelist="jarvis3"

# use GPU
##SBATCH --gpus=geforce:1
##SBATCH --gpus=quadro_rtx_6000:1
#SBATCH --gpus=nvidia_geforce_gtx_1080_ti:1

# number of requested nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# memory per node
#SBATCH --mem=64000

# CPU allocated
#SBATCH --cpus-per-task=6

#SBATCH --job-name=med-ssl
#SBATCH --time=1-00:10:00

#----------------------------------------------------------

MYHOME=/ubc/cs/research/shield/projects/payamfz
PROJPATH=$MYHOME/medical-ssl-segmentation

conda activate tf2-gpu

# export CUDA_DIR=$PROJPATH/mycode
# export XLA_FLAGS=--xla_gpu_cuda_data_dir=$PROJPATH/mycode

# python $PROJPATH/mycode/neural_nets/vgg16.py \
#     --ouput_name=vgg16-trans-warmup \
#     --learning_rate=1e-3 --image_size=448 --epochs=2 --batch_size=64 --train_size=65536 \
#     --mode=train_then_eval --transfer_learning=True

python $PROJPATH/mycode/neural_nets/vgg16.py \
    --ouput_name=vgg16-continue \
    --learning_rate=1e-3 --image_size=448 --epochs=8 --batch_size=64 --train_size=65536 \
    --mode=train_then_eval --transfer_learning=True \
    --load_checkpoint=./out_archive/vgg16-fixed-loss-memory/vgg16-trans-warmup_2024-03-19_1041/model/checkpoints

# python $PROJPATH/mycode/neural_nets/vgg16_test.py \
#     --ouput_name=vgg16-miniBatchTest \
#     --learning_rate=1e-3 --image_size=448 --epochs=200 --batch_size=16 --train_size=32 \
#     --mode=train_then_eval --transfer_learning=True