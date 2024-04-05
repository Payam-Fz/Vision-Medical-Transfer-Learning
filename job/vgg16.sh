#!/bin/bash
# a file for job output, you can check job progress
#SBATCH --output=../out/job_logs/vgg16-gradual-1-4_%j.out

# a file for errors
#SBATCH --error=../out/job_logs/vgg16-gradual-1-4_%j.err

# select the node edith
#SBATCH --partition=edith
#SBATCH --nodelist="edith4"

# use GPU
##SBATCH --gpus=geforce:1
#SBATCH --gpus=quadro_rtx_6000:1
##SBATCH --gpus=nvidia_geforce_gtx_1080_ti:1

# number of requested nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# memory per node
#SBATCH --mem=64000

# CPU allocated
#SBATCH --cpus-per-task=6

#SBATCH --job-name=vgg
#SBATCH --time=6-00:10:00

#----------------------------------------------------------

MYHOME=/ubc/cs/research/shield/projects/payamfz
PROJPATH=$MYHOME/medical-ssl-segmentation

conda activate tf2-gpu

export CUDA_DIR=$PROJPATH/mycode
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$PROJPATH/mycode

# ---------- WARMUP ----------- #
# python $PROJPATH/mycode/neural_nets/vgg16.py \
#     --ouput_name=vgg16-trans-warmup \
#     --learning_rate=1e-3 --image_size=448 --epochs=2 --batch_size=64 --train_size=65536 \
#     --mode=train_then_eval --transfer_learning=True

# ---------- CONTINUE TRAIN ----------- #
# python $PROJPATH/mycode/neural_nets/vgg16.py \
#     --ouput_name=vgg16-unfreeze-4 \
#     --learning_rate=1e-6 --image_size=448 --epochs=8 --batch_size=64 --train_size=65536 \
#     --mode=train_then_eval --unfreeze_blocks=4 \
#     --load_checkpoint=./out_archive/vgg16-fixed-loss-memory/vgg16-trans-warmup_2024-03-19_1041/model/checkpoints

# ---------- EVALUATE ----------- #
# python $PROJPATH/mycode/neural_nets/vgg16.py \
#     --ouput_name=vgg16-unfreeze \
#     --learning_rate=1e-6 --image_size=448 --epochs=8 --batch_size=64 --train_size=65536 \
#     --mode=eval --unfreeze_blocks=2 \
#     --load_checkpoint=./out_archive/vgg16-fixed-loss-memory/vgg16-unfreeze_2024-03-24_2251/model/checkpoints

# ---------- MINI-BATCH TEST ----------- #
# python $PROJPATH/mycode/neural_nets/vgg16_test.py \
#     --ouput_name=vgg16-miniBatchTest \
#     --learning_rate=1e-3 --image_size=448 --epochs=200 --batch_size=16 --train_size=32 \
#     --mode=train_then_eval --transfer_learning=True

# ---------- FULL TRAIN ----------- #
python $PROJPATH/mycode/neural_nets/vgg16.py \
    --ouput_name=vgg16-gradual-1-4 \
    --learning_rate=1e-5 --image_size=224 --epochs=30 --batch_size=64 --train_size=65536 \
    --mode=train_then_eval --min_unfreeze_blocks=1 --max_unfreeze_blocks=4 \
    --load_checkpoint=./out/vgg16-gradual-fix-color_2024-04-03_0941/0_unfrozen_block/model/checkpoints