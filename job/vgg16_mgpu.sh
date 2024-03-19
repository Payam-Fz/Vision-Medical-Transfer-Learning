#!/bin/bash
# a file for job output, you can check job progress
#SBATCH --output=../out/job_logs/vgg16%j.out

# a file for errors
#SBATCH --error=../out/job_logs/vgg16%j.err

# select the node edith
#SBATCH --partition=edith
#SBATCH --nodelist="jarvis3"

# use GPU
##SBATCH --gpus=geforce:4
#SBATCH --gpus=nvidia_geforce_gtx_1080_ti:4

# number of requested nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# memory per node
#SBATCH --mem=60000

# CPU allocated
#SBATCH --cpus-per-task=4

#SBATCH --job-name=med-ssl
#SBATCH --time=1-00:10:00

#----------------------------------------------------------

MYHOME=/ubc/cs/research/shield/projects/payamfz
PROJPATH=$MYHOME/medical-ssl-segmentation

conda activate tf2-gpu
python -c "import tensorflow as tf; print('GPU LIST:', tf.config.list_physical_devices('GPU'))"

# python $PROJPATH/mycode/neural_nets/vgg16_mgpu.py \
#     --ouput_name=vgg16_job --gpu_mem_limit=10240 \
#     --learning_rate=1e-3 --epochs=10 \
#     --image_size=448 --batch_size=128 --train_size=50000 \
#     --transfer_learning=False

python $PROJPATH/mycode/neural_nets/vgg16_mgpu.py \
    --ouput_name=vgg16_test --gpu_mem_limit=10240 \
    --learning_rate=1e-3 --epochs=1 \
    --image_size=448 --batch_size=16 --train_size=512 \
    --transfer_learning=False
