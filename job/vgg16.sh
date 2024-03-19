#!/bin/bash
# a file for job output, you can check job progress
#SBATCH --output=../out/job_logs/vgg16_test%j.out

# a file for errors
#SBATCH --error=../out/job_logs/vgg16_test%j.err

# select the node edith
#SBATCH --partition=edith
#SBATCH --nodelist="jarvis2"

# use GPU
##SBATCH --gpus=geforce:4
#SBATCH --gpus=nvidia_geforce_gtx_1080_ti:1

# number of requested nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# memory per node
#SBATCH --mem=28000

# CPU allocated
#SBATCH --cpus-per-task=6

#SBATCH --job-name=med-ssl
#SBATCH --time=1-00:10:00

#----------------------------------------------------------

MYHOME=/ubc/cs/research/shield/projects/payamfz
PROJPATH=$MYHOME/medical-ssl-segmentation

conda activate tf2-gpu
python -c "import tensorflow as tf; print('GPU LIST:', tf.config.list_physical_devices('GPU'))"

# python $PROJPATH/mycode/neural_nets/vgg16.py \
#     --ouput_name=vgg16-trans-warmup \
#     --learning_rate=1e-3 --image_size=448 --epochs=2 --batch_size=64 --train_size=65536 \
#     --mode=train_then_eval --transfer_learning=True


python $PROJPATH/mycode/neural_nets/vgg16_test.py \
    --ouput_name=vgg16-miniBatchTest \
    --learning_rate=1e-3 --image_size=448 --epochs=200 --batch_size=16 --train_size=32 \
    --mode=train_then_eval --transfer_learning=True