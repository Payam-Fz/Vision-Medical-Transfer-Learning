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
#SBATCH --mem=64200

# CPU allocated
#SBATCH --cpus-per-task=4

#SBATCH --job-name=med-ssl
#SBATCH --time=8:00:00

#----------------------------------------------------------

MYHOME=/ubc/cs/research/shield/projects/payamfz
PROJPATH=$MYHOME/medical-ssl-segmentation

conda activate tf2-gpu
python -c "import tensorflow as tf; print('GPU LIST:', tf.config.list_physical_devices('GPU'))"

python $PROJPATH/mycode/neural_nets/vgg16.py \
    --ouput_name=vgg16 \
    --image_size=448 --epochs=10 --batch_size=32