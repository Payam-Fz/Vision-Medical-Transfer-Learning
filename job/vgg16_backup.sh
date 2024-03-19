#!/bin/bash
# a file for job output, you can check job progress
#SBATCH --output=../out/job_logs/vgg16-transfer%j.out

# a file for errors
#SBATCH --error=../out/job_logs/vgg16-transfer%j.err

# select the node edith
#SBATCH --partition=edith
#SBATCH --nodelist="jarvis2"

# use GPU
#SBATCH --gpus=nvidia_geforce_gtx_1080_ti:1

# number of requested nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# memory per node
#SBATCH --mem=28000

# CPU allocated
#SBATCH --cpus-per-task=2

#SBATCH --job-name=med-ssl
#SBATCH --time=1-00:10:00

#----------------------------------------------------------

MYHOME=/ubc/cs/research/shield/projects/payamfz
PROJPATH=$MYHOME/medical-ssl-segmentation

conda activate tf2-gpu
python -c "import tensorflow as tf; print('GPU LIST:', tf.config.list_physical_devices('GPU'))"

python $PROJPATH/mycode/neural_nets/vgg16_backup.py \
    --ouput_name=vgg16-transfer \
    --learning_rate=1e-3 --image_size=448 --epochs=10 --batch_size=32 \
    --load_checkpoint=./out_archive/vgg16-transfer/vgg16-transfer_2024-03-13_1815_epoch1-2/model/checkpoints \
    --transfer_learning=True