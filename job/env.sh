#!/bin/bash
# a file for job output, you can check job progress
#SBATCH --output=../out/logs/env%j.out

# a file for errors
#SBATCH --error=../out/logs/env%j.err

# select the node edith
#SBATCH --partition=edith
#SBATCH --nodelist="jarvis1"

# use GPU
##SBATCH --gpus=geforce:1
#SBATCH --gpus=nvidia_geforce_gtx_1080_ti:4

# number of requested nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# memory per node
#SBATCH --mem=4096

# CPU allocated
#SBATCH --cpus-per-task=1

#SBATCH --job-name=env-test
#SBATCH --time=05:00

#----------------------------------------------------------

MYHOME=/ubc/cs/research/shield/projects/payamfz
PROJPATH=$MYHOME/medical-ssl-segmentation

conda activate tf2-gpu
# conda activate gputest
nvidia-smi
nvcc --version
python -c "import tensorflow as tf; print('GPU LIST:', tf.config.list_physical_devices('GPU'))"
python -c "import tensorflow as tf; print('available:', tf.test.is_gpu_available())"

# python $PROJPATH/finetuning2.py --dataset=MIMIC-CXR \
#   --base_model_path=./base-models/simclr/r152_2x_sk1/hub/ \
#   --epochs=1 --batch_size=2 --learning_rate=1.0
