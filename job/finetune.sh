#!/bin/bash
# a file for job output, you can check job progress
#SBATCH --output=../out/logs/finetune%j.out

# a file for errors
#SBATCH --error=../out/logs/finetune%j.err

# select the node edith
#SBATCH --nodelist="edith2"
#SBATCH --partition=edith

# use one 2080Ti GPU
#SBATCH --gpus=geforce:2

# number of requested nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# memory per node
#SBATCH --mem=65536

# CPU allocated
#SBATCH --cpus-per-task=1

#SBATCH --job-name=med-ssl
#SBATCH --time=8:00:00

#----------------------------------------------------------

MYHOME=/ubc/cs/research/shield/projects/payamfz
PROJPATH=$MYHOME/medical-ssl-segmentation

conda activate gputest
# nvidia-smi
# nvcc --version
python -c "import tensorflow as tf; print('GPU LIST:', tf.config.list_physical_devices('GPU'))"
# python -c "import tensorflow as tf; print('available:', tf.test.is_gpu_available())"

# python $PROJPATH/finetuning2.py --dataset=MIMIC-CXR \
#   --base_model_path=./base-models/simclr/r152_2x_sk1/hub/ \
#   --epochs=10 --batch_size=64 --learning_rate=1.0

python $PROJPATH/finetuning2.py --dataset=MIMIC-CXR \
  --base_model_path=./base-models/simclr/r152_2x_sk1/hub/ \
  --epochs=1 --batch_size=2 --learning_rate=1.0
