#!/bin/bash
# a file for job output, you can check job progress
#SBATCH --output=../out/logs/finetune%j.out

# a file for errors
#SBATCH --error=../out/logs/finetune%j.err

# select the node edith
#SBATCH --partition=edith
#SBATCH --nodelist="jarvis4"

# use GPU
##SBATCH --gpus=geforce:4
#SBATCH --gpus=nvidia_geforce_gtx_1080_ti:2

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
# nvidia-smi
# nvcc --version
python -c "import tensorflow as tf; print('GPU LIST:', tf.config.list_physical_devices('GPU'))"
# python -c "import tensorflow as tf; print('available:', tf.test.is_gpu_available())"

# python $PROJPATH/finetuning2.py --dataset=MIMIC-CXR \
#   --base_model_path=./base-models/simclr/r152_2x_sk1/hub/ \
#   --epochs=4 --batch_size=64 --learning_rate=0.1

# python $PROJPATH/finetuning2.py --dataset=MIMIC-CXR \
#   --base_model_path=./base-models/remedis/cxr-50x1-remedis-s/ \
#   --epochs=4 --batch_size=64 --learning_rate=0.1

python $PROJPATH/finetuning_simclr.py --dataset=MIMIC-CXR \
  --epochs=1 --batch_size=64 --learning_rate=0.1
