#!/bin/bash
# a file for job output, you can check job progress
#SBATCH --output=../out/job_logs/env%j.out

# a file for errors
#SBATCH --error=../out/job_logs/env%j.err

# select the node edith
#SBATCH --partition=edith
#SBATCH --nodelist="edith4"

# use GPU
#SBATCH --gpus=geforce:1
##SBATCH --gpus=nvidia_geforce_gtx_1080_ti:4

# number of requested nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# memory per node
#SBATCH --mem=4096

# CPU allocated
#SBATCH --cpus-per-task=4

#SBATCH --job-name=env-test
#SBATCH --time=15:00:00

#----------------------------------------------------------

conda create --name tf2-gpu-hf --clone tf2-gpu

# conda activate tf2-gpu
# conda activate gputest
conda activate tf2-gpu-hf
pip install transformers


nvidia-smi
nvcc --version
python -c "import tensorflow as tf; print('GPU LIST:', tf.config.list_physical_devices('GPU'))"
# python -c "import tensorflow as tf; print('available:', tf.test.is_gpu_available())"

# echo "which nvcc:"
# which nvcc
# echo "locate nvcc:"
# locate nvcc


# echo "nvidia-cuda-toolkit:"
# ls -la /usr/lib/nvidia-cuda-toolkit

# echo "nvidia-cuda-toolkit/*:"
# ls -la /usr/lib/nvidia-cuda-toolkit/*

# echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
# echo "CUDA_DIR: $CUDA_DIR"


# export LD_LIBRARY_PATH=/usr/lib/nvidia-cuda-toolkit/libdevice:$LD_LIBRARY_PATH
# export PATH=/usr/lib/nvidia-cuda-toolkit/bin:$PATH

# export CUDA_DIR=/usr/bin/nvcc
# echo $CUDA_DIR
# ls -la ${CUDA_DIR}/nvvm/libdevice

# cp -r /usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc /ubc/cs/research/shield/projects/payamfz/medical-ssl-segmentation/

# python $PROJPATH/finetuning2.py --dataset=MIMIC-CXR \
#   --base_model_path=./base-models/simclr/r152_2x_sk1/hub/ \
#   --epochs=1 --batch_size=2 --learning_rate=1.0
