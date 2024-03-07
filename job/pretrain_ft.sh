#!/bin/bash
# a file for job output, you can check job progress
#SBATCH --output=../out/logs/pretrain-ft%j.out

# a file for errors
#SBATCH --error=../out/logs/pretrain-ft%j.err

# select the node edith
#SBATCH --partition=edith
#SBATCH --nodelist="jarvis4"

# use GPU
##SBATCH --gpus=geforce:4
#SBATCH --gpus=nvidia_geforce_gtx_1080_ti:4

# number of requested nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# memory per node
#SBATCH --mem=58000

# CPU allocated
#SBATCH --cpus-per-task=4

#SBATCH --job-name=pret
#SBATCH --time=8:00:00

#----------------------------------------------------------

MYHOME=/ubc/cs/research/shield/projects/payamfz
PROJPATH=$MYHOME/medical-ssl-segmentation

conda activate tf2-gpu-simclr

python -c "import tensorflow as tf; print('GPU LIST:', tf.config.list_physical_devices('GPU'))"


# FINETUNE
# BASED ON PAPER
python ../run.py --mode=train_then_eval --train_mode=finetune \
  --fine_tune_after_block=4 --zero_init_logits_layer=True --rotation_range=15.0 \
  --variable_schema='(?!global_step|(?:.*/|^)Momentum|head)' \
  --global_bn=False --optimizer=adam --learning_rate=1e-2 --weight_decay=1e-6 --train_steps=2500 \
  --train_epochs=10 --train_batch_size=64 --warmup_epochs=0  --use_tpu=False \
  --dataset=mimic_cxr --image_size=128 --eval_split=test --resnet_depth=50 --width_multiplier=1  \
  --checkpoint=./out/models/simclr_50x2_pretrain_2024-02-21_0227 --model_dir=./out/models --model_name=simclr_50x1_finetune
