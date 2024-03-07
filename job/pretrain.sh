#!/bin/bash
# a file for job output, you can check job progress
#SBATCH --output=../out/logs/pretrain%j.out

# a file for errors
#SBATCH --error=../out/logs/pretrain%j.err

# select the node edith
#SBATCH --partition=edith
#SBATCH --nodelist="jarvis5"

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

# ORIGINAL
# python $PROJPATH/run2.py --train_mode=pretrain \
#   --train_batch_size=512 --train_epochs=1000 \
#   --learning_rate=1.0 --weight_decay=1e-4 --temperature=0.5 \
#   --dataset=cifar10 --image_size=32 --eval_split=test --resnet_depth=18 \
#   --use_blur=False --color_jitter_strength=0.5 \
#   --model_dir=/tmp/simclr_test --use_tpu=False

# PRETRAIN
# python $PROJPATH/router.py pretrain --mode=train_then_eval --train_mode=pretrain  \
#   --train_batch_size=32 --train_epochs=10 --use_tpu=False\
#   --learning_rate=1.0 --weight_decay=1e-4 --temperature=0.5 \
#   --dataset=mimic_cxr --eval_split=test --train_split=train \
#   --resnet_depth=50 --width_multiplier=3 --use_bit=False --image_size=224 \
#   --model_dir=./out/models --model_name=simclr_50x2_pretrain

# BASED ON PAPER but lower epochs (100 instead of 1000) and batch size 256 instead of 1024
python ../run.py --mode=train_then_eval --train_mode=pretrain --train_steps=210000 \
  --train_batch_size=64 --train_epochs=100 --use_tpu=False \
  --learning_rate=0.3 --weight_decay=0.0004 --temperature=0.1 --rotation_range=15.0 \
  --dataset=mimic_cxr --eval_split=test --train_split=train \
  --resnet_depth=50 --width_multiplier=1 --use_bit=False --image_size=128 \
  --model_dir=./out/models --model_name=simclr_50x2_pretrain

# EVALUATE
# python $PROJPATH/pretrain/run.py --mode=eval --train_mode=pretrain  \
#   --train_batch_size=32 --train_epochs=10 --use_tpu=False\
#   --learning_rate=1.0 --weight_decay=1e-4 --temperature=0.5 \
#   --dataset=mimic_cxr --eval_split=test --train_split=train \
#   --resnet_depth=50 --width_multiplier=1 --use_bit=False --image_size=224 \
#   --model_dir=./out/models/remedis/pretrain_02-15_05-07

# FINETUNE
# BASED ON PAPER
# python ../run.py --mode=train_then_eval --train_mode=finetune \
#   --fine_tune_after_block=4 --zero_init_logits_layer=True --rotation_range=15.0 \
#   --variable_schema='(?!global_step|(?:.*/|^)Momentum|head)' \
#   --global_bn=False --optimizer=adam --learning_rate=1e-2 --weight_decay=1e-6 --train_steps=250000 \
#   --train_epochs=10 --train_batch_size=64 --warmup_epochs=0  --use_tpu=False \
#   --dataset=mimic_cxr --image_size=224 --eval_split=test --resnet_depth=50 --width_multiplier=1  \
#   --checkpoint=./out/models/simclr_50x2_pretrain_2024-02-21_0135 --model_dir=./out/models --model_name=simclr_50x1_finetune
