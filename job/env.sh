#!/bin/bash
# a file for job output, you can check job progress
#SBATCH --output=out/logs/env.out


# a file for errors
#SBATCH --error=out/logs/env.err


# select the node edith
#SBATCH --partition=edith
#SBATCH --nodelist="edith1"


# use one 2080Ti GPU
#SBATCH --gpus=geforce:0


# number of requested nodes
#SBATCH --nodes=1

# memory per node
#SBATCH --mem=2000

# CPU allocated
#SBATCH --cpus-per-task=1

#SBATCH --job-name=env
#SBATCH --time=20:00

# module add Python/3.8.6-GCCcore-10.2.0

# RUNPATH=/ubc/cs/research/shield/projects/payamfz
# cd $RUNPATH

#echo "PATH: $PATH"
#echo "PYTHONPATH: $PYTHONPATH"

# Check if VIRTUAL_ENV is set
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Python virtual environment is activated: $VIRTUAL_ENV"
else
    echo "Python virtual environment is not activated."
fi
#echo "Before activation: $(which python)"
source /ubc/cs/research/shield/projects/payamfz/tensorenv/bin/activate
# echo "After activation: $(which python)"

# Check if VIRTUAL_ENV is set
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Python virtual environment is activated: $VIRTUAL_ENV"
else
    echo "Python virtual environment is not activated."
fi

echo "PATH: $PATH"
echo "PYTHONPATH: $PYTHONPATH"

# export PATH=/ubc/cs/research/shield/projects/payamfz/env/bin:$PATH
# echo "After PATH modification: $PATH"
export PYTHONPATH=/ubc/cs/research/shield/projects/payamfz/tensorenv/lib/python3.6/site-packages:$PYTHONPATH
# echo "After PATH modification: $PATH"
# echo "After PYTHONPATH modification: $PYTHONPATH"

# pip install -r ../cleanvenv.txt
pip show numpy

pip install --upgrade 'numpy~=1.19.5'


pip show numpy
# pip install --upgrade tensorflow==2.5


echo "which python: $(which python)"
echo "which python3: $(which python3)"
echo "which pip: $(which pip)"
echo "pip list"
pip list

echo "pip show tensorflow_addons" 
pip show tensorflow_addons

deactivate

