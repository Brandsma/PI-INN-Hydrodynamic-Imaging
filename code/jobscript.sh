#!/bin/bash
#SBATCH --time=04:59:59
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=8000
#SBATCH --mail-type=BEGIN,END,TIME_LIMIT
#SBATCH --mail-user=a.brandsma.6@student.rug.nl
 
module purge
module load Python/3.9.6-GCCcore-11.2.0
module load TensorFlow/2.5.0-fosscuda-2020b
 
pip install tqdm matplotlib datetime pydot
 
echo "Starting data creation..."
python ./1-data_creation/simulation/simulation.py

echo "Starting data combination..."
python ./1-data_creation/simulation/combine_data.py

echo "Starting LSTM training..."
python ./LSTM/main.py
