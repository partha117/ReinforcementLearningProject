#!/bin/bash
#SBATCH --account=def-m2nagapp
#SBATCH --ntasks=1               # number of MPI processes
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G      # memory; default unit is megabytes
#SBATCH --time=48:20:00           # time (DD-HH:MM)
#SBATCH --gres=gpu:v100l:1
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=p9chakra@uwtaerloo.ca
#SBATCH --mail-type=ALL
export CUDA_LAUNCH_BLOCKING=1; python /home/partha9/CS885-RProject/DQNV2.py \
--file_path /project/def-m2nagapp/partha9/LTR/ \
--cache_path /scratch/partha9/.buffer_cache_dqn_eclipse \
--train_data_path Data/TrainData/Bench_BLDS_Eclipse_Platform_UI_Dataset.csv \
--save_path /project/def-m2nagapp/partha9/LTR/Models/DQN/Eclipse_Platform_UI/ \
--start_from 0 \
--project_name Eclipse_Platform_UI
