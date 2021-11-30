#!/bin/bash
#SBATCH --account=def-m2nagapp
#SBATCH --ntasks=1               # number of MPI processes
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G      # memory; default unit is megabytes
#SBATCH --time=18:20:00           # time (DD-HH:MM)
#SBATCH --gres=gpu:p100:1
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=p9chakra@uwtaerloo.ca
#SBATCH --mail-type=ALL
export CUDA_LAUNCH_BLOCKING=1; python /home/partha9/CS885-RProject/AC_Entropy.py \
--file_path /project/def-m2nagapp/partha9/LTR/ \
--cache_path /scratch/partha9/.buffer_cache_ac_jdt \
--prev_policy_model_path /project/def-m2nagapp/partha9/LTR/New_AC_policy_model_107.0.pt \
--prev_value_model_path /project/def-m2nagapp/partha9/LTR/New_AC_value_model_107.0.pt \
--train_data_path Data/TrainData/Bench_BLDS_JDT_Dataset.csv \
--project_name JDT
