#!/bin/bash
#SBATCH --account=def-m2nagapp
#SBATCH --ntasks=1               # number of MPI processes
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G      # memory; default unit is megabytes
#SBATCH --time=60:20:00           # time (DD-HH:MM)
#SBATCH --gres=gpu:v100l:2
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=p9chakra@uwtaerloo.ca
#SBATCH --mail-type=ALL
export CUDA_LAUNCH_BLOCKING=1; python /home/partha9/CS885-RProject/AC_EntropyV2.py \
--file_path /project/def-m2nagapp/partha9/LTR/ \
--cache_path /scratch/partha9/.buffer_cache_ac_aspectj \
--train_data_path Data/TrainData/Bench_BLDS_Aspectj_Dataset.csv \
--prev_policy_model_path /project/def-m2nagapp/partha9/LTR/Models/AC/Entropy/AspectJ/AspectJ_AC_Entropy_V2_policy_model_16.0.pt \
--prev_value_model_path /project/def-m2nagapp/partha9/LTR/Models/AC/Entropy/AspectJ/AspectJ_AC_Entropy_V2_value_model_16.0.pt \
--save_path /project/def-m2nagapp/partha9/LTR/Models/AC/Entropy/AspectJ/ \
--start_from 930 \
--project_name AspectJ
