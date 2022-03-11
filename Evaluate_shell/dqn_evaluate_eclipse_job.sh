#!/bin/bash
#SBATCH --account=def-m2nagapp
#SBATCH --ntasks=1               # number of MPI processes
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G      # memory; default unit is megabytes
#SBATCH --time=3:20:00           # time (DD-HH:MM)
#SBATCH --gres=gpu:v100l:1
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=p9chakra@uwtaerloo.ca
#SBATCH --mail-type=ALL
export CUDA_LAUNCH_BLOCKING=1; python /home/partha9/CS885-RProject/Evaluate_DQNV2.py \
--file_path /project/def-m2nagapp/partha9/LTR/ \
--model_path Models/DQN/Eclipse_Platform_UI/Eclipse_Platform_UI_DQN_policy_model_199.0.pt \
--result_path /project/def-m2nagapp/partha9/LTR/Results/DQN/Eclipse_Platform_UI/ \
--test_data_path Data/TestData/Eclipse_Platform_UI_test.csv \
--project_name Eclipse_Platform_UI
