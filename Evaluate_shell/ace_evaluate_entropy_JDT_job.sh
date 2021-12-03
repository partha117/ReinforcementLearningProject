#!/bin/bash
#SBATCH --account=def-m2nagapp
#SBATCH --ntasks=1               # number of MPI processes
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G      # memory; default unit is megabytes
#SBATCH --time=4:20:00           # time (DD-HH:MM)
#SBATCH --gres=gpu:p100:1
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=p9chakra@uwtaerloo.ca
#SBATCH --mail-type=ALL
export CUDA_LAUNCH_BLOCKING=1; python /home/partha9/CS885-RProject/Evaluate_AC.py \
--file_path /project/def-m2nagapp/partha9/LTR/ \
--model_path Models/AC/Entropy/JDT/JDT_New_AC_Entropy_policy_model_81.0.pt \
--result_path /project/def-m2nagapp/partha9/LTR/Results/AC/Entropy/JDT/ \
--test_data_path Data/TestData/JDT_test.csv \
--project_name JDT
