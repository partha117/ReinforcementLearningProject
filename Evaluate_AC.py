import random
import pickle
from tqdm import tqdm
import  json
import argparse
import torch
from AC import PolicyModel, to_one_hot
from Evaluate_Random import calculate_top_k
from NAC import PolicyModel as NewPolicyModel
import os
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
from Buffer import get_replay_buffer, get_priority_replay_buffer
import numpy as np
from Environment import LTREnvV2
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path
from stable_baselines3.common.buffers import ReplayBuffer
from Buffer import CustomBuffer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', default="/project/def-m2nagapp/partha9/LTR/", help='File Path')
    parser.add_argument('--test_data_path', help='Test Data Path')
    parser.add_argument('--project_name', help='Project Name')
    parser.add_argument('--model_path', help='Project Name')
    parser.add_argument('--result_path', help='Project Name')
    options = parser.parse_args()
    # file_path = "/project/def-m2nagapp/partha9/LTR/"
    # test_data_path = "Data/TestData/AspectJ_test.csv"
    # project_name = "AspectJ"
    file_path = options.file_path
    test_data_path = options.test_data_path
    project_name = options.project_name
    model_path = options.model_path
    result_path = options.result_path
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Using {}".format(dev))
    env = LTREnvV2(data_path=file_path + test_data_path, model_path="microsoft/codebert-base",
                   tokenizer_path="microsoft/codebert-base", action_space_dim=31, report_count=None, max_len=512,
                   use_gpu=False, caching=True, file_path=file_path, project_list=[project_name], test_env=True)

    model = NewPolicyModel(env=env)
    state_dict = torch.load(file_path + model_path)
    model.load_state_dict(state_dict=state_dict)
    model = model.to(dev)
    all_rr = []
    counts = None
    for _ in tqdm(range(env.suppoerted_len)):
        all_rr.append(-100)
        done = False
        picked = []
        hidden = [torch.zeros([1, 1, model.lstm_hidden_space]).to(dev),
                  torch.zeros([1, 1, model.lstm_hidden_space]).to(dev)]
        prev_obs = env.reset()
        while not done:
            prev_actions = to_one_hot(picked, max_size=env.action_space.n)
            prev_actions = torch.from_numpy(prev_actions).to(dev).type(torch.float)
            prev_obs = torch.from_numpy(np.expand_dims(prev_obs, axis=0)).float().to(dev)
            hidden = [item.to(dev).type(torch.float) for item in hidden]
            with torch.no_grad():
                action, hidden = model(x=prev_obs, actions=prev_actions, hidden=hidden)
            action = torch.distributions.Categorical(action).sample()
            action = int(action[0][0].detach().cpu().numpy())
            prev_obs, reward, done, info, rr = env.step(action, return_rr=True)
            picked.append(action)
            if all_rr[-1] < rr:
                all_rr[-1] = rr
            counts = calculate_top_k(source=env.picked, target=env.match_id, counts=counts)
    all_rr = np.array(all_rr)
    all_rr = all_rr[all_rr > 0]
    mean_rr = all_rr.mean()
    actual_rank = 1.0/all_rr



    Path(result_path).mkdir(exist_ok=True,parents=True)
    json.dump({"mrr": mean_rr}, open(result_path + "_mrr.json", "w"))
    np.save(result_path + "_ranks..npy", actual_rank)
    plt.figure(figsize=(500, 500))
    plt.hist(1.0/all_rr, bins=30)
    plt.savefig(result_path + "_histogram.eps", format='eps', dpi=100)
    plt.figure(figsize=(500, 500))
    plt.boxplot(1.0/all_rr)
    plt.savefig(result_path + "_boxplot.eps", format='eps', dpi=100)