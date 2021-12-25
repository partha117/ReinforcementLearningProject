import random
import pickle
from tqdm import tqdm
import  json
import argparse
import torch
from AC import PolicyModel, to_one_hot
from Evaluate_Random import calculate_top_k
from AC_EntropyV2 import PolicyModel as NewPolicyModel
import os
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
from Buffer import get_replay_buffer, get_priority_replay_buffer
import numpy as np
from Environment import LTREnvV4
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path
from stable_baselines3.common.buffers import ReplayBuffer
from Buffer import CustomBuffer
def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    return np.mean([average_precision(r) for r in rs])
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
    env = LTREnvV4(data_path=file_path + test_data_path, model_path="microsoft/codebert-base",
                   tokenizer_path="microsoft/codebert-base", action_space_dim=31, report_count=None, code_max_len=2048,report_max_len=512,
                   use_gpu=False, caching=True, file_path=file_path, project_list=[project_name], test_env=True, window_size=500)

    model = NewPolicyModel(env=env)
    state_dict = torch.load(file_path + model_path, map_location="cuda:0")
    model.load_state_dict(state_dict=state_dict)
    model = model.to(dev)
    all_rr = []
    counts = None
    precision_array = []
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
            action = int(action[0].detach().cpu().numpy())
            prev_obs, reward, done, info, rr = env.step(action, return_rr=True)
            picked.append(action)
            if all_rr[-1] < rr:
                all_rr[-1] = rr
            counts = calculate_top_k(source=env.picked, target=env.match_id, counts=counts)

        ranked_array = np.zeros(len(env.picked))
        ranked_array[np.where(np.isin(np.array(env.picked), np.array(env.match_id)))] = 1
        print(ranked_array)
        print(env.picked)
        print(env.match_id)
        precision_array.append(average_precision(ranked_array))
        print(precision_array[-1])
    all_rr = np.array(all_rr)
    all_rr = all_rr[all_rr > 0]
    mean_rr = all_rr.mean()
    actual_rank = 1.0/all_rr
    map_value = np.array(precision_array).mean()



    Path(result_path).mkdir(exist_ok=True,parents=True)
    json.dump({"mrr": mean_rr,"map": map_value}, open(result_path + "_mrr.json", "w"))
    np.save(result_path + "_ranks.npy", actual_rank)
    plt.figure(figsize=(500, 500))
    plt.hist(1.0/all_rr, bins=30)
    plt.savefig(result_path + "_histogram.eps", format='eps', dpi=50)
    plt.figure(figsize=(500, 500))
    plt.boxplot(1.0/all_rr)
    plt.savefig(result_path + "_boxplot.eps", format='eps', dpi=50)