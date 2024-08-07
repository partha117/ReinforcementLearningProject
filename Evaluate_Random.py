import random
import pickle
from tqdm import tqdm
import torch
from AC import PolicyModel, to_one_hot
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
class RandomModel(object):
    def __init__(self, env):
        self.env = env
    def forward(self, prev_actions):

        prev_actions = prev_actions.cpu().numpy()[0]
        max_indices = np.argwhere(prev_actions == np.max(prev_actions)).reshape(-1)
        choice = np.random.choice(len(max_indices), 1, replace=False)[0]
        return max_indices[choice]

    def __call__(self, *args, **kwargs):
        return self.forward(**kwargs)

def calculate_top_k(source, target, k=32, counts=None):
    if counts is None:
        counts = np.zeros(k)
    counts[len(source)] += int(any([item in source for item in target]))
    return counts
if __name__ == "__main__":
    file_path = "" #"/project/def-m2nagapp/partha9/LTR/"
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Using {}".format(dev))
    env = LTREnvV2(data_path=file_path + "Data/TestData/AspectJ_test.csv", model_path="microsoft/codebert-base",
                   tokenizer_path="microsoft/codebert-base", action_space_dim=31, report_count=92, max_len=512,
                   use_gpu=False, caching=True, file_path=file_path, project_list=['AspectJ'], test_env=True)

    model = RandomModel(env=env)
    all_rr = []
    counts = None
    for _ in tqdm(range(env.suppoerted_len)):
        all_rr.append(-100)
        done = False
        picked = []
        prev_obs = env.reset()
        while not done:
            prev_actions = to_one_hot(picked, max_size=env.action_space.n)
            prev_actions = torch.from_numpy(prev_actions).to(dev).type(torch.float)
            prev_obs = torch.from_numpy(np.expand_dims(prev_obs, axis=0)).float().to(dev)
            with torch.no_grad():
                action = model(prev_actions=prev_actions)
            prev_obs, reward, done, info, rr = env.step(action, return_rr=True)
            picked.append(action)
            if all_rr[-1] < rr:
                all_rr[-1] = rr
            counts = calculate_top_k(source=env.picked, target=env.match_id, counts=counts)
        env.picked.append(action)
        counts = calculate_top_k(source=env.picked, target=env.match_id, counts=counts)
    all_rr = np.array(all_rr)
    all_rr = all_rr[all_rr > 0]
    print(all_rr.mean(), len(all_rr))
    print((counts/env.suppoerted_len)*100)
    # print(all_rr)
    # print(1.0/all_rr)
    plt.hist(1.0/all_rr, bins=30)
    plt.show()
    plt.boxplot(1.0/all_rr)
    plt.show()