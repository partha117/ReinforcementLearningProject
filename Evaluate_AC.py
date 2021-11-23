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

if __name__ == "__main__":
    file_path = "" #"/project/def-m2nagapp/partha9/LTR/"
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = LTREnvV2(data_path=file_path + "Data/TrainData/Bench_BLDS_Dataset.csv", model_path="microsoft/codebert-base",
                   tokenizer_path="microsoft/codebert-base", action_space_dim=31, report_count=50, max_len=512,
                   use_gpu=False, caching=True, file_path=file_path)

    model = NewPolicyModel(env=env)
    state_dict = torch.load("New_AC_policy_model_8.0.pt")
    model.load_state_dict(state_dict=state_dict)
    model = model.to(dev)
    all_rr = []
    for _ in tqdm(range(50)):
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
            action, hidden = model(x=prev_obs, actions=prev_actions, hidden=hidden)
            action = torch.distributions.Categorical(action).sample()
            action = int(action[0][0].detach().cpu().numpy())
            prev_obs, reward, done, info, rr = env.step(action, return_rr=True)
            picked.append(action)
            if all_rr[-1] < rr:
                all_rr[-1] = rr
    all_rr = np.array(all_rr)
    all_rr = all_rr[all_rr > 0]
    print(all_rr.mean(), len(all_rr))
    print(all_rr)