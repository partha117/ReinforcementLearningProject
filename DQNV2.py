import random
import pickle
import argparse
import psutil
import torch
import os
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
from torch.nn.functional import softmax
from AC_EntropyV2 import TwoDConv, TwoDConvReport, update_learning_rate
from Buffer import get_replay_buffer, get_priority_replay_buffer
import numpy as np
from Environment import LTREnvV2, LTREnvV4, LTREnvV5
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
from pathlib import Path
from stable_baselines3.common.buffers import ReplayBuffer
from Buffer import CustomBuffer
class AttLayer(nn.Module):
    def __init__(self, limit=768):
        super(AttLayer, self).__init__()
        self.limit = limit
        self.report_attention_layer = nn.Linear(in_features=self.limit, out_features=self.limit)
        self.report_linear_layer = nn.Linear(in_features=self.limit, out_features=self.limit)
        self.code_linear_layer = nn.Linear(in_features=self.limit, out_features=self.limit)


    def forward(self, x):
         # print(x.shape)
        report = x[:, 0, :self.limit]
        source = x[:, :, self.limit:]
        report_ll = self.report_linear_layer(report)
         # print(report_ll.shape)
        code_ll = self.code_linear_layer(source)
         # print(code_ll.shape)
        multiplication = torch.bmm(torch.unsqueeze(report_ll, 1), code_ll.swapaxes(1, 2))
         # print(multiplication.shape)
        att_value = softmax(multiplication, dim=2)
        att_value = att_value.squeeze(1).unsqueeze(2)
        #r_att_value = self.report_attention_layer(report)
        return att_value * source,  report

class DoubleDQN(nn.Module):
    def __init__(self, env, limit=768, training=True):
        super(DoubleDQN, self).__init__()
        self.limit = limit
        self.env = env
        self.training = training
        self.attention = AttLayer(limit=self.limit)
        self.linear1 = nn.Linear(in_features=self.limit * 2, out_features=self.limit)
        self.linear2 = nn.Linear(in_features=self.limit, out_features=1)

    def forward(self, x):
        source, report = self.attention(x)
        x = torch.concat([source, torch.stack([report for i in range(source.shape[1])]).swapaxes(0, 1)], axis=2)
        x = F.relu(F.dropout(self.linear1(x), p=0.2, training=self.training))
        x = F.dropout(self.linear2(x), p=0.1, training=self.training)
        return softmax(x.squeeze(2), dim=1)


def run_one_iter(q_net, target_net, state, action, reward, next_state, done, optim, gamma, picked=None):
    output = q_net(state)
     # print("output", output.shape)
     # print("action", action.shape)
    current_Q_values = output.squeeze(1).gather(1, action)
    next_Q_values = target_net(next_state.type(torch.float))
    next_Q_values = next_Q_values.detach().squeeze(1)
    next_Q_values[~picked.squeeze(1).type(torch.bool)] = torch.min(next_Q_values) - 3
    next_max_Q_value = torch.max(next_Q_values, dim=1).values
    next_max_Q_value = next_max_Q_value.view(next_Q_values.shape[0], 1)
    target_Q_value = reward + gamma * next_max_Q_value * (1 - done.int())
    loss = torch.nn.MSELoss()(target_Q_value.type(torch.float32), current_Q_values.type(torch.float32))
    loss = torch.sqrt(loss)

    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss

def to_one_hot(array, max_size):
    temp = np.ones(max_size)
    temp[array] = 0
    return np.expand_dims(temp, axis=0)


def train_dqn_epsilon(buffer, env, total_time_step=10000, sample_size=30, learning_rate=0.01, update_frequency=300,
                      tau=0.03, file_path="", save_frequency=30, multi=False, start_from=0,lr_frequency=200):
    q_network = DoubleDQN(env=env).to(dev)
    target_q_network = DoubleDQN(env=env).to(dev)
    if prev_model_path is not None:
        state_dict = torch.load(prev_model_path)
        q_network.load_state_dict(state_dict=state_dict)
        for target_param, local_param in zip(target_q_network.parameters(),
                                             q_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    episode_len_array = []
    episode_reward = []
    pbar = trange(start_from, total_time_step)
    dqn_loss = None

    for e in pbar:
        done = False
        prev_obs = env.reset()
        picked = []
        reward_array = []
        pbar.set_description("Avg. reward {} Avg. episode {} Loss {}".format(np.array(episode_reward).mean(),
                                                                             np.array(episode_len_array).mean(),
                                                                             dqn_loss))
        episode_len = 0
        while not done:
            episode_len += 1
            prev_obs = torch.Tensor(prev_obs).to(dev)
            prev_obs = prev_obs.unsqueeze(0)
            with torch.no_grad():
                action = q_network(prev_obs)
            if np.random.rand() <= np.max([0.05, 1.0 / np.log(e)]):
                available = [item for item in range(env.action_space.n) if item not in picked]
                action = random.sample(available, 1)[0]
                picked.append(action)
            else:
                action = action.cpu()
                 # print(action)
                action[
                    ~torch.from_numpy(to_one_hot(picked, max_size=env.action_space.n)).type(torch.bool)] = torch.min(
                    action) - 3
                max_action = torch.argmax(action)
                action = int(max_action.detach().numpy())
                picked.append(action)
            obs, reward, done, info = env.step(action)
             # print("obs",obs.shape)
            reward_array.append(reward)
            info['picked'] = picked
            buffer.add(prev_obs.detach().cpu().numpy(), obs, np.array([action]), np.array([reward]), np.array([done]),
                       [info])
            prev_obs = obs
            #  # print("Episode length: {}".format(episode_len))
        if len(buffer) > 400:
            samples = buffer.sample(sample_size)
            state, action, reward, next_state, batch_done, info = samples  # samples.observations, samples.actions, samples.rewards, samples.next_observations, samples.dones, samples.info
             # print("state", state.shape)
            state = state.squeeze(1)
            batch_picked = torch.tensor(np.array(
                [to_one_hot(item['picked'], max_size=env.action_space.n) for item in info])).to(dev).type(
                torch.float)
            dqn_loss = run_one_iter(q_net=q_network, target_net=target_q_network, state=state.to(dev),
                                action=action.to(dev), reward=reward.to(dev),
                                next_state=next_state.to(dev), done=batch_done.to(dev), optim=optimizer, gamma=0.9, picked=batch_picked)
            if e % update_frequency == 0:
                for target_param, local_param in zip(target_q_network.parameters(),
                                                     q_network.parameters()):
                    target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
        episode_reward.append(np.array(reward_array).sum())
        episode_len_array.append(episode_len)

        if e % lr_frequency == 0 and e != 0:
            update_learning_rate(optimizer, 10)
        if e % save_frequency == 0:
            save_num = e / save_frequency
            if os.path.isfile(save_path + "{}_DQN_policy_model_{}.pt".format(project_name, save_num - 1)):
                os.remove(save_path + "{}_DQN_policy_model_{}.pt".format(project_name, save_num - 1))
            if os.path.isfile(save_path + "{}_DQN_value_model_{}.pt".format(project_name, save_num - 1)):
                os.remove(save_path + "{}_DQN_value_model_{}.pt".format(project_name, save_num - 1))
            if os.path.isfile(save_path + "{}_DQN_Episode_Reward.pickle".format(project_name)):
                os.remove(save_path + "{}_DQN_Episode_Reward.pickle".format(project_name))
            if os.path.isfile(save_path + "{}_DQN_Episode_Length.pickle".format(project_name)):
                os.remove(save_path + "{}_DQN_Episode_Length.pickle".format(project_name))

            torch.save(q_network.state_dict(),
                       save_path + "{}_DQN_policy_model_{}.pt".format(project_name, save_num))

            with open(save_path + "{}_DQN_Episode_Reward.pickle".format(project_name), "wb") as f:
                pickle.dump(episode_reward, f)

            with open(save_path + "{}_DQN_Episode_Length.pickle".format(project_name), "wb") as f:
                pickle.dump(episode_len_array, f)
            buffer.save_others()
    return q_network


if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', default="/project/def-m2nagapp/partha9/LTR/", help='File Path')
    parser.add_argument('--cache_path', default="/scratch/partha9/.buffer_cache_dqn", help='Cache Path')
    parser.add_argument('--prev_model_path', default=None, help='Trained model Path')
    parser.add_argument('--train_data_path', help='Training Data Path')
    parser.add_argument('--save_path', help='Save Path')
    parser.add_argument('--start_from', default=0, help='Start from')
    parser.add_argument('--project_name', help='Project Name')
    options = parser.parse_args()
    file_path = options.file_path
    cache_path = options.cache_path
    prev_model_path = options.prev_model_path
    train_data_path = options.train_data_path
    project_name = options.project_name
    save_path = options.save_path
    start_from = int(options.start_from)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    # file_path = ""  # "/project/def-m2nagapp/partha9/LTR/"
    # cache_path = ".cache"  # "/scratch/partha9/.buffer_cache_ac"
    # prev_model_path = None  # "/project/def-m2nagapp/partha9/LTR/AspectJ_New_AC_policy_model_124.0.pt"
    # train_data_path = "Data/TrainData/Bench_BLDS_Aspectj_Dataset.csv"
    # project_name = "AspectJ"
    # save_path = ""
    # start_from = 0
    # dev = "cpu"
    Path(file_path).mkdir(parents=True, exist_ok=True)
    env = LTREnvV5(data_path=file_path + train_data_path, model_path="microsoft/codebert-base",
                   tokenizer_path="microsoft/codebert-base", action_space_dim=31, report_count=None, code_max_len=2048,
                   report_max_len=512,
                   use_gpu=False, caching=True, file_path=file_path, project_list=[project_name], window_size=500)
    obs = env.reset()
    buffer = CustomBuffer(6000, cache_path=cache_path, delete=(start_from == 0), start_from=start_from * 31)
    model = train_dqn_epsilon(buffer=buffer, sample_size=16, env=env, total_time_step=6000, update_frequency=300,learning_rate=0.001,
                              tau=0.01, file_path=file_path)