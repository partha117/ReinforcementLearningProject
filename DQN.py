import random
import pickle
import psutil
import torch
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


class DoubleDQN(nn.Module):
    def __init__(self, env):
        super(DoubleDQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.lstm_hidden_space = 256

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(env.observation_space.shape[1])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(env.observation_space.shape[0])))
        linear_input_size = convw * convh * 32 + env.action_space.n
        # self.lin_layer1 = nn.Linear(linear_input_size, 128)
        self.lstm = nn.LSTM(input_size=linear_input_size, hidden_size=self.lstm_hidden_space, batch_first=True)
        self.lin_layer2 = nn.Linear(self.lstm_hidden_space, env.action_space.n)

    def forward(self, x, actions, hidden=None):
        x = x.unsqueeze(1) if x.dim() == 3 else x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        # x = F.relu(self.lin_layer1(x))
        x = torch.concat([x.unsqueeze(1), actions.unsqueeze(1) if actions.dim() != 3 else actions], axis=2)
        x, (new_h, new_c) = self.lstm(x, (hidden[0], hidden[1]))
        # x = x.squeeze(0)
        x = self.lin_layer2(x)
        return x, [new_h, new_c]


def run_one_iter(q_net, target_net, state, action, reward, next_state, done, optim, gamma, hiddens=None, picked=None):
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    output, _ = q_net(state, actions=picked, hidden=hiddens)
    current_Q_values = output.squeeze(1).gather(1, action)
    next_Q_values, _ = target_net(next_state.type(torch.float), actions=picked, hidden=hiddens)
    next_Q_values = next_Q_values.detach().squeeze(1)
    next_max_Q_value = torch.max(next_Q_values, dim=1).values
    next_max_Q_value = next_max_Q_value.view(next_Q_values.shape[0], 1)
    target_Q_value = reward + gamma * next_max_Q_value * (1 - done.int())
    loss = torch.nn.MSELoss()(target_Q_value.type(torch.float32), current_Q_values.type(torch.float32))
    loss = torch.sqrt(loss)

    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss


def train_dqn(buffer, env, total_time_step=10000, sample_size=30, learning_rate=0.01, update_frequency=500, tau=0.3):
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    q_network = DoubleDQN(env=env).to(dev)
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    target_q_network = DoubleDQN(env=env).to(dev)
    loss_accumulator = []
    for iter_no in tqdm(range(total_time_step)):
        samples = buffer.sample(sample_size)
        state, action, reward, next_state, done = samples.observations, samples.actions, samples.rewards, samples.next_observations, samples.dones
        loss = run_one_iter(q_net=q_network, target_net=target_q_network, state=state.to(dev), action=action.to(dev),
                            reward=reward.to(dev),
                            next_state=next_state.to(dev), done=done.to(dev), optim=optimizer, gamma=0.9)
        loss_accumulator.append(loss.detach().cpu().numpy())
        if iter_no % update_frequency == 0:
            for target_param, local_param in zip(target_q_network.parameters(),
                                                 q_network.parameters()):
                target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    return q_network


def to_one_hot(array, max_size):
    temp = np.ones(max_size)
    temp[array] = 0
    return np.expand_dims(temp, axis=0)


def train_dqn_epsilon(buffer, env, total_time_step=10000, sample_size=30, learning_rate=0.01, update_frequency=300,
                      tau=0.03, file_path="", save_frequency=30):
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    q_network = DoubleDQN(env=env).to(dev)
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    target_q_network = DoubleDQN(env=env).to(dev)
    episode_len_array = []
    episode_reward = []
    pbar = tqdm(range(total_time_step))
    for e in pbar:
        done = False
        prev_obs = env.reset()
        hidden = [torch.zeros([1, 1, q_network.lstm_hidden_space]).to(dev),
                  torch.zeros([1, 1, q_network.lstm_hidden_space]).to(dev)]
        picked = []
        reward_array = []
        # pbar.set_description("Avg. reward {} Avg. episode {} Mem: {}".format(np.array(episode_reward).mean(),
        #                                                                      np.array(episode_len_array).mean(),
        #                                                                      psutil.Process(
        #                                                                          os.getpid()).memory_info().rss / 1024 ** 2))
        pbar.set_description("Avg. reward {} Avg. episode {}".format(np.array(episode_reward).mean(),
                                                                             np.array(episode_len_array).mean()))
        episode_len = 0
        while not done:
            episode_len += 1
            prev_obs = torch.Tensor(prev_obs).to(dev)
            prev_obs = prev_obs.unsqueeze(0)
            with torch.no_grad():
                action, temp_hidden = q_network(prev_obs,
                                           actions=torch.from_numpy(to_one_hot(picked, max_size=env.action_space.n)).to(
                                               dev).type(torch.float), hidden=hidden)
            if np.random.rand() <= np.max([0.05, 1.0 / np.log(e)]):
                available = [item for item in range(env.action_space.n) if item not in picked]
                action = random.sample(available, 1)[0]
                picked.append(action)
            else:
                action = action.cpu()
                action[0][
                    ~torch.from_numpy(to_one_hot(picked, max_size=env.action_space.n)).type(torch.bool)] = torch.min(
                    action) - 3
                max_action = torch.argmax(action)
                action = int(max_action.detach().numpy())
                picked.append(action)
            obs, reward, done, info = env.step(action)
            reward_array.append(reward)
            info['hidden'] = [item.detach().cpu().numpy() for item in hidden]
            info['picked'] = picked
            hidden = temp_hidden
            buffer.add(prev_obs.detach().cpu().numpy(), obs, np.array([action]), np.array([reward]), np.array([done]),
                       [info])
            prev_obs = obs
            # print("Episode length: {}".format(episode_len))
        if len(buffer) > 200:
            samples = buffer.sample(sample_size)
            state, action, reward, next_state, batch_done, info = samples  # samples.observations, samples.actions, samples.rewards, samples.next_observations, samples.dones, samples.info
            batch_hidden = torch.tensor(np.array(
                [np.stack([np.array(item['hidden'][0]) for item in info], axis=2)[0],
                 np.stack([np.array(item['hidden'][1]) for item in info], axis=2)[0]])).to(dev)
            batch_picked = torch.tensor(
                [to_one_hot(item['picked'], max_size=env.action_space.n) for item in info]).to(dev).type(
                torch.float)
            run_one_iter(q_net=q_network, target_net=target_q_network, state=state.to(dev),
                                action=action.to(dev), reward=reward.to(dev),
                                next_state=next_state.to(dev), done=batch_done.to(dev), optim=optimizer, gamma=0.9,
                                hiddens=batch_hidden, picked=batch_picked)
            if e % update_frequency == 0:
                for target_param, local_param in zip(target_q_network.parameters(),
                                                     q_network.parameters()):
                    target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
        episode_reward.append(np.array(reward_array).sum())
        episode_len_array.append(episode_len)

        if e % save_frequency == 0:
            save_num = e / save_frequency
            if os.path.isfile(file_path + "dqn_model_{}.pt".format(save_num - 1)):
               os.remove(file_path + "dqn_model_{}.pt".format(save_num - 1))
            if os.path.isfile(file_path + "DQN_Episode_Reward.pickle"):
               os.remove(file_path + "DQN_Episode_Reward.pickle")
            if os.path.isfile(file_path + "DQN_Episode_Length.pickle"):
               os.remove(file_path + "DQN_Episode_Length.pickle")
            torch.save(q_network.state_dict(), file_path + "dqn_model_{}.pt".format(save_num))
            with open(file_path + "DQN_Episode_Reward.pickle", "wb") as f:
                pickle.dump(episode_reward, f)

            with open(file_path + "DQN_Episode_Length.pickle", "wb") as f:
                pickle.dump(episode_len_array, f)
    return q_network


if __name__ == "__main__":
    file_path = "/project/def-m2nagapp/partha9/LTR/"
    cache_path = "/scratch/partha9/.buffer_cache_dqn"
    Path(file_path).mkdir(parents=True, exist_ok=True)
    env = LTREnvV2(data_path=file_path + "Data/TrainData/Bench_BLDS_Dataset.csv", model_path="microsoft/codebert-base",
                   tokenizer_path="microsoft/codebert-base", action_space_dim=31, report_count=50, max_len=512,
                   use_gpu=False, caching=True, file_path=file_path)
    obs = env.reset()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    buffer = CustomBuffer(6000, cache_path=cache_path)
    model = train_dqn_epsilon(buffer=buffer, sample_size=128, env=env, total_time_step=6000, update_frequency=300,
                              tau=0.01, file_path=file_path)