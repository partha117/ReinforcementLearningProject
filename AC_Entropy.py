import glob
import psutil
import os
from pathlib import Path
from tqdm import tqdm
import torch
from torch import nn
from Buffer import CustomBuffer
from DQN import train_dqn_epsilon, to_one_hot
from Environment import LTREnvV2
import torch.nn.functional as F
import numpy as np
import pickle


class ValueModel(nn.Module):
    def __init__(self, env):
        super(ValueModel, self).__init__()
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
        self.lstm = nn.LSTM(input_size=linear_input_size, hidden_size=self.lstm_hidden_space, batch_first=True)
        self.lin_layer2 = nn.Linear(self.lstm_hidden_space, 1)

    def forward(self, x, actions, hidden=None):
        x = x.unsqueeze(1) if x.dim() == 3 else x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.concat([x.unsqueeze(1), actions.unsqueeze(1) if actions.dim() != 3 else actions], axis=2)
        x, (new_h, new_c) = self.lstm(x, (hidden[0], hidden[1]))
        x = self.lin_layer2(x)
        return x, [new_h, new_c]


class PolicyModel(nn.Module):
    def __init__(self, env):
        super(PolicyModel, self).__init__()
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
        x = self.lin_layer2(x)
        # return torch.softmax((x * actions), dim=-1), [new_h, new_c]
        x = torch.softmax(x, dim=-1) * actions
        x = x / x.sum()
        return x, [new_h, new_c]


def a2c_step(policy_net, optimizer_policy, optimizer_value, states, advantages, batch_picked, batch_hidden, lambda_val=1):
    """update critic"""
    value_loss = advantages.pow(2).mean()
    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()

    """update policy"""
    probs, _ = policy_net(states, actions=batch_picked, hidden=batch_hidden)
    dist = torch.distributions.Categorical(probs=probs)
    action = dist.sample()
    policy_loss = -dist.log_prob(action) * advantages.detach() + lambda_val * dist.entropy()
    policy_loss = policy_loss.mean()
    optimizer_policy.zero_grad()
    policy_loss.backward()
    optimizer_policy.step()


def to_device(device, *args):
    return [x.to(device) for x in args]


def estimate_advantages(rewards, done, states, next_states, gamma, device, value_model, batch_hidden_value, batch_picked):
    rewards, masks, states, next_states = rewards.to(device), done.to(device).type(torch.float), states.to(device).type(
        torch.float), next_states.to(device).type(torch.float)
    advantages = rewards + (1.0 - masks) * gamma * value_model(next_states, batch_picked, batch_hidden_value)[0].detach() - value_model(states, batch_picked, batch_hidden_value)[0]
    return advantages


def update_params(samples, value_net, policy_net, policy_optimizer, value_optimizer, gamma, device):
    state, action, reward, next_state, done, info = samples
    batch_hidden = torch.tensor(np.array(
        [np.stack([np.array(item['hidden'][0]) for item in info], axis=2)[0],
         np.stack([np.array(item['hidden'][1]) for item in info], axis=2)[0]])).to(device)
    batch_hidden_value = torch.tensor(np.array(
        [np.stack([np.array(item['hidden_value'][0]) for item in info], axis=2)[0],
         np.stack([np.array(item['hidden_value'][1]) for item in info], axis=2)[0]])).to(device)
    batch_picked = torch.tensor(np.array(
        [to_one_hot(item['picked'], max_size=env.action_space.n) for item in info])).to(device).type(
        torch.float)

    """get advantage estimation from the trajectories"""
    advantages = estimate_advantages(reward, done, state, next_state, gamma, device, value_net, batch_hidden_value, batch_picked)

    """perform TRPO update"""
    a2c_step(policy_net, policy_optimizer, value_optimizer, state.type(torch.float).to(device), advantages,
             batch_picked, batch_hidden)


def train_actor_critic(total_time_step, sample_size, save_frequency=30):
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    policy_model = PolicyModel(env=env)
    value_model = ValueModel(env=env)
    if prev_policy_model_path is not None:
        state_dict = torch.load(prev_policy_model_path)
        policy_model.load_state_dict(state_dict=state_dict)
    if prev_value_model_path is not None:
        state_dict = torch.load(prev_value_model_path)
        value_model.load_state_dict(state_dict=state_dict)
    policy_model = policy_model.to(dev)
    value_model = value_model.to(dev)
    optimizer_policy = torch.optim.Adam(policy_model.parameters(), lr=0.01)
    optimizer_value = torch.optim.Adam(value_model.parameters(), lr=0.01)
    pbar = tqdm(range(total_time_step))
    episode_len_array = []
    episode_reward = []
    for e in pbar:
        done = False
        prev_obs = env.reset()
        hidden = [torch.zeros([1, 1, policy_model.lstm_hidden_space]).to(dev),
                  torch.zeros([1, 1, policy_model.lstm_hidden_space]).to(dev)]
        hidden_value = [torch.zeros([1, 1, value_model.lstm_hidden_space]).to(dev),
                  torch.zeros([1, 1, value_model.lstm_hidden_space]).to(dev)]
        picked = []
        reward_array = []
        # pbar.set_description("Avg. reward {} Avg. episode {} Mem: {}".format(np.array(episode_reward).mean(),
        #                                                              np.array(episode_len_array).mean(),
        #                                                                      psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2))
        pbar.set_description("Avg. reward {} Avg. episode {}".format(np.array(episode_reward).mean(),
                                                                             np.array(episode_len_array).mean()))
        episode_len = 0
        while not done:
            episode_len += 1
            prev_obs = torch.Tensor(prev_obs).to(dev)
            prev_obs = prev_obs.unsqueeze(0)
            temp_action = torch.from_numpy(to_one_hot(picked, max_size=env.action_space.n)).to(
                dev).type(torch.float)
            with torch.no_grad():
                action, temp_hidden = policy_model(prev_obs, actions=temp_action, hidden=hidden)
                _, temp_hidden_value = policy_model(prev_obs, actions=temp_action, hidden=hidden_value)
            action = torch.distributions.Categorical(action).sample()
            action = int(action[0][0].cpu().numpy())
            picked.append(action)
            obs, reward, done, info = env.step(action)
            reward_array.append(reward)
            info['hidden'] = [item.cpu().numpy() for item in hidden]
            info['picked'] = picked
            info['hidden'] = [item.cpu().numpy() for item in hidden]
            info['hidden_value'] = [item.cpu().numpy() for item in hidden_value]
            hidden = temp_hidden
            hidden_value = temp_hidden_value
            buffer.add(prev_obs.cpu().numpy(), obs, np.array([action]), np.array([reward]), np.array([done]),
                       [info])
            prev_obs = obs
        if len(buffer) > 50:
            samples = buffer.sample(sample_size)
            update_params(samples=samples, value_net=value_model, policy_net=policy_model,
                          policy_optimizer=optimizer_policy, value_optimizer=optimizer_value, gamma=0.99, device=dev)
            episode_reward.append(np.array(reward_array).sum())
            episode_len_array.append(episode_len)
        if e % save_frequency == 0:
            save_num = e / save_frequency
            if os.path.isfile(file_path + "New_AC_Entropy_policy_model_{}.pt".format(save_num - 1)):
                os.remove(file_path + "New_AC_Entropy_policy_model_{}.pt".format(save_num - 1))
            if os.path.isfile(file_path + "New_AC_Entropy_value_model_{}.pt".format(save_num - 1)):
                os.remove(file_path + "New_AC_Entropy_value_model_{}.pt".format(save_num - 1))
            if os.path.isfile(file_path + "New_AC_Entropy_Episode_Reward.pickle"):
                os.remove(file_path + "New_AC_Entropy_Episode_Reward.pickle")
            if os.path.isfile(file_path + "New_AC_Entropy_Episode_Length.pickle"):
                os.remove(file_path + "New_AC_Entropy_Episode_Length.pickle")

            torch.save(policy_model.state_dict(), file_path + "New_AC_Entropy_policy_model_{}.pt".format(save_num))
            torch.save(value_model.state_dict(), file_path + "New_AC_Entropy_value_model_{}.pt".format(save_num))

            with open(file_path + "New_AC_Entropy_Episode_Reward.pickle", "wb") as f:
                pickle.dump(episode_reward, f)

            with open(file_path + "New_AC_Entropy_Episode_Length.pickle", "wb") as f:
                pickle.dump(episode_len_array, f)
    return policy_model, value_model


if __name__ == "__main__":
    file_path = "/project/def-m2nagapp/partha9/LTR/"
    cache_path = "/scratch/partha9/.buffer_cache_ac"
    prev_policy_model_path = "/project/def-m2nagapp/partha9/LTR/AspectJ_New_AC_policy_model_124.0.pt"
    prev_value_model_path = "/project/def-m2nagapp/partha9/LTR/AspectJ_New_AC_value_model_124.0.pt"
    Path(file_path).mkdir(parents=True, exist_ok=True)
    env = LTREnvV2(data_path=file_path + "Data/TrainData/Bench_BLDS_Aspectj_Dataset.csv", model_path="microsoft/codebert-base",
                   tokenizer_path="microsoft/codebert-base", action_space_dim=31, report_count=100, max_len=512,
                   use_gpu=False, caching=True, file_path=file_path, project_list=['AspectJ'])
    obs = env.reset()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    buffer = CustomBuffer(6000, cache_path=cache_path)
    policy, value = train_actor_critic(total_time_step=7500, sample_size=128)
