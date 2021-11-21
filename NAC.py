import glob
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
        # x = x.squeeze(0)
        x = self.lin_layer2(x)
        return torch.softmax(x, dim=1), [new_h, new_c]

    def select_action(self, x, actions, hidden=None):
        action_prob, hidden_state = self.forward(x, actions, hidden)
        action = action_prob[-1].multinomial(1)
        return action, hidden_state

    def get_kl(self, x, actions, hidden=None):
        action_prob1, _ = self.forward(x, actions, hidden)
        action_prob0 = action_prob1.detach()
        kl = action_prob0 * (torch.log(action_prob0) - torch.log(action_prob1))
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x, current_action, actions, hidden=None):
        action_prob, _ = self.forward(x, actions, hidden)
        return torch.log(action_prob.squeeze(1).gather(1, current_action))

    def get_fim(self, x, actions, hidden=None):
        action_prob, _ = self.forward(x, actions, hidden)
        M = action_prob.pow(-1).view(-1).detach()
        return M, action_prob, {}


def a2c_step(policy_net, value_net, optimizer_policy, optimizer_value, states, actions, returns, advantages, l2_reg, batch_picked, batch_hidden, batch_hidden_states_value_model):
    """update critic"""
    values_pred, _ = value_net(states, actions=batch_picked, hidden=batch_hidden_states_value_model)
    value_loss = (values_pred.view(-1,1) - returns).pow(2).mean()
    # weight decay
    for param in value_net.parameters():
        value_loss += param.pow(2).sum() * l2_reg
    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()

    """update policy"""
    log_probs = policy_net.get_log_prob(states, current_action=actions, actions=batch_picked, hidden=batch_hidden)
    policy_loss = -(log_probs * advantages).mean()
    optimizer_policy.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 40)
    optimizer_policy.step()


def to_device(device, *args):
    return [x.to(device) for x in args]


def estimate_advantages(rewards, masks, values, gamma, tau, device):
    rewards, masks, values = rewards.to(device), masks.to(device), values.to(device)
    tensor_type = type(rewards)
    deltas = tensor_type(rewards.size(0), 1).to(device)
    advantages = tensor_type(rewards.size(0), 1).to(device)

    prev_value = torch.tensor(0).to(device)
    prev_advantage = torch.tensor(0).to(device)
    for i in reversed(range(rewards.size(0))):
        deltas[i] = (rewards[i] + (gamma * prev_value * masks[i]) - values[i])[0]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    returns = values + advantages
    advantages = (advantages - advantages.mean()) / advantages.std()

    advantages, returns = to_device(device, advantages, returns)
    return advantages, returns


def update_params(samples, value_net, policy_net, policy_optimizer, value_optimizer, gamma, tau, device, l2_reg):
    state, action, reward, next_state, done, info = samples
    batch_hidden = torch.tensor(np.array(
        [np.stack([np.array(item['hidden'][0]) for item in info], axis=2)[0],
         np.stack([np.array(item['hidden'][1]) for item in info], axis=2)[0]])).to(device)
    batch_hidden_states_value_model = torch.tensor(np.array(
        [np.stack([np.array(item['hidden_states_value_model'][0]) for item in info], axis=2)[0],
         np.stack([np.array(item['hidden_states_value_model'][1]) for item in info], axis=2)[0]])).to(device)
    batch_picked = torch.tensor(
        [to_one_hot(item['picked'], max_size=env.action_space.n) for item in info]).to(device).type(
        torch.float)
    with torch.no_grad():
        values, _ = value_net(state.type(torch.float).to(device),actions=batch_picked, hidden=batch_hidden_states_value_model)

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(reward, done, values.view(-1,1), gamma, tau, device)

    """perform TRPO update"""
    a2c_step(policy_net, value_net, policy_optimizer, value_optimizer, state.type(torch.float).to(device), action.to(device), returns, advantages, l2_reg, batch_picked, batch_hidden, batch_hidden_states_value_model)

def train_actor_critic(total_time_step,sample_size, save_frequency=30):
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    policy_model = PolicyModel(env=env)
    value_model = ValueModel(env=env)
    # if len(glob.glob(file_path + "policy_model_*.pt")) > 0:
    #     policy_model.load_state_dict(torch.load(glob.glob(file_path + "policy_model_*.pt")[0], map_location=dev))
    # if len(glob.glob(file_path + "value_model_*.pt")) > 0:
    #     value_model.load_state_dict(torch.load(glob.glob(file_path + "value_model_*.pt")[0], map_location=dev))
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
        hidden_states_value_model = [torch.zeros([1, 1, value_model.lstm_hidden_space]).to(dev),
                  torch.zeros([1, 1, value_model.lstm_hidden_space]).to(dev)]
        picked = []
        reward_array = []
        pbar.set_description("Avg. reward {} Avg. episode {}".format(np.array(episode_reward).mean(),
                                                                     np.array(episode_len_array).mean()))
        episode_len = 0
        while not done:
            episode_len += 1
            prev_obs = torch.Tensor(prev_obs).to(dev)
            prev_obs = prev_obs.unsqueeze(0)
            temp_action = torch.from_numpy(to_one_hot(picked, max_size=env.action_space.n)).to(
                                           dev).type(torch.float)
            action, hidden = policy_model.select_action(prev_obs,actions=temp_action, hidden=hidden)
            _, hidden_states_value_model = value_model(prev_obs, actions=temp_action, hidden=hidden_states_value_model )
            action = int(action[0][0].detach().cpu().numpy())
            picked.append(action)
            obs, reward, done, info = env.step(action)
            reward_array.append(reward)
            info['hidden'] = [item.detach().cpu().numpy() for item in hidden]
            info['hidden_states_value_model'] = [item.detach().cpu().numpy() for item in hidden_states_value_model]
            info['picked'] = picked
            buffer.add(prev_obs.detach().cpu().numpy(), obs, np.array([action]), np.array([reward]), np.array([done]),
                       [info])
            prev_obs = obs
            if len(buffer) > 200:
                samples = buffer.sample(sample_size)
                state, action, reward, next_state, batch_done, info = samples  # samples.observations, samples.actions, samples.rewards, samples.next_observations, samples.dones, samples.info
                update_params(samples=samples, value_net=value_model,policy_net=policy_model,policy_optimizer=optimizer_policy,value_optimizer=optimizer_value,gamma=0.99,tau=0.95,device=dev,l2_reg=1e-3)
        episode_reward.append(np.array(reward_array).sum())
        episode_len_array.append(episode_len)
        if e % save_frequency == 0:
            save_num = e / save_frequency
            if os.path.isfile(file_path + "New_AC_policy_model_{}.pt".format(save_num - 1)):
               os.remove(file_path + "New_AC_policy_model_{}.pt".format(save_num - 1))
            if os.path.isfile(file_path + "New_AC_value_model_{}.pt".format(save_num - 1)):
               os.remove(file_path + "New_AC_value_model_{}.pt".format(save_num - 1))
            if os.path.isfile(file_path + "New_AC_Episode_Reward.pickle"):
               os.remove(file_path + "New_AC_Episode_Reward.pickle")
            if os.path.isfile(file_path + "New_AC_Episode_Length.pickle"):
               os.remove(file_path + "New_AC_Episode_Length.pickle")

            torch.save(policy_model.state_dict(), file_path + "New_AC_policy_model_{}.pt".format(save_num))
            torch.save(value_model.state_dict(), file_path + "New_AC_value_model_{}.pt".format(save_num))

            with open(file_path + "New_AC_Episode_Reward.pickle", "wb") as f:
                pickle.dump(episode_reward, f)

            with open(file_path + "New_AC_Episode_Length.pickle", "wb") as f:
                pickle.dump(episode_len_array, f)
    return policy_model, value_model

if __name__ == "__main__":
    file_path = "/project/def-m2nagapp/partha9/LTR/"
    Path(file_path).mkdir(parents=True, exist_ok=True)
    env = LTREnvV2(data_path=file_path + "Data/TrainData/Bench_BLDS_Dataset.csv", model_path="microsoft/codebert-base",
                   tokenizer_path="microsoft/codebert-base", action_space_dim=31, report_count=50, max_len=512,
                   use_gpu=False, caching=True, file_path=file_path)
    obs = env.reset()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    buffer = CustomBuffer(8000)
    policy, value = train_actor_critic(total_time_step=7500, sample_size=64)
