import glob
import psutil
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import torch
from torch import nn
from Buffer import CustomBuffer
from DQN import train_dqn_epsilon, to_one_hot
from Environment import LTREnvV3
import torch.nn.functional as F
import numpy as np
import pickle

class TwoDConvReport(nn.Module):
    def __init__(self, in_channel, env):
        super(TwoDConvReport, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=4, kernel_size=5, stride=3)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, stride=3)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=5, stride=3)
        self.bn3 = nn.BatchNorm2d(4)

        def conv2d_size_out(size, kernel_size=5, stride=3):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(env.observation_space.shape[1])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(env.observation_space.shape[0])))
        self.linear_input_size = convw * convh * 4

    def forward(self, x):
        # # # # print("twpo", x.shape)
        seq_length = x.size(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), seq_length, -1)
        return x
class TwoDConv(nn.Module):
    def __init__(self, in_channel, env):
        super(TwoDConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(16)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(env.observation_space.shape[1])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(env.observation_space.shape[0])))
        self.linear_input_size = convw * convh * 16

    def forward(self, x):
        # # # # print("twpo", x.shape)
        seq_length = x.size(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), seq_length, -1)
        return x


class ValueModel(nn.Module):
    def __init__(self, env):
        super(ValueModel, self).__init__()
        self.source_conv_net = TwoDConv(env=env, in_channel=env.action_space.n)
        self.report_conv_net = TwoDConvReport(env=env, in_channel=1)
        self.lstm_hidden_space = 256

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(env.observation_space.shape[1])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(env.observation_space.shape[0])))
        linear_input_size = 4872  # convw * convh * 32 * 2
        self.action_space = env.action_space.n
        self.lstm = nn.LSTM(input_size=linear_input_size, hidden_size=self.lstm_hidden_space, batch_first=True)
        self.lin_layer2 = nn.Linear(self.lstm_hidden_space * env.action_space.n, 1)

    def forward(self, x, actions, hidden=None):
        # # print("policy", x.shape)
        x_source = self.source_conv_net(x[:, :, :, 768:])
        x_report = self.report_conv_net(x[:, 0, :, :768].unsqueeze(1))
        x_report = x_report.repeat(1, self.action_space, 1)
        # # print("source", x_source.shape)
        # # print("report", x_report.shape)
        x = torch.concat([x_report, x_source], axis=2)
        # # print("concat", x.shape)
        x, (new_h, new_c) = self.lstm(x, (hidden[0], hidden[1]))
        x = x.reshape(x.size(0), -1)
        x = self.lin_layer2(x)
        # print("value s", x.shape)
        return x, [new_h, new_c]


class PolicyModel(nn.Module):
    def __init__(self, env):
        super(PolicyModel, self).__init__()
        self.source_conv_net = TwoDConv(env=env, in_channel=env.action_space.n)
        self.report_conv_net = TwoDConvReport(env=env, in_channel=1)
        self.lstm_hidden_space = 256

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(env.observation_space.shape[1])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(env.observation_space.shape[0])))
        linear_input_size = 4872 # convw * convh * 32 * 2
        self.action_space = env.action_space.n
        self.lstm = nn.LSTM(input_size=linear_input_size, hidden_size=self.lstm_hidden_space, batch_first=True)
        self.lin_layer2 = nn.Linear(self.lstm_hidden_space * env.action_space.n, env.action_space.n)

    def forward(self, x, actions, hidden=None):
        # # print("policy", x.shape)
        x_source = self.source_conv_net(x[:, :, :, 768:])
        x_report = self.report_conv_net(x[:, 0, :, :768].unsqueeze(1))
        x_report = x_report.repeat(1, self.action_space, 1)
        # print("source", x_source.shape)
        # print("report", x_report.shape)
        x = torch.concat([x_report, x_source], axis=2)
        # print("concat", x.shape)
        x, (new_h, new_c) = self.lstm(x, (hidden[0], hidden[1]))
        x = x.reshape(x.size(0), -1)
        # print("before lin", x.shape)
        x = self.lin_layer2(x)
        # print("after lin", x.shape)
        # print(torch.softmax(x, dim=-1).shape, actions.shape)
        actions = actions.squeeze(1) if actions.dim() == 3 else actions
        x = torch.softmax(x, dim=-1) * actions
        x = x / x.sum()
        # # # print("Policy s")
        # print("policy shape", x.shape)
        return x, [new_h, new_c]



def a2c_step(policy_net, optimizer_policy, optimizer_value, states, advantages, batch_picked, batch_hidden,
             lambda_val=1):
    """update critic"""
    # # print("starting a2c")
    value_loss = advantages.pow(2).mean()
    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()

    """update policy"""
    # # print("getting policy")
    # # print(states.shape)
    probs, _ = policy_net(states, actions=batch_picked, hidden=batch_hidden)
    # print("probs", probs.shape)
    dist = torch.distributions.Categorical(probs=probs)
    action = dist.sample()
    # print(dist.log_prob(action).shape, advantages.shape)
    policy_loss = -dist.log_prob(action) * advantages.detach() + lambda_val * dist.entropy()
    policy_loss = policy_loss.mean()
    optimizer_policy.zero_grad()
    policy_loss.backward()
    optimizer_policy.step()
    # # print("a2c end")


def to_device(device, *args):
    return [x.to(device) for x in args]


def estimate_advantages(rewards, done, states, next_states, gamma, device, value_model, batch_hidden_value, batch_hidden_value_next,
                        batch_picked):
    # # print("startin advantage")
    rewards, masks, states, next_states = rewards.to(device), done.to(device).type(torch.float), states.to(device).type(
        torch.float), next_states.to(device).type(torch.float)
    # # print("d1", rewards.shape)
    # # print("d1", masks.shape)
    advantages = rewards + (1.0 - masks) * gamma * value_model(next_states, batch_picked, batch_hidden_value_next)[
        0].detach() - value_model(states, batch_picked, batch_hidden_value)[0]
    # # print("estimate advantage1")
    # print("advantages", advantages.shape)
    return advantages


def update_params(samples, value_net, policy_net, policy_optimizer, value_optimizer, gamma, device):
    state, action, reward, next_state, done, info = samples
    next_state = next_state.squeeze(1)
    # # print("update params", state.shape,next_state.shape)
    batch_hidden = torch.tensor(np.array(
        [np.stack([np.array(item['hidden'][0]) for item in info], axis=2)[0],
         np.stack([np.array(item['hidden'][1]) for item in info], axis=2)[0]])).to(device)
    batch_hidden_value = torch.tensor(np.array(
        [np.stack([np.array(item['hidden_value'][0]) for item in info], axis=2)[0],
         np.stack([np.array(item['hidden_value'][1]) for item in info], axis=2)[0]])).to(device)
    batch_hidden_value_next = torch.tensor(np.array(
        [np.stack([np.array(item['hidden_value_next'][0]) for item in info], axis=2)[0],
         np.stack([np.array(item['hidden_value_next'][1]) for item in info], axis=2)[0]])).to(device)
    batch_picked = torch.tensor(np.array(
        [to_one_hot(item['picked'], max_size=env.action_space.n) for item in info])).to(device).type(
        torch.float)

    """get advantage estimation from the trajectories"""
    advantages = estimate_advantages(reward, done, state, next_state, gamma, device, value_net, batch_hidden_value, batch_hidden_value_next,
                                     batch_picked)

    """perform TRPO update"""
    a2c_step(policy_net, policy_optimizer, value_optimizer, state.type(torch.float).to(device), advantages,
             batch_picked, batch_hidden)


def train_actor_critic(total_time_step, sample_size, project_name, save_frequency=30):
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
        # # print("starting pbar")
        done = False
        prev_obs = env.reset()
        # # print("Got observation")
        hidden = [torch.zeros([1, 1, policy_model.lstm_hidden_space]).to(dev),
                  torch.zeros([1, 1, policy_model.lstm_hidden_space]).to(dev)]
        hidden_value = [torch.zeros([1, 1, value_model.lstm_hidden_space]).to(dev),
                        torch.zeros([1, 1, value_model.lstm_hidden_space]).to(dev)]
        picked = []
        reward_array = []
        pbar.set_description("Avg. reward {} Avg. episode {}".format(np.array(episode_reward).mean(),
                                                                     np.array(episode_len_array).mean()))
        episode_len = 0
        # # print("starting episode loop")
        while not done:
            episode_len += 1
            # # print("Before", prev_obs.shape)
            prev_obs = torch.Tensor(prev_obs).to(dev)
            # # print("Before1", prev_obs.shape)
            prev_obs = prev_obs.unsqueeze(0)
            # # print("Here", prev_obs.shape)
            temp_action = torch.from_numpy(to_one_hot(picked, max_size=env.action_space.n)).to(
                dev).type(torch.float)
            with torch.no_grad():
                action, temp_hidden = policy_model(prev_obs, actions=temp_action, hidden=hidden)
                _, temp_hidden_value = value_model(prev_obs, actions=temp_action, hidden=hidden_value)
            # print(action.shape)
            action = torch.distributions.Categorical(action).sample()
            # # print(action.shape)
            action = int(action[0].cpu().numpy())
            # # print("Taken", action)
            picked.append(action)
            obs, reward, done, info = env.step(action)
            # # print("new state", prev_obs.shape, obs.shape)
            reward_array.append(reward)
            info['hidden'] = [item.cpu().numpy() for item in hidden]
            info['picked'] = picked
            info['hidden'] = [item.cpu().numpy() for item in hidden]
            info['hidden_value'] = [item.cpu().numpy() for item in hidden_value]
            info['hidden_value_next'] = [item.cpu().numpy() for item in temp_hidden_value]
            hidden = temp_hidden
            hidden_value = temp_hidden_value
            buffer.add(prev_obs.squeeze(0).cpu().numpy(), obs, np.array([action]), np.array([reward]), np.array([done]),
                       [info])
            prev_obs = obs
        if len(buffer) > 70:
            # # print("In buffer sampling")
            samples = buffer.sample(sample_size)
            update_params(samples=samples, value_net=value_model, policy_net=policy_model,
                          policy_optimizer=optimizer_policy, value_optimizer=optimizer_value, gamma=0.99, device=dev)
            episode_reward.append(np.array(reward_array).sum())
            episode_len_array.append(episode_len)
        if e % save_frequency == 0:
            save_num = e / save_frequency
            if os.path.isfile(save_path + "{}_AC_Entropy_V2_policy_model_{}.pt".format(project_name, save_num - 1)):
                os.remove(save_path + "{}_AC_Entropy_V2_policy_model_{}.pt".format(project_name, save_num - 1))
            if os.path.isfile(save_path + "{}_AC_Entropy_V2_value_model_{}.pt".format(project_name, save_num - 1)):
                os.remove(save_path + "{}_AC_Entropy_V2_value_model_{}.pt".format(project_name, save_num - 1))
            if os.path.isfile(save_path + "{}_AC_Entropy_V2_Episode_Reward.pickle".format(project_name)):
                os.remove(save_path + "{}_AC_Entropy_V2_Episode_Reward.pickle".format(project_name))
            if os.path.isfile(save_path + "{}_AC_Entropy_V2_Episode_Length.pickle".format(project_name)):
                os.remove(save_path + "{}_AC_Entropy_V2_Episode_Length.pickle".format(project_name))

            torch.save(policy_model.state_dict(),
                       save_path + "{}_AC_Entropy_V2_policy_model_{}.pt".format(project_name, save_num))
            torch.save(value_model.state_dict(),
                       save_path + "{}_AC_Entropy_V2_value_model_{}.pt".format(project_name, save_num))

            with open(save_path + "{}_AC_Entropy_V2_Episode_Reward.pickle".format(project_name), "wb") as f:
                pickle.dump(episode_reward, f)

            with open(save_path + "{}_AC_Entropy_V2_Episode_Length.pickle".format(project_name), "wb") as f:
                pickle.dump(episode_len_array, f)
    return policy_model, value_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', default="/project/def-m2nagapp/partha9/LTR/", help='File Path')
    parser.add_argument('--cache_path', default="/scratch/partha9/.buffer_cache_ac", help='Cache Path')
    parser.add_argument('--prev_policy_model_path', default=None, help='Trained Policy Path')
    parser.add_argument('--prev_value_model_path', default=None, help='Trained Value Path')
    parser.add_argument('--train_data_path', help='Training Data Path')
    parser.add_argument('--save_path', help='Save Path')
    parser.add_argument('--project_name', help='Project Name')
    options = parser.parse_args()
    file_path = options.file_path
    cache_path = options.cache_path
    prev_policy_model_path = options.prev_policy_model_path
    prev_value_model_path = options.prev_value_model_path
    train_data_path = options.train_data_path
    project_name = options.project_name
    save_path = options.save_path
    # file_path = ""  # "/project/def-m2nagapp/partha9/LTR/"
    # cache_path = ".cache"  # "/scratch/partha9/.buffer_cache_ac"
    # prev_policy_model_path = None  # "/project/def-m2nagapp/partha9/LTR/AspectJ_New_AC_policy_model_124.0.pt"
    # prev_value_model_path = None  # "/project/def-m2nagapp/partha9/LTR/AspectJ_New_AC_value_model_124.0.pt"
    # train_data_path = "Data/TrainData/Bench_BLDS_Aspectj_Dataset.csv"
    # project_name = "AspectJ"
    # save_path = ""
    Path(file_path).mkdir(parents=True, exist_ok=True)
    env = LTREnvV3(data_path=file_path + train_data_path, model_path="microsoft/codebert-base",
                   tokenizer_path="microsoft/codebert-base", action_space_dim=31, report_count=100, max_len=512,
                   use_gpu=False, caching=True, file_path=file_path, project_list=[project_name])
    obs = env.reset()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    buffer = CustomBuffer(6000, cache_path=cache_path)
    policy, value = train_actor_critic(total_time_step=7500, sample_size=32, project_name=project_name)

