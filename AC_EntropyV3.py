import glob
import psutil
from torch.nn.functional import softmax
import argparse
import os
from pathlib import Path
from tqdm import tqdm, trange
import torch
from torch import nn
from Buffer import CustomBuffer
from Environment import LTREnvV4, LTREnvV5
import torch.nn.functional as F
import numpy as np
import pickle



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

class ValueModel(nn.Module):
    def __init__(self, env, limit=768, training=True):
        super(ValueModel, self).__init__()
        self.limit = limit
        self.env = env
        self.training = training
        self.attention = AttLayer(limit=self.limit)
        self.linear1 = nn.Linear(in_features=self.limit * 2, out_features=256)
        self.linear2 = nn.Linear(in_features=256 * env.action_space.n, out_features=1)

    def forward(self, x):
        source, report = self.attention(x)
        x = torch.concat([source, torch.stack([report for i in range(source.shape[1])]).swapaxes(0, 1)], axis=2)
        x = F.relu(F.dropout(self.linear1(x), training=self.training, p=0.2))
        x = x.flatten(1)
        x = F.dropout(self.linear2(x), training=self.training, p=0.2)
        return F.sigmoid(x)

class PolicyModel(nn.Module):
    def __init__(self, env, limit=768, training=True):
        super(PolicyModel, self).__init__()
        self.limit = limit
        self.env = env
        self.training = training
        self.attention = AttLayer(limit=self.limit)
        self.linear1 = nn.Linear(in_features=self.limit * 2, out_features=self.limit)
        self.linear2 = nn.Linear(in_features=self.limit, out_features=1)

    def forward(self, x, actions):
        source, report = self.attention(x)
        x = torch.concat([source, torch.stack([report for i in range(source.shape[1])]).swapaxes(0, 1)], axis=2)
        x = F.relu(F.dropout(self.linear1(x), training=self.training, p=0.2))
        x = F.dropout(self.linear2(x), training=self.training, p=0.2)
        actions = actions.squeeze(1) if actions.dim() == 3 else actions
        x = softmax(x.squeeze(2), dim=1)
        x = torch.softmax(x, dim=-1) * actions
        x = x / x.sum()
        return x


def a2c_step(policy_net, optimizer_policy, optimizer_value, states, advantages, batch_picked,
             lambda_val=200):
    """update critic"""
    # # # # print("starting a2c")
    value_loss = advantages.pow(2).mean()
    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()

    """update policy"""
    # # # # print("getting policy")
    # # # # print(states.shape)
    probs = policy_net(states, actions=batch_picked)
    # # # print("probs", probs.shape)
    dist = torch.distributions.Categorical(probs=probs)
    action = dist.sample()
    # print(dist.log_prob(action).shape, advantages.shape)
    policy_loss = -dist.log_prob(action).unsqueeze(1) * advantages.detach() - lambda_val * dist.entropy().unsqueeze(1)
    policy_loss = policy_loss.mean()
    optimizer_policy.zero_grad()
    policy_loss.backward()
    optimizer_policy.step()
    return policy_loss
    # # # # print("a2c end")


def to_device(device, *args):
    return [x.to(device) for x in args]


def estimate_advantages(rewards, done, states, next_states, gamma, device, value_model, batch_picked):
    # # # # print("startin advantage")
    rewards, masks, states, next_states = rewards.to(device), done.to(device).type(torch.float), states.to(device).type(
        torch.float), next_states.to(device).type(torch.float)
    # # # # print("d1", rewards.shape)
    # print("d1", masks.shape)
    # print("v model", value_model(next_states).shape)
    advantages = rewards + (1.0 - masks) * gamma * value_model(next_states).detach() - value_model(states)
    # # # # print("estimate advantage1")
    # print("advantages", advantages.shape)
    return advantages


def update_params(samples, value_net, policy_net, policy_optimizer, value_optimizer, gamma, device, multi=False):
    from DQN import to_one_hot
    state, action, reward, next_state, done, info = samples
    next_state = next_state.squeeze(1)
    # # # # print("update params", state.shape,next_state.shape)
    batch_picked = torch.tensor(np.array(
        [to_one_hot(item['picked'], max_size=env.action_space.n) for item in info])).to(device).type(
        torch.float)
    """get advantage estimation from the trajectories"""
    advantages = estimate_advantages(reward, done, state, next_state, gamma, device, value_net, batch_picked)

    """perform TRPO update"""
    policy_loss = a2c_step(policy_net, policy_optimizer, value_optimizer, state.type(torch.float).to(device),
                           advantages,
                           batch_picked)
    return policy_loss


def update_learning_rate(optimizer, factor):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] / factor


def train_actor_critic(total_time_step, sample_size, project_name, start_from, save_frequency=30, multi=False, lr_frequency=200):
    from DQN import to_one_hot
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
    optimizer_policy = torch.optim.Adam(policy_model.parameters(), lr=0.001)
    optimizer_value = torch.optim.Adam(value_model.parameters(), lr=0.001)
    print("Loop starting from", start_from)
    pbar = trange(start_from, total_time_step)
    episode_len_array = []
    episode_reward = []
    policy_loss = None
    for e in pbar:
        # # # # print("starting pbar")
        print("current e", e)
        done = False
        prev_obs = env.reset()
        # # # # print("Got observation")
        picked = []
        reward_array = []
        pbar.set_description("Avg. reward {} Avg. episode {} Loss {}".format(np.array(episode_reward).mean(),
                                                                             np.array(episode_len_array).mean(),
                                                                             policy_loss))
        episode_len = 0
        # # # # print("starting episode loop")
        while not done:
            episode_len += 1
            # # # # print("Before", prev_obs.shape)
            prev_obs = torch.Tensor(prev_obs).to(dev)  # if not multi else torch.Tensor(prev_obs).to("cuda:1")
            # # # # print("Before1", prev_obs.shape)
            prev_obs = prev_obs.unsqueeze(0)
            # # # # print("Here", prev_obs.shape)
            temp_action = torch.from_numpy(to_one_hot(picked, max_size=env.action_space.n)).to(
                dev).type(torch.float)
            with torch.no_grad():
                action = policy_model(prev_obs, actions=temp_action)
            # print(action.shape)
            action = torch.distributions.Categorical(action).sample()
            # print(action.shape)
            action = int(action[0].cpu().numpy())
            # # # # print("Taken", action)
            picked.append(action)
            obs, reward, done, info = env.step(action)
            # # # # print("new state", prev_obs.shape, obs.shape)
            reward_array.append(reward)
            info['picked'] = picked
            buffer.add(prev_obs.squeeze(0).cpu().numpy(), obs, np.array([action]), np.array([reward]), np.array([done]),
                       [info])
            prev_obs = obs
        episode_reward.append(np.array(reward_array).sum())
        episode_len_array.append(episode_len)
        if len(buffer) > 400:
            # # # # print("In buffer sampling")
            samples = buffer.sample(sample_size)
            policy_loss = update_params(samples=samples, value_net=value_model, policy_net=policy_model,
                                        policy_optimizer=optimizer_policy, value_optimizer=optimizer_value, gamma=0.99,
                                        device=dev, multi=multi)
            policy_loss = policy_loss.detach().cpu().numpy()
        if e % lr_frequency == 0 and e != 0:
            update_learning_rate(optimizer_policy, 5)
            update_learning_rate(optimizer_value, 5)
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
            buffer.save_others()
    return policy_model, value_model


if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', default="/project/def-m2nagapp/partha9/LTR/", help='File Path')
    parser.add_argument('--cache_path', default="/scratch/partha9/.buffer_cache_ac", help='Cache Path')
    parser.add_argument('--prev_policy_model_path', default=None, help='Trained Policy Path')
    parser.add_argument('--prev_value_model_path', default=None, help='Trained Value Path')
    parser.add_argument('--train_data_path', help='Training Data Path')
    parser.add_argument('--save_path', help='Save Path')
    parser.add_argument('--start_from', default=0, help='Start from')
    parser.add_argument('--project_name', help='Project Name')
    options = parser.parse_args()
    file_path = options.file_path
    cache_path = options.cache_path
    prev_policy_model_path = options.prev_policy_model_path
    prev_value_model_path = options.prev_value_model_path
    train_data_path = options.train_data_path
    project_name = options.project_name
    save_path = options.save_path
    start_from = int(options.start_from)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    # file_path = ""  # "/project/def-m2nagapp/partha9/LTR/"
    # cache_path = ".cache"  # "/scratch/partha9/.buffer_cache_ac"
    # prev_policy_model_path = None  # "/project/def-m2nagapp/partha9/LTR/AspectJ_New_AC_policy_model_124.0.pt"
    # prev_value_model_path = None  # "/project/def-m2nagapp/partha9/LTR/AspectJ_New_AC_value_model_124.0.pt"
    # train_data_path = "Data/TrainData/Bench_BLDS_Aspectj_Dataset.csv"
    # project_name = "AspectJ"
    # save_path = ""
    # start_from = 0
    # dev = "cuda:0"
    Path(file_path).mkdir(parents=True, exist_ok=True)
    Path(save_path).mkdir(parents=True, exist_ok=True)
    env = LTREnvV5(data_path=file_path + train_data_path, model_path="microsoft/codebert-base",
                   tokenizer_path="microsoft/codebert-base", action_space_dim=31, report_count=None, code_max_len=2048,
                   report_max_len=512,
                   use_gpu=False, caching=True, file_path=file_path, project_list=[project_name], window_size=500)
    obs = env.reset()

    buffer = CustomBuffer(6000, cache_path=cache_path, delete=(start_from == 0), start_from=start_from * 31)
    policy, value = train_actor_critic(total_time_step=7500, sample_size=32, project_name=project_name, multi=True,
                                       start_from=start_from)
