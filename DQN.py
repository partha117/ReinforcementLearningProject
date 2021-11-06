import torch
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

class DoubleDQN(nn.Module):
    def __init__(self, env):
        super(DoubleDQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(env.observation_space.shape[1])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(env.observation_space.shape[0])))
        linear_input_size = convw * convh * 32
        self.lin_layer1 = nn.Linear(linear_input_size, 256)
        self.lstm_layer = nn.LSTM(input_size=256, hidden_size=128, batch_first=True)
        self.lin_layer2 = nn.Linear(128, env.action_space.n)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.lin_layer1(x))
        x = self.lin_layer2(x)
        return x


def run_one_iter(q_net, target_net, state, action, reward, next_state, done, optim, gamma):
    current_Q_values = q_net(state).gather(1, action)
    next_Q_values = target_net(next_state).detach()
    next_max_Q_value = torch.max(next_Q_values, dim=1).values
    next_max_Q_value = next_max_Q_value.view(next_Q_values.shape[0],1)
    target_Q_value = reward + gamma * next_max_Q_value * (1 - done)
    loss = torch.nn.MSELoss()(target_Q_value, current_Q_values)
    loss = torch.sqrt(loss)

    # # clip the bellman error between [-1 , 1]
    # clipped_bellman_error = bellman_error.clamp(-1, 1)
    # # Note: clipped_bellman_delta * -1 will be right gradient
    # d_error = clipped_bellman_error * -1.0
    # Clear previous gradients before backward pass
    optim.zero_grad()
    loss.backward()
    optim.step()
    # for target_param, local_param in zip(target_model.parameters(),
    #                                      local_model.parameters()):
    #     target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
    return loss


def train_dqn(buffer, env, total_time_step=10000, sample_size=30, learning_rate=0.01, update_frequency=500, tau=0.3):
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    q_network = DoubleDQN(env=env).to(dev)
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    target_q_network = DoubleDQN(env=env).to(dev)
    loss_accumulator = []
    window_loss_accumulator = []
    reward_accumulator = []
    window_reward_accumulator = []
    for iter_no in tqdm(range(total_time_step)):
        samples = buffer.sample(sample_size)
        state, action, reward, next_state, done = samples.observations, samples.actions, samples.rewards, samples.next_observations, samples.dones
        loss = run_one_iter(q_net=q_network, target_net=target_q_network, state=state.to(dev), action=action.to(dev), reward=reward.to(dev),
                            next_state=next_state.to(dev), done=done.to(dev), optim=optimizer, gamma=0.9)
        loss_accumulator.append(loss.detach().cpu().numpy())
        # test_q_value = q_network(state).detach()
        window_loss = np.array(loss_accumulator[-21:-1])
        window_loss_accumulator.append(window_loss.mean())
        if iter_no % update_frequency == 0:
            for target_param, local_param in zip(target_q_network.parameters(),
                                                 q_network.parameters()):
                target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    plt.plot(window_loss_accumulator)
    plt.show()
    return q_network


if __name__ == "__main__":
    env = LTREnvV2(data_path="Data/TrainData/Bench_BLDS_Dataset.csv", model_path="microsoft/codebert-base",
                   tokenizer_path="microsoft/codebert-base", action_space_dim=31, report_count=50, max_len=512,
                   use_gpu=False, caching=True)
    obs = env.reset()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    buffer = get_replay_buffer(8000, env, early_stop=5000, device="cpu")
    model = train_dqn(buffer=buffer, sample_size=64, env=env, total_time_step=5000, update_frequency=200,tau=0.01)
    Path("TrainedModels/").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "TrainedModels/DDQN.pt")
    # model = DoubleDQN(env=env)
    # state_dict = torch.load("TrainedModels/DDQN.pt")
    # model.load_state_dict(state_dict=state_dict)
    # done = False
    # picked = []
    # while not done:
    #     q_values = model(torch.from_numpy(np.expand_dims(obs, axis=0)).float()).detach()
    #     print(q_values)
    #     max_indices = torch.argmax(q_values)
    #     obs, reward, done, info = env.step(max_indices)
    #     print("-----", reward)
        # max_value = torch.max(q_values)
        # max_indices = (q_values == max_value).nonzero(as_tuple=False)
        # probable_step = None
        # for item in max_indices:
        #     if item[1] not in picked:
        #         obs, reward, done, info = env.step(item[1])
        #         picked.append(item[1])
        #         break
        # print(reward, picked)