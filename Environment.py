import numpy as np
import gym
import pandas as pd
from gym import spaces
import torch
import zlib
import random
from transformers import AutoModel, AutoTokenizer
from pathlib import Path

class LTREnv(gym.Env):
    def __init__(self, data_path, model_path, tokenizer_path, action_space_dim, report_count, max_len=512, use_gpu=True):
        super(LTREnv, self).__init__()
        if use_gpu:
            self.dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.dev = "cpu"
        self.current_file = None
        self.current_id = None
        self.df = None  # done
        self.sampled_id = None  # done
        self.filtered_df = None  # done
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(tokenizer_path).to(self.dev)
        self.data_path = data_path
        self.action_space_dim = action_space_dim
        self.action_space = spaces.Discrete(self.action_space_dim)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(1537,), dtype=np.float32)

        self.report_count = report_count
        self.previous_obs = None
        self.picked = []
        self.remained = []
        self.t = 0
        self.__get_ids()

    @staticmethod
    def decode(text):
        return zlib.decompress(bytes.fromhex(text)).decode()

    @staticmethod
    def reduce_dimension_by_mean_pooling(embeddings, attention_mask, to_numpy=False):
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings.detach() * mask
        summed = torch.sum(masked_embeddings, 1)
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / summed_mask
        return mean_pooled.numpy() if to_numpy else mean_pooled

    def __get_ids(self):
        self.df = pd.read_csv(self.data_path)
        self.df = self.df[self.df['report'].notna()]
        id_list = self.df.groupby('id')['cid'].count()
        id_list = id_list[id_list == int(self.action_space_dim)].index.to_list()
        id_list = self.df[(self.df['id'].isin(id_list)) & (self.df['match'] == 1)]['id'].unique().tolist()
        self.sampled_id = random.sample(id_list, min(len(id_list), self.report_count))

    def reset(self):
        self.previous_obs = None
        self.current_id = random.sample(self.sampled_id, 1)[0]
        self.__get_filtered_df()
        self.picked = []
        self.remained = self.filtered_df['cid'].tolist()
        self.t = 0
        self.picked.append(random.sample(self.remained, 1)[0])
        self.remained.remove(self.picked[-1])
        return self.__get_observation()

    def __calculate_reward(self):
        if self.t == 0:
            return 0
        else:
            relevance = self.df[self.df['cid'] == self.picked[-1]]['match'].tolist()[0]
            return relevance / np.log2(self.t + 1) #if relevance == 1 else -np.log2(self.t + 1)

    def step(self, action):
        temp = self.filtered_df['cid'].tolist()[action]
        info = {"invalid": False}
        obs, reward, done = None, None, None
        if temp in self.remained:
            self.picked.append(temp)
            self.remained.remove(temp)
            obs = self.__get_observation()
            reward = self.__calculate_reward()
            done = self.t == len(self.filtered_df)
        else:
            info['invalid'] = True
            obs = self.previous_obs
            done = True # self.t == len(self.filtered_df)
            # ToDo: Check it
            reward = -100
        return obs, reward, done, info

    def __get_filtered_df(self):
        self.filtered_df = self.df[self.df["id"] == self.current_id].reset_index()
        if hasattr(self, "caching") and getattr(self, "caching"):
            self.filtered_df = self.filtered_df.sample(frac=1, random_state=13).reset_index(drop=True)
        else:
            self.filtered_df = self.filtered_df.sample(frac=1).reset_index(drop=True)

    def __get_observation(self):
        self.t += 1
        report_data, code_data = self.df[
                                     (self.df['cid'] == self.picked[-1]) & (self.df['id'] == self.current_id)].report, \
                                 self.df[
                                     (self.df['cid'] == self.picked[-1]) & (
                                             self.df['id'] == self.current_id)].file_content
        report_data, code_data = report_data.values.tolist()[0], code_data.values.tolist()[0]
        report_token, code_token = self.tokenizer.batch_encode_plus([report_data], max_length=self.max_len,
                                                                    pad_to_multiple_of=self.max_len,
                                                                    truncation=True,
                                                                    padding=True,
                                                                    return_tensors='pt'), \
                                   self.tokenizer.batch_encode_plus([self.decode(code_data)], max_length=self.max_len,
                                                                    pad_to_multiple_of=self.max_len,
                                                                    truncation=True,
                                                                    padding=True,
                                                                    return_tensors='pt')
        report_output, code_output = self.model(**report_token.to(self.dev)), self.model(**code_token.to(self.dev))
        report_embedding, code_embedding = self.reduce_dimension_by_mean_pooling(report_output.last_hidden_state,
                                                                                 report_token['attention_mask']), \
                                           self.reduce_dimension_by_mean_pooling(code_output.last_hidden_state,
                                                                                 code_token['attention_mask'])
        final_rep = np.concatenate([report_embedding, code_embedding, [[self.t]]], axis=1)[0]
        self.previous_obs = final_rep
        return final_rep

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        # agent is represented as a cross, rest as a dot
        print("Current ranking of the documents")
        for i, item in enumerate(self.picked):
            print("Ranking: {} Document Cid: {} Timestep: {}".format(i + 1, item, self.t))

class LTREnvV2(LTREnv):
    def __init__(self, data_path, model_path, tokenizer_path, action_space_dim, report_count, max_len=512, use_gpu=True, caching=False):
        super(LTREnvV2, self).__init__(data_path, model_path, tokenizer_path, action_space_dim, report_count, max_len=512, use_gpu=use_gpu)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(31, 1537), dtype=np.float32)
        self.all_embedding = []
        self.caching = caching
    def reset(self):
        self.all_embedding = []
        return super(LTREnvV2, self).reset()

    def _LTREnv__get_observation(self):
        self.t += 1

        if len(self.all_embedding) == 0:
            if not self.caching or not Path(".caching/{}_all_embedding.npy".format(self.current_id)).is_file():
                for row in self.filtered_df.iterrows():
                    report_data, code_data = row[1].report, row[1].file_content
                    report_token, code_token = self.tokenizer.batch_encode_plus([report_data], max_length=self.max_len,
                                                                                pad_to_multiple_of=self.max_len,
                                                                                truncation=True,
                                                                                padding=True,
                                                                                return_tensors='pt'), \
                                               self.tokenizer.batch_encode_plus([self.decode(code_data)], max_length=self.max_len,
                                                                                pad_to_multiple_of=self.max_len,
                                                                                truncation=True,
                                                                                padding=True,
                                                                                return_tensors='pt')
                    report_output, code_output = self.model(**report_token.to(self.dev)), self.model(**code_token.to(self.dev))
                    report_embedding, code_embedding = self.reduce_dimension_by_mean_pooling(report_output.last_hidden_state,
                                                                                             report_token['attention_mask']), \
                                                       self.reduce_dimension_by_mean_pooling(code_output.last_hidden_state,
                                                                                             code_token['attention_mask'])
                    final_rep = np.concatenate([report_embedding, code_embedding, [[0]]], axis=1)[0]
                    self.all_embedding.append(final_rep)
                if self.caching:
                    Path(".caching/").mkdir(parents=True, exist_ok=True)
                    np.save(".caching/{}_all_embedding.npy".format(self.current_id), self.all_embedding)

            else:
                self.all_embedding = np.load(".caching/{}_all_embedding.npy".format(self.current_id)).tolist()
        action_index = self.filtered_df['cid'].tolist().index(self.picked[-1])
        # temp_embedding = self.all_embedding
        # temp_embedding[action_index] = np.zeros_like(self.all_embedding[action_index])
        # stacked_rep = np.stack(temp_embedding)
        # stacked_rep[:, -1] = self.t
        self.all_embedding[action_index] = np.zeros_like(self.all_embedding[action_index])
        stacked_rep = np.stack(self.all_embedding)
        stacked_rep[action_index, -1] = self.t
        self.previous_obs = stacked_rep
        return stacked_rep


class NaivePrioritizedBuffer(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def add(self, state, next_state, action, reward, done,infos):
        assert state.ndim == next_state.ndim
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        states = np.concatenate(batch[0])
        actions = batch[1]
        rewards = batch[2]
        next_states = np.concatenate(batch[3])
        dones = batch[4]

        return states, actions, next_states, dones, rewards, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

def get_replay_buffer(buffer_size,env,device="cpu",priority=False,alpha=0.6):
    if not priority:
        buffer = ReplayBuffer(buffer_size,env.observation_space,env.action_space,device)
    else:
        buffer = NaivePrioritizedBuffer(capacity=buffer_size,prob_alpha=alpha)
    current_size = 0
    while current_size < buffer_size:
        prev_obs = env.reset()
        action_list = env.filtered_df['cid'].sample(frac=1).tolist()
        # ToDo: one action already been picked at reset
        for action_id in action_list:
            action = env.filtered_df['cid'].tolist().index(action_id)
            obs, reward, done, info = env.step(action)
            print(action, reward)
            buffer.add(prev_obs,obs,np.array([action]),np.array([reward]),np.array([done]),[info])
            prev_obs = obs
            current_size += 1
            if done:
                break
    return buffer
if __name__ == "__main__":
    from stable_baselines3 import DQN
    from stable_baselines3.common.buffers import BaseBuffer, ReplayBuffer, ReplayBufferSamples
    env = LTREnvV2(data_path="Data/TrainData/Bench_BLDS_Dataset.csv", model_path="microsoft/codebert-base",
                 tokenizer_path="microsoft/codebert-base", action_space_dim=31, report_count=50, max_len=512, use_gpu=False, caching=True)
    obs = env.reset()
    buff = get_replay_buffer(5000, env, device="cuda:0",priority=True)
    buff.sample(30)
    model = DQN("MlpPolicy", env, verbose=1, buffer_size=5000,)
    model.learn(total_timesteps=100000)
    Path("Models/RL").mkdir(parents=True, exist_ok=True)
    model.save("Models/RL/DQN")
    # print("Loading Model ...")
    # model = DQN.load("Models/RL/DQN")
    #
    # obs = env.reset()
    # for i in range(10000):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = env.step(action)
    #     print("Reward: {}".format(reward))
    #     env.render()
    #     if done:
    #         obs = env.reset()
