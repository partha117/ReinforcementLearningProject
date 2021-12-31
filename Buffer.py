from collections import deque
import random
import numpy as np
import os
import gzip
import zlib
from pathlib import Path
import torch
import pandas as pd
from transformers import RobertaTokenizer, AutoModel, AutoTokenizer
from stable_baselines3.common.buffers import ReplayBuffer
from labml_nn.rl.dqn.replay_buffer import ReplayBuffer as PriorityReplayBuffer
def get_priority_replay_buffer(buffer_size,env,alpha=0.6):
    buffer = PriorityReplayBuffer(buffer_size,alpha=alpha)
    current_size = 0
    while current_size < buffer_size:
        prev_obs = env.reset()
        action_list = env.filtered_df['cid'].tolist()
        # ToDo: one action already been picked at reset
        for action_id in action_list:
            action = env.filtered_df['cid'].tolist().index(action_id)
            obs, reward, done, info = env.step(action)
            buffer.add(obs=prev_obs, action=np.array([action]), reward=np.array([reward]), next_obs=obs, done=np.array([done]))
            prev_obs = obs
            current_size += 1
            if done:
                break
    return buffer
def get_replay_buffer(buffer_size, env,early_stop=None,device="cpu"):
    buffer = ReplayBuffer(buffer_size,env.observation_space,env.action_space,device)
    current_size = 0
    while current_size < (early_stop if early_stop else buffer_size):
        prev_obs = env.reset()
        action_list = env.filtered_df['cid'].tolist()
        # if random.random() > 0.4:
        #     action_location = [i for i, item in enumerate(prev_obs[:, 0:1535]) if np.all(item == 0)]
        #     action_list.remove(action_location)
        # ToDo: one action already been picked at reset
        for action_id in action_list:
            action = env.filtered_df['cid'].tolist().index(action_id)
            obs, reward, done, info = env.step(action)
            buffer.add(prev_obs,obs,np.array([action]),np.array([reward]),np.array([done]),[info])
            prev_obs = obs
            current_size += 1
            if done:
                break
    return buffer


def compute_reward(t, picked, df):
    """
    Reward function for MDP
    """
    if t == 0:
        return 0
    else:
        # temp = np.array([df[df['cid'] == item]['match'].tolist()[0] for item in picked])
        # index = np.argmax(temp) + 1
        # a = np.array([0, 0, 1, 0, 0, 0, 1, 0,0,0,0,0,0,0,1])
        # position = np.argwhere(a == 1).reshape(-1)
        # position_padded = np.insert(position, len(position), 0)
        # position_shifted = np.insert(position, 0, 0)
        # difference = (position_padded - position_shifted)[1:-1]
        # penalty = np.sum(np.log(difference))
        # if np.all(temp == 0):
        #     return np.log2(t + 1)
        # elif len(temp == 1) == 1:
        #     return (1.0/index)
        # else:
        #     pass
        # return (1.0/index) / np.log2(t + 1)
        relevance = df[df['cid'] == picked[-1]]['match'].tolist()[0]

        return relevance / np.log2(t + 1)


def decode(text):
    return zlib.decompress(bytes.fromhex(text)).decode()


def reduce_dimension_by_mean_pooling(embeddings, attention_mask, to_numpy=False):
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    masked_embeddings = embeddings.detach() * mask
    summed = torch.sum(masked_embeddings, 1)
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / summed_mask
    return mean_pooled.numpy() if to_numpy else mean_pooled

class CustomBuffer(object):
    def __init__(self, buffer_length, cache_path=".Buffer", delete=True):
        self.state = []
        self.next_state = []
        self.action = []
        self.reward = []
        self.done = []
        self.info = []
        self.cache = cache_path
        if delete:
            os.system("rm -r {}".format(self.cache))
        Path(self.cache).mkdir(parents=True, exist_ok=True)
    def add(self, state, next_state, action, reward, done, info):
        # self.state.append(state)
        with gzip.GzipFile("{}/{}_state.npy.gz".format(self.cache, len(self.action)), "w") as state_file:
            np.save(state_file, arr=state)
        # np.save("{}/{}_state.npy".format(self.cache, len(self.action)), state)
        # self.next_state.append(next_state)
        with gzip.GzipFile("{}/{}_next_state.npy.gz".format(self.cache, len(self.action)), "w") as next_state_file:
            np.save(next_state_file, arr=next_state)
        # np.save("{}/{}_next_state.npy".format(self.cache, len(self.action)), next_state)
        self.action.append(action)
        self.reward.append(reward)
        self.done.append(done)
        self.info = self.info + info
    def sample(self, size):
        indices = np.random.choice(len(self.action), size)
        # state, action, reward, next_state, done, info = np.array(self.state)[indices], np.array(self.action)[indices], np.array(self.reward)[indices], np.array(self.next_state)[indices], np.array(self.done)[indices], np.array(self.info)[indices]
        # try:
        #     state, action, reward, next_state, done, info = np.array([np.load("{}/{}_state.npy".format(self.cache, item)) for item in indices]), np.array(self.action)[indices], \
        #                                                     np.array(self.reward)[indices], np.array([np.load("{}/{}_next_state.npy".format(self.cache, item)) for item in indices]), np.array(self.done)[indices], np.array(self.info)[
        #                                                         indices]
        try:
            state_temp = []
            next_state_temp = []
            for item in indices:
                with gzip.GzipFile("{}/{}_state.npy.gz".format(self.cache, item), "r") as state_file:
                    state_temp.append(np.load(state_file))
                with gzip.GzipFile("{}/{}_next_state.npy.gz".format(self.cache, item), "r") as next_state_file:
                    next_state_temp.append(np.load(next_state_file))
            state = np.array(state_temp)
            next_state = np.array(next_state_temp)
            del state_temp
            del next_state_temp
            action, reward, done, info = np.array(self.action)[indices], \
                                                            np.array(self.reward)[indices], np.array(self.done)[indices], np.array(self.info)[
                                                                indices]
        except Exception as ex:
            print(self.action)
            raise ex
        return torch.from_numpy(state), torch.from_numpy(action), torch.from_numpy(reward), torch.from_numpy(next_state), torch.from_numpy(done), info
    def __len__(self):
        return len(self.action)
class State:

    def __init__(self, t, qid, picked, remaining, caching=False):
        self.t = t
        self.qid = qid
        self.picked = picked
        self.remaining = remaining
        self.caching = caching
        self.vector = None

    def initial(self):
        return self.t == 0

    def terminal(self):
        return len(self.remaining) == 0

    def get_as_vector(self, tokenizer, model, df, max_len, use_gpu=True):
        if self.vector is not None:
            return self.vector
        else:
            if use_gpu:
                dev = "cuda:0" if torch.cuda.is_available() else "cpu"
            else:
                dev = "cpu"
            report_data, code_data = df[df['cid'] == self.picked[-1]].report, df[
                (df['cid'] == self.picked[-1]) & (df['id'] == self.qid)].file_content
            report_data = report_data.values[0] if report_data is not None else " "
            code_data = code_data.values[0] if code_data is not None else " "
            report_token, code_token = tokenizer.batch_encode_plus([report_data], max_length=max_len,
                                                                   pad_to_multiple_of=max_len,
                                                                   truncation=True,
                                                                   padding=True,
                                                                   return_tensors='pt'), \
                                       tokenizer.batch_encode_plus([decode(code_data)], max_length=max_len,
                                                                   pad_to_multiple_of=max_len,
                                                                   truncation=True,
                                                                   padding=True,
                                                                   return_tensors='pt')
            report_output, code_output = model(**report_token.to(dev)), model(**code_token.to(dev))
            report_embedding, code_embedding = reduce_dimension_by_mean_pooling(report_output.last_hidden_state,
                                                                                report_token['attention_mask']), \
                                               reduce_dimension_by_mean_pooling(code_output.last_hidden_state,
                                                                                code_token['attention_mask'])
            final_rep = np.concatenate([report_embedding, code_embedding, [[self.t]]], axis=1)[0]
            if self.caching:
                self.vector = final_rep
            return final_rep


class BasicBuffer:

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.buffer.maxlen:
            experience = (state, action, np.array([reward]), next_state, done)
            self.buffer.append(experience)
            return True
        else:
            return False

    def push_batch(self, df, n, action_no=None, caching=True):
        # id_list = df[df['match'] == 1]['id'].unique().tolist()
        # sampled_id = random.sample(id_list, min(len(id_list), n))
        df = df[df['report'].notna()]
        id_list = df.groupby('id')['cid'].count()
        action_no = action_no if action_no is not None else id_list.mode()[0]
        id_list = id_list[id_list == int(action_no)].index.to_list()
        id_list = df[(df['id'].isin(id_list)) & (df['match'] == 1)]['id'].unique().tolist()
        sampled_id = random.sample(id_list, min(len(id_list), n))
        queue_filled_up = False
        for id in sampled_id:
            filtered_df = df[df["id"] == id].reset_index()
            filtered_df = filtered_df.sample(frac=1).reset_index(drop=True)
            if queue_filled_up:
                break
            for t, row in enumerate(filtered_df.iterrows()):
                # code_data = row[1]['file_content']
                # report_data = row[1]['report']
                # code_token = self.tokenizer(code_data)
                # report_token = self.tokenizer(report_data)
                # code_embedding = self.model(**code_token)
                # report_embedding = self.model(**report_token)
                if t == 0:
                    temp = set(filtered_df['cid'])
                    temp = temp - set([row[1]["cid"]])
                    old_state = State(t, row[1]["id"], [row[1]["cid"]], list(temp), caching=caching)
                else:
                    picked = old_state.picked + [row[1]["cid"]]
                    temp = set(filtered_df['cid'])
                    temp = temp - set(picked)
                    new_state = State(t, row[1]["id"], picked, list(temp), caching=caching)
                    reward = compute_reward(t, new_state.picked, filtered_df)
                    queue_filled_up = not self.push(old_state, row[1]["cid"], reward, new_state,
                                                    t + 1 == len(filtered_df))
                    old_state = new_state
                    if queue_filled_up:
                        break

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch,
                next_state_batch, done_batch)

    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":
    random.seed(a=13)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = RobertaTokenizer("Models/tokenizer" + "/aster-vocab.json",
                                 "Models/tokenizer" + "/aster-merges.txt")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    buffer = BasicBuffer(max_size=500)
    df = pd.read_csv("Data/TrainData/Bench_BLDS_Dataset.csv")
    buffer.push_batch(df, 50)
    state_batch, action_batch, reward_batch, \
    next_state_batch, done_batch = buffer.sample(1)
    print(state_batch[0].get_as_vector(df=df, tokenizer=tokenizer, model=model, max_len=512, use_gpu=False))
