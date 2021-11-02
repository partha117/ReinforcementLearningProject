import gym
import numpy as np
from gym import spaces
from transformers import RobertaTokenizer, RobertaModel
from Model import ClassifierModel, ClassificationHead
import torch
import pandas as pd
import zlib

def decode(text):
    return zlib.decompress(bytes.fromhex(text)).decode()
class SourceCodeEnv(gym.Env):
    def __init__(self, embedding_size, tokenizer_path, model_path, data_path,reward_estimator_path, level_limit=3):
        super(SourceCodeEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.tokenizer_path = tokenizer_path
        self.model_path = model_path
        self.data_path = data_path
        self.level_limit = level_limit
        self.reward_estimator_path = reward_estimator_path
        self.tokenizer = RobertaTokenizer(self.tokenizer_path + "/aster-vocab.json",
                                 self.tokenizer_path + "/aster-merges.txt")
        self.model = RobertaModel.from_pretrained(self.model_path)
        self.reward_estimator = torch.load(self.reward_estimator_path)
        self.df = pd.read_csv(self.data_path)
        self.chunk_list = []
        self.queue = []
        self.current_index = 0
        self.current_line = 0
        self.last_chunk_end = 0
        self.current_level = 0
        self.total_line = 0
        self.current_chunk = None
        self.action_space = spaces.Discrete(n=2)
        self.observation_space = spaces.Box(low=np.full(shape=(embedding_size + 3, 1), fill_value=-np.inf),
                                            high=np.full(shape=(embedding_size + 3, 1), fill_value=np.inf), shape=(embedding_size + 3, 1),
                                            dtype=np.float32)

    def step(self, action):
        # Execute one time step within the environment
        if action:
            self._take_action()
            reward = np.random.randint(low=1, high= 5, size=(1))
            # Todo replace by estimator
        else:
            self.current_line += 1
            reward = 0

        if self.current_level >= self.level_limit or self.current_line >= self.total_line:
            done = True
        else:
            done = False
        obs = self._next_observation()
        return obs, reward, done, {}
    def _take_action(self):
        self.queue.append((self.last_chunk_end, self.current_line, self.current_level))
        self.queue.append((self.current_line + 1, self.total_line, self.current_level))
        self.chunk_list.append((self.last_chunk_end, self.current_line))
        self.chunk_list.append((self.current_line + 1 if self.current_line != self.total_line else self.current_line, self.total_line))
        self.current_chunk = self.queue.pop(0)
        # ToDo: Do not open empty chunk
        self.current_level += 1
        self.current_line = self.current_chunk[0]
        self.last_chunk_end = self.current_chunk[0]
    def reset(self):
        # Reset the state of the environment to an initial state
        self.chunk_list = []
        self.queue = []
        self.current_index = 0
        self.current_line = 0
        self.last_chunk_end = 0
        self.current_level = 0
        self.total_line = 0
        self.current_chunk = None
        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass

    def _next_observation(self):
        current_index = self.df.iloc[self.current_index]
        current_index_code, current_index_report = current_index.file_content, current_index.report
        temp = decode(current_index_code).split("\n")
        self.total_line = len(temp) if self.current_chunk is None else self.current_chunk[1]
        current_code_line = temp[self.current_line]
        current_code_line_data = self.tokenizer.batch_encode_plus([current_code_line], max_length=512, pad_to_multiple_of=512,
                                            truncation=True,
                                            padding=True,
                                            return_tensors='pt')
        current_report_data = self.tokenizer.batch_encode_plus([current_index_report], max_length=512,
                                                                  pad_to_multiple_of=512,
                                                                  truncation=True,
                                                                  padding=True,
                                                                  return_tensors='pt')
        current_code_line_embeddings = self.model(**current_code_line_data)
        # current_report_embeddings = self.model(**current_report_data)
        current_code_line_embeddings = torch.mean(current_code_line_embeddings['last_hidden_state'][0].detach(),dim=1)
        return np.append(current_code_line_embeddings, [[self.current_line],[self.current_level]])

if __name__ == "__main__":
    bug_env = SourceCodeEnv(embedding_size=512,
                            tokenizer_path="Models/tokenizer/",
                            model_path="roberta-base",
                            data_path="Data/Small_dataset.csv",
                            reward_estimator_path="Models/Model_LSTM_12_Embed_size_256_Overlap_size_0",
                            level_limit=3)
    for i in range(10):
        bug_env.step(0)

    bug_env.step(1)
    for i in range(10):
        bug_env.step(0)

    bug_env.step(1)

    for i in range(10):
        bug_env.step(0)

    bug_env.step(1)

