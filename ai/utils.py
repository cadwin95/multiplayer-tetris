import torch
import numpy as np
from collections import deque
import random
from typing import List, Tuple

class ReplayBuffer:
    """경험 재생 버퍼"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)

def prepare_training_data(batch: List[Tuple], device: torch.device) -> Tuple:
    """배치 데이터를 텐서로 변환"""
    states = torch.FloatTensor([item[0] for item in batch]).to(device)
    actions = torch.LongTensor([item[1] for item in batch]).to(device)
    rewards = torch.FloatTensor([item[2] for item in batch]).to(device)
    next_states = torch.FloatTensor([item[3] for item in batch]).to(device)
    dones = torch.FloatTensor([item[4] for item in batch]).to(device)
    
    return states, actions, rewards, next_states, dones

def save_model(model: torch.nn.Module, path: str):
    """모델 저장"""
    torch.save(model.state_dict(), path)

def load_model(model: torch.nn.Module, path: str):
    """모델 로드"""
    model.load_state_dict(torch.load(path))
