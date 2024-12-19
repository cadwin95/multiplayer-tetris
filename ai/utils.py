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
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        actual_batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, actual_batch_size)
        
        states = np.array([item[0] for item in batch])
        actions = np.array([item[1] for item in batch])
        rewards = np.array([item[2] for item in batch])
        next_states = np.array([item[3] for item in batch])
        dones = np.array([item[4] for item in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)

def prepare_training_data(batch: Tuple[np.ndarray, ...], device: torch.device) -> Tuple[torch.Tensor, ...]:
    """배치 데이터를 텐서로 변환"""
    states, actions, rewards, next_states, dones = batch
    
    states = torch.from_numpy(states).to(device)
    actions = torch.from_numpy(actions).long().to(device)
    rewards = torch.from_numpy(rewards).float().to(device)
    next_states = torch.from_numpy(next_states).to(device)
    dones = torch.from_numpy(dones).float().to(device)
    
    return states, actions, rewards, next_states, dones

def save_model(model: torch.nn.Module, path: str):
    """모델 저장"""
    torch.save({
        'model': model.state_dict(),
        'optimizer': model.optimizer.state_dict() if hasattr(model, 'optimizer') else None,
        'episode': model.episode if hasattr(model, 'episode') else 0,
        'epsilon': model.epsilon if hasattr(model, 'epsilon') else 0
    }, path)

def load_model(model: torch.nn.Module, path: str):
    """모델 로드"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    if hasattr(model, 'optimizer') and checkpoint['optimizer']:
        model.optimizer.load_state_dict(checkpoint['optimizer'])
    if hasattr(model, 'episode'):
        model.episode = checkpoint['episode']
    if hasattr(model, 'epsilon'):
        model.epsilon = checkpoint['epsilon']
