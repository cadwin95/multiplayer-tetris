import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from model import create_model
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
from environment import TetrisEnv
from collections import deque

class ExpertDataset(Dataset):
    def __init__(self, states, actions):
        self.states = torch.FloatTensor(states)
        self.actions = torch.tensor(actions, dtype=torch.long)
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

def collect_expert_data(num_episodes=1000, max_steps_per_episode=1000):
    """전문가 데이터 수집"""
    env = TetrisEnv()
    states = []
    actions = []
    scores = []
    
    for episode in tqdm(range(num_episodes), desc="데이터 수집 중"):
        state, _ = env.reset()
        episode_score = 0
        
        for step in range(max_steps_per_episode):
            valid_actions = env.get_valid_actions()
            # 간단한 휴리스틱 전략
            action = max(valid_actions, key=lambda a: {
                0: -1 if env.board[:, 0].sum() > env.board[:, -1].sum() else 0,  # left
                1: 1 if env.board[:, 0].sum() > env.board[:, -1].sum() else 0,   # right
                2: 2,  # rotate (항상 시도)
                3: -1,  # down (낮은 우선순위)
                4: 3,  # hardDrop (높은 우선순위)
                5: -2   # hold (가장 낮은 우선순위)
            }[a])
            
            states.append(state)
            actions.append(action)
            
            next_state, reward, done, _, _ = env.step(action)
            episode_score += reward
            state = next_state
            
            if done:
                break
        
        scores.append(episode_score)
        
        if episode > 0 and episode % 100 == 0:
            mean_score = np.mean(scores[-100:])
            print(f"\n최근 100 에피소드 평균 점수: {mean_score:.2f}")
    
    return np.array(states), np.array(actions), scores

def pretrain(
    num_episodes: int = 1000,
    model_complexity: str = 'simple',
    batch_size: int = 64,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    save_interval: int = 5
):
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")
    
    # 전문가 데이터 수집
    print("\n전문가 데이터 수집 시작...")
    states, actions, scores = collect_expert_data(num_episodes)
    
    # 데이터 저장
    os.makedirs('./data', exist_ok=True)
    np.save('./data/X_train.npy', states)
    np.save('./data/y_train.npy', actions)
    np.save('./data/scores.npy', scores)
    
    # 데이터 형태 출력
    print(f"수집된 데이터 - states: {states.shape}, actions: {actions.shape}")
    print(f"평균 점수: {np.mean(scores):.2f}")
    print(f"최고 점수: {np.max(scores):.2f}")
    
    # Dataset 생성
    dataset = ExpertDataset(states, actions)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # 모델 초기화
    model = create_model(model_complexity).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # 로깅 설정
    log_dir = f'runs/pretrain_{model_complexity}_{int(time.time())}'
    writer = SummaryWriter(log_dir)
    
    # 체크포인트 디렉토리 생성
    os.makedirs(f'models/{model_complexity}/pretrain_checkpoints', exist_ok=True)
    
    # 학습 루프
    print("\n사전 학습 시작...")
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for states, actions in progress_bar:
            states = states.to(device)
            actions = actions.to(device)
            
            # 순전파
            predictions = model(states)
            loss = criterion(predictions, actions)
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 통계 업데이트
            total_loss += loss.item()
            _, predicted = torch.max(predictions.data, 1)
            correct_predictions += (predicted == actions).sum().item()
            total_predictions += actions.size(0)
            
            # 진행 상황 업데이트
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'accuracy': f'{100 * correct_predictions / total_predictions:.2f}%'
            })
        
        # 에포크 통계
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct_predictions / total_predictions
        
        # TensorBoard 로깅
        writer.add_scalar('Pretrain/Loss', avg_loss, epoch)
        writer.add_scalar('Pretrain/Accuracy', accuracy, epoch)
        
        print(f"\nEpoch {epoch+1} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # 체크포인트 저장
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = f'models/{model_complexity}/pretrain_checkpoints/model_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
        
        # 최고 성능 모델 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = f'models/{model_complexity}/pretrain_best_model.pth'
            torch.save({
                'episode': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'scores': scores,
                'epsilon': 0.1
            }, best_model_path)
    
    writer.close()
    print("\n사전 학습 완료!")
    return best_model_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--complexity', 
                       choices=['simple', 'medium', 'complex', 'transformer', 'tetris'],
                       default='simple',
                       help='모델 복잡도 선택')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='데이터 수집 에피소드 수')
    parser.add_argument('--epochs', type=int, default=50,
                       help='학습 에포크 수')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='배치 크기')
    args = parser.parse_args()
    
    best_model_path = pretrain(
        num_episodes=args.episodes,
        model_complexity=args.complexity,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
    print(f"최고 성능 모델 저장됨: {best_model_path}")