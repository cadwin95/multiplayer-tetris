import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import signal
import sys

from environment import TetrisEnv
from model import DQN
from utils import ReplayBuffer, prepare_training_data, save_model

def train(
    num_episodes: int = 10000,
    batch_size: int = 32,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.1,
    epsilon_decay: float = 0.995,
    learning_rate: float = 3e-4,
    target_update: int = 5,
    memory_size: int = 50000,
    save_path: str = 'tetris_model.pth',
    checkpoint_interval: int = 1000
):
    # GPU 사용 가능 여부 확인 및 출력
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Ctrl+C 시그널 핸들러 설정
    def signal_handler(sig, frame):
        print('\nTraining interrupted. Saving model...')
        save_model(policy_net, save_path)
        writer.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    env = TetrisEnv()
    
    # 네트워크 초기화
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    # 옵티마이저와 메모리 초기화
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    memory = ReplayBuffer(memory_size)
    
    # 텐서보드 설정
    writer = SummaryWriter('runs/tetris_training')
    
    # 학습 메트릭스
    scores = []
    epsilon = epsilon_start
    
    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset()
        score = 0
        
        while True:
            # 액션 선택 (입실론-그리디)
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).to(device)
                    if len(state_tensor.shape) == 1:
                        state_tensor = state_tensor.unsqueeze(0)  # 배치 차원 추가
                    action = policy_net(state_tensor).max(1)[1].item()
            
            # 환경과 상호작용
            next_state, reward, done, _, info = env.step(action)
            score += reward
            
            # 메모리에 저장
            memory.push(state, action, reward, next_state, done)
            state = next_state
            
            # 배치 학습
            if len(memory) > batch_size:
                batch = memory.sample(batch_size)
                states, actions, rewards, next_states, dones = prepare_training_data(batch, device)
                
                # Q-값 계산
                current_q = policy_net(states).gather(1, actions.unsqueeze(1))
                next_q = target_net(next_states).max(1)[0].detach()
                target_q = rewards + (gamma * next_q * (1 - dones))
                
                # 손실 계산 및 최적화
                loss = nn.MSELoss()(current_q.squeeze(), target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 텐서보드에 기록
                writer.add_scalar('Loss', loss.item(), episode)
            
            if done:
                break
        
        # 점수 기록
        scores.append(score)
        writer.add_scalar('Score', score, episode)
        
        # 타겟 네트워크 업데이트
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # 입실론 감소
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # 진행상황 출력
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f'Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {epsilon:.2f}')
        
        # 주기적으로 체크포인트 저장
        if episode % checkpoint_interval == 0:
            checkpoint_path = f'tetris_model_ep{episode}.pth'
            save_model(policy_net, checkpoint_path)
            print(f'\nSaved checkpoint to {checkpoint_path}')
            
            # 학습 그래프도 중간중간 저장
            plt.figure(figsize=(10, 5))
            plt.plot(scores)
            plt.title(f'Training Progress (Episode {episode})')
            plt.xlabel('Episode')
            plt.ylabel('Score')
            plt.savefig(f'training_progress_ep{episode}.png')
            plt.close()
    
    # 모델 저장
    save_model(policy_net, save_path)
    
    # 학습 결과 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(scores)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.savefig('training_progress.png')
    plt.close()

if __name__ == "__main__":
    train()
