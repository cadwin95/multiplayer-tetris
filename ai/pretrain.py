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
import sys

class ExpertDataset(Dataset):
    def __init__(self, states, actions):
        self.states = torch.FloatTensor(states)
        self.actions = torch.tensor(actions, dtype=torch.long)
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

def collect_expert_data(num_episodes=1000, max_steps_per_episode=1000, render_interval=100):
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

def validate_data(states, actions, scores):
    """학습 데이터의 품질을 검증합니다."""
    print("\n=== 데이터 검증 ===")
    
    # 기본 통계
    print(f"총 데이터 수: {len(states):,}개")
    print(f"상태 데이터 형태: {states.shape}")
    print(f"액션 분포:")
    unique, counts = np.unique(actions, return_counts=True)
    action_names = ['left', 'right', 'rotate', 'down', 'hardDrop', 'hold']
    for action_id, count in zip(unique, counts):
        percentage = (count / len(actions)) * 100
        print(f"  - {action_names[action_id]}: {count:,}개 ({percentage:.1f}%)")
    
    # 점수 통계
    print(f"\n점수 통계:")
    print(f"  - 평균: {np.mean(scores):.1f}")
    print(f"  - 중앙값: {np.median(scores):.1f}")
    print(f"  - 최소: {np.min(scores):.1f}")
    print(f"  - 최대: {np.max(scores):.1f}")
    
    # 상태 데이터 검증
    print("\n상태 데이터 검증:")
    print(f"  - NaN 값: {'있음' if np.isnan(states).any() else '없음'}")
    print(f"  - 무한대 값: {'있음' if np.isinf(states).any() else '없음'}")
    print(f"  - 최소값: {states.min():.2f}")
    print(f"  - 최대값: {states.max():.2f}")
    
    # 데이터 샘플 출력
    print("\n데이터 샘플 (처음 3개):")
    for i in range(min(3, len(states))):
        print(f"\n샘플 {i+1}:")
        print(f"  액션: {action_names[actions[i]]}")
        print(f"  점수: {scores[i]}")
        # 보드 상태를 시각화 (첫 200개 값만)
        board = states[i][:200].reshape(20, 10)
        print("  보드 상태:")
        for row in board:
            print("  " + "".join(["□" if cell == 0 else "■" for cell in row]))

def calculate_action_quality(states, actions, env):
    """각 액션의 품질(reward)을 계산"""
    qualities = []
    for state, action in zip(states, actions):
        board = state[:200].cpu().numpy().reshape(20, 10)
        env.reset()
        env.board = board.copy()
        _, reward, done, _, _ = env.step(action)
        qualities.append(reward)
    # float32 데이터 타입으로 변환하여 반환
    return torch.tensor(qualities, device=states.device, dtype=torch.float32)

def pretrain(
    num_episodes: int = 1000,
    model_complexity: str = 'simple',
    batch_size: int = 64,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    save_interval: int = 5,
    use_db_data: bool = False  # DB 데이터 사용 여부
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")
    
    # DB 데이터 또는 전문가 데이터 수집
    if use_db_data:
        print("\n저장된 게임 히스토리 데이터 로드 중...")
        try:
            # 데이터 로드
            data_dir = './data'
            states = np.load(os.path.join(data_dir, 'X_train.npy'))
            actions = np.load(os.path.join(data_dir, 'y_train.npy'))
            print(f"저장된 데이터 로드 완료 - states: {states.shape}, actions: {actions.shape}")
            
            # 백업 파일이 있는지 확인
            backup_exists = all(
                os.path.exists(os.path.join(data_dir, f'{f}.backup'))
                for f in ['X_train.npy', 'y_train.npy']
            )
            
            # 데이터 유효성 검사
            if len(states) != len(actions) or len(states) != 13202:  # 예상되는 데이터 크기
                print(f"\n경고: 데이터 크기가 예상과 다릅니다!")
                if backup_exists:
                    response = input("백업 파일에서 복원하시겠습니까? (y/n): ")
                    if response.lower() == 'y':
                        states = np.load(os.path.join(data_dir, 'X_train.npy.backup'))
                        actions = np.load(os.path.join(data_dir, 'y_train.npy.backup'))
                        print(f"백업에서 복원된 데이터 - states: {states.shape}, actions: {actions.shape}")
            
            # 액션 분포 확인
            print("\n액션 분포:")
            action_counts = {i: 0 for i in range(6)}
            for action in actions:
                if action in action_counts:
                    action_counts[action] += 1
                else:
                    print(f"경고: 잘못된 액션 값 발견: {action}")
            
            action_names = ['left', 'right', 'rotate', 'down', 'hardDrop', 'hold']
            for action_id, count in action_counts.items():
                percentage = (count / len(actions)) * 100
                print(f"  - {action_names[action_id]}: {count}개 ({percentage:.1f}%)")
            
            # 점수 계산 (이전 코드와 동일)
            print("\n저장된 데이터의 점수 계산 중...")
            env = TetrisEnv()
            scores = []
            current_episode_score = 0
            current_board = None
            
            for i, (state, action) in enumerate(tqdm(zip(states, actions), total=len(states))):
                board = state[:200].reshape(20, 10)
                
                if current_board is None or not np.array_equal(current_board, board):
                    if current_board is not None:
                        scores.append(current_episode_score)
                    env.reset()
                    env.board = board.copy()
                    current_episode_score = 0
                
                _, reward, done, _, _ = env.step(action)
                current_episode_score += reward
                current_board = board.copy()
                
                if done:
                    scores.append(current_episode_score)
                    current_board = None
            
            if current_episode_score > 0:
                scores.append(current_episode_score)
            
            scores = np.array(scores)
            print(f"\n점수 통계:")
            print(f"  - 평균 점수: {np.mean(scores):.2f}")
            print(f"  - 최고 점수: {np.max(scores):.2f}")
            print(f"  - 최저 점수: {np.min(scores):.2f}")
            print(f"  - 에피소드 수: {len(scores)}")
            
        except FileNotFoundError:
            print("오류: 저장된 데이터 파일을 찾을 수 없습니다.")
            print("먼저 fetch_game_history.py를 실행하여 데이터를 생성해주세요.")
            sys.exit(1)
    else:
        print("\n전문가 데이터 수집 시작...")
        states, actions, scores = collect_expert_data(num_episodes)
    
    # 데이터 검증
    validate_data(states, actions, scores)
    
    # 데이터 저장
    os.makedirs('./ai/data', exist_ok=True)
    np.save('./ai/data/X_train.npy', states)
    np.save('./ai/data/y_train.npy', actions)
    np.save('./ai/data/scores.npy', scores)
    
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
    criterion = nn.CrossEntropyLoss(reduction='none')  # 개별 손실값 유지
    
    # 로깅 설정
    log_dir = f'./ai/runs/pretrain_{model_complexity}_{int(time.time())}'
    writer = SummaryWriter(log_dir)
    
    # 체크포인트 디렉토리 생성
    os.makedirs(f'./ai/models/{model_complexity}/pretrain_checkpoints', exist_ok=True)
    
    # 학습 루프
    print("\n사전 학습 시작...")
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for states, actions in dataloader:
            states = states.to(device)
            actions = actions.to(device)
            
            # 각 액션의 품질 계산
            action_qualities = calculate_action_quality(states, actions, env)
            
            # 순전파
            predictions = model(states)
            base_loss = criterion(predictions, actions)
            
            # reward 부호에 따라 다른 학습 방향 적용
            positive_mask = action_qualities > 0
            negative_mask = action_qualities <= 0
            
            # 긍정적인 reward: 해당 액션을 더 선택하도록
            # 부정적인 reward: 해당 액션을 덜 선택하도록 (reverse loss)
            loss = torch.zeros_like(base_loss)
            loss[positive_mask] = base_loss[positive_mask] * action_qualities[positive_mask]
            loss[negative_mask] = -base_loss[negative_mask] * action_qualities[negative_mask]
            
            loss = loss.mean()
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 학습 상태 출력
            if len(positive_mask) > 0:
                pos_ratio = positive_mask.float().mean().item()
                print(f"Positive actions: {pos_ratio:.2%}, "
                      f"Avg positive quality: {action_qualities[positive_mask].mean():.2f}, "
                      f"Avg negative quality: {action_qualities[negative_mask].mean():.2f}")
        
        # 에포크 통계
        avg_loss = total_loss / len(dataloader)
        
        # TensorBoard 로깅
        writer.add_scalar('Pretrain/Loss', avg_loss, epoch)
        
        print(f"\nEpoch {epoch+1} - Loss: {avg_loss:.4f}")
        
        # 체크포인트 저장
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = f'./ai/models/{model_complexity}/pretrain_checkpoints/model_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
        
        # 최고 성능 모델 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = f'./ai/models/{model_complexity}/pretrain_best_model.pth'
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
    parser.add_argument('--use-db-data', action='store_true',
                       help='DB 데이터 사용 여부')
    args = parser.parse_args()
    
    best_model_path = pretrain(
        num_episodes=args.episodes,
        model_complexity=args.complexity,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        use_db_data=args.use_db_data
    )
    print(f"최고 성능 모델 저장됨: {best_model_path}")