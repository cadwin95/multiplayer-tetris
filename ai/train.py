import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from environment import TetrisEnv
from model import create_model, count_parameters

class PrioritizedReplayBuffer:
    """우선순위가 있는 리플레이 버퍼"""
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha  # 우선순위 지수
        self.beta = beta    # 중요도 샘플링 지수
        self.epsilon = 1e-6  # 0으로 나누기 방지
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size):
        buffer_len = len(self.buffer)
        if buffer_len == 0:
            return None, None
        
        # 우선순위를 확률로 변환
        priorities = np.array(self.priorities, dtype=np.float64)
        priorities = np.clip(priorities, self.epsilon, None)  # 0 방지
        
        # alpha로 우선순위 조정
        probs = priorities ** self.alpha
        probs = probs / np.sum(probs)  # 정규화
        
        # NaN 체크 및 처리
        if np.isnan(probs).any():
            probs = np.ones(len(probs)) / len(probs)
        
        # 샘플링
        indices = np.random.choice(buffer_len, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        return samples, indices
    
    def update_priorities(self, indices, new_priorities):
        for idx, priority in zip(indices, new_priorities):
            priority = max(self.epsilon, float(priority))  # NaN 및 음수 방지
            self.priorities[idx] = min(priority, 1e6)  # 너무 큰 값 방지
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)

def plot_training_progress(scores, window_sizes=[10, 50, 100], filename='training_progress.png'):
    """학습 진행 상황 시각화"""
    plt.figure(figsize=(12, 8))
    
    # 개별 점수
    plt.plot(scores, alpha=0.3, label='Score', color='gray')
    
    # 이동 평균선
    for window in window_sizes:
        moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
        plt.plot(moving_avg, label=f'MA-{window}')
    
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 통계 정보 추가
    stats_text = (
        f'Total Episodes: {len(scores)}\n'
        f'Best Score: {max(scores):.0f}\n'
        f'Recent Avg (100): {np.mean(scores[-100:]):.1f}\n'
        f'Recent Max (100): {max(scores[-100:]):.0f}'
    )
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', fontfamily='monospace')
    
    plt.savefig(filename)
    plt.close()

def get_model_paths(complexity: str):
    """모델 복잡도별 저장 경로 정의"""
    base_dir = f'models/{complexity}'
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(f'{base_dir}/checkpoints', exist_ok=True)
    os.makedirs(f'{base_dir}/plots', exist_ok=True)
    os.makedirs(f'{base_dir}/plots/checkpoints', exist_ok=True)
    
    return {
        'model': f'{base_dir}/tetris_model.pth',
        'checkpoint': f'{base_dir}/checkpoints/tetris_model_ep{{episode}}.pth',
        'plot': f'{base_dir}/plots/checkpoints/progress_ep{{episode}}.png',
        'final_plot': f'{base_dir}/plots/training_progress.png'
    }

def render_best_episode(env, model, device, save_dir='best_episode', save_interval=5, max_steps=1000, num_tries=5):
    """최고 점수 에피소드 시각화 (여러 번 시도)"""
    os.makedirs(save_dir, exist_ok=True)
    
    best_score = float('-inf')
    best_frames = None
    best_actions = None
    
    print("\nTrying multiple episodes to find the best one to render...")
    
    # 여러 번 시도하여 가장 좋은 에피소드 찾기
    for try_num in range(num_tries):
        frames = []
        actions = []
        score = 0
        step = 0
        
        state, _ = env.reset()
        
        while True:
            if step >= max_steps:
                break
            
            # 현재 상태 저장 (액션 실행 전)
            current_board = env.board.copy()
            current_piece = env.current_piece.copy()
            
            # 액션 선택 및 실행
            valid_actions = env.get_valid_actions()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = model(state_tensor)
                
                # 유효한 액션에 대한 Q-value만 고려
                masked_q_values = q_values.clone()
                masked_q_values[0] = float('-inf')
                masked_q_values[0, valid_actions] = q_values[0, valid_actions]
                action = masked_q_values[0].argmax().item()
                
                print(f"\nStep {step}:")
                print(f"Q-values: {q_values.squeeze().tolist()}")
                print(f"Selected action: {['Left', 'Right', 'Rotate','Down', 'HardDrop', 'Hold'][action]}")
            
            next_state, reward, done, _, _ = env.step(action)
            print(f"Reward: {reward}")
            score += reward
            
            # 프레임 저장
            display_board = current_board.copy()
            piece_shape = current_piece['shape']
            pos = current_piece['position']
            
            for y, row in enumerate(piece_shape):
                for x, cell in enumerate(row):
                    if cell:
                        board_y = pos[1] + y
                        board_x = pos[0] + x
                        if 0 <= board_y < 20 and 0 <= board_x < 10:
                            display_board[board_y][board_x] = 2
            
            if step % save_interval == 0:
                frames.append({
                    'board': display_board,
                    'score': score,
                    'action': action,
                    'step': step,
                    'q_values': q_values.squeeze().tolist(),
                    'valid_actions': valid_actions
                })
            
            actions.append(action)
            state = next_state
            step += 1
            
            if done:
                break
        
        print(f"Try {try_num + 1}/{num_tries}: Score = {score:.1f}, Steps = {step}")
        
        if score > best_score:
            best_score = score
            best_frames = frames.copy()
            best_actions = actions.copy()
    
    print(f"\nSelected episode with score: {best_score:.1f}")
    
    if best_frames is None:
        print("No valid episode found!")
        return best_score, []
    
    # 이미지 생성 및 저장
    print(f"\nGenerating {len(best_frames)} images...")
    figs = []
    for i, frame in enumerate(best_frames):
        print(f"\rGenerating image {i+1}/{len(best_frames)}", end="")
        fig = plt.figure(figsize=(12, 8))
        
        # 보드 상태 표시 (왼쪽)
        plt.subplot(1, 2, 1)
        cmap = plt.cm.colors.ListedColormap(['white', 'gray', 'red'])
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        plt.imshow(frame['board'], cmap=cmap, norm=norm)
        plt.title(f'Step: {frame["step"]}\nScore: {frame["score"]}')
        plt.grid(True)
        
        # Q-value 바 차트 (오른쪽)
        plt.subplot(1, 2, 2)
        action_names = ['Left', 'Right', 'Rotate', 'Down', 'HardDrop', 'Hold']
        q_values = frame['q_values']
        valid_actions = frame['valid_actions']
        
        colors = ['lightgray'] * len(q_values)
        for valid_action in valid_actions:
            colors[valid_action] = 'blue'
        colors[frame['action']] = 'red'
        
        plt.bar(action_names, q_values, color=colors)
        plt.title('Q-values by Action\n(Blue: Valid, Red: Selected)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        figs.append(fig)
    
    # 한 번에 저장
    for i, fig in enumerate(figs):
        print(f"\rSaving image {i+1}/{len(figs)}", end="")
        fig.savefig(f'{save_dir}/step_{i:03d}.png')
        plt.close(fig)
    
    # GIF 생성
    try:
        import imageio
        print("\n\nCreating GIF animation...")
        images = []
        for i in range(len(best_frames)):
            print(f"\rLoading image {i+1}/{len(best_frames)} for GIF", end="")
            images.append(imageio.imread(f'{save_dir}/step_{i:03d}.png'))
        print("\nSaving GIF...", end="")
        imageio.mimsave(f'{save_dir}/game.gif', images, duration=0.5)
        print(" Done!")
    except ImportError:
        print("\nimageio not found. Skipping animation creation.")
    
    return best_score, best_actions

def evaluate_model(env, model, device, num_episodes=5):
    """모델 평가 (시각화 없이)"""
    model.eval()  # 평가 모드로 설정
    scores = []
    
    with torch.no_grad():  # gradient 계산 비활성화
        for _ in range(num_episodes):
            state, _ = env.reset()
            score = 0
            done = False
            
            while not done:
                valid_actions = env.get_valid_actions()
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = model(state_tensor)
                
                # 유효한 액션에 대해서만 Q-value 고려
                masked_q_values = q_values.clone()
                masked_q_values[0] = float('-inf')
                masked_q_values[0, valid_actions] = q_values[0, valid_actions]
                action = masked_q_values[0].argmax().item()
                
                next_state, reward, done, _, _ = env.step(action)
                score += reward
                state = next_state
            
            scores.append(score)
    
    model.train()  # 다시 학습 모드로 설정
    return np.mean(scores), np.max(scores)

def train(
    num_episodes=10000,
    batch_size=64,
    gamma=0.95,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay=0.9999,
    learning_rate=5e-5,
    memory_size=50000,
    target_update=100,
    save_interval=1000,
    plot_interval=100,
    model_complexity='simple',
    resume_from=None,
    pretrained_model=None,
    min_memory_size=1000,
    update_frequency=4,
    render_interval=5000,
    eval_interval=1000
):
    # 모델 복잡도별 경로 설정
    paths = get_model_paths(model_complexity)
    
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")
    
    # 환경과 모델 초기화
    env = TetrisEnv()
    policy_net = create_model(model_complexity).to(device)
    target_net = create_model(model_complexity).to(device)
    
    # pretrained 모델 로드 (있는 경우)
    if pretrained_model and os.path.exists(pretrained_model):
        print(f"\nPretrained 모델 로드 중: {pretrained_model}")
        checkpoint = torch.load(pretrained_model, map_location=device)
        policy_net.load_state_dict(checkpoint['model_state_dict'])
        target_net.load_state_dict(checkpoint['model_state_dict'])
        print("Pretrained 모델 로드 완료")
        
        # pretrained 모델의 성능 평가
        print("\nPretrained 모델 성능 평가 중...")
        mean_score, max_score = evaluate_model(env, policy_net, device)
        print(f"Pretrained 모델 평균 점수: {mean_score:.2f}")
        print(f"Pretrained 모델 최고 점수: {max_score:.2f}\n")
    
    # 모델 파라미터 수 출력
    num_params = count_parameters(policy_net)
    print(f"\nModel Architecture: {model_complexity}")
    print(f"Total trainable parameters: {num_params:,}")
    print(f"Model size: {num_params * 4 / (1024*1024):.2f} MB\n")
    
    # 옵티마이저 초기화
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    
    # 체크포인트에서 복원
    start_episode = 0
    scores = []
    epsilon = epsilon_start
    
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        
        policy_net.load_state_dict(checkpoint['model_state_dict'])
        target_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
        
        start_episode = checkpoint.get('episode', 0) + 1
        epsilon = checkpoint.get('epsilon', epsilon_start)
        scores = checkpoint.get('scores', [])
        
        print(f"Resumed from episode {start_episode}")
        print(f"Current epsilon: {epsilon:.3f}")
        if scores:
            print(f"Current average score (100): {np.mean(scores[-100:]):.2f}")
    
    memory = PrioritizedReplayBuffer(memory_size)
    
    def process_batch(batch):
        """배치 데이터 처리 최적화"""
        # numpy 배열로 먼저 변환하여 속도 향상
        states = np.array([s for s, _, _, _, _ in batch])
        actions = np.array([a for _, a, _, _, _ in batch])
        rewards = np.array([r for _, _, r, _, _ in batch])
        next_states = np.array([s for _, _, _, s, _ in batch])
        dones = np.array([d for _, _, _, _, d in batch])
        
        # 상태가 올바른 차원(607)인지 확인
        assert states.shape[1] == 607, f"Expected state dimension 607, got {states.shape[1]}"
        
        # 한 번에 텐서로 변환
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        return states, actions, rewards, next_states, dones
    
    # 초기 랜덤 경험 수집
    print("Collecting initial experiences...")
    state, _ = env.reset()
    for _ in range(min_memory_size):
        valid_actions = env.get_valid_actions()
        action = random.choice(valid_actions)
        next_state, reward, done, _, _ = env.step(action)
        memory.push(state, action, reward, next_state, done)
        
        if done:
            state, _ = env.reset()
        else:
            state = next_state
    
    print(f"Initial memory size: {len(memory)}")
    
    # 성능 모니터링을 위한 변수들
    scores = []
    recent_scores = deque(maxlen=100)  # 최근 100개 점수만 저장
    best_score = float('-inf')
    best_model_state = None
    total_steps = 0
    training_start_time = time.time()
    
    # 로깅 초기화
    log_dir = f'runs/tetris_{model_complexity}_{int(time.time())}'
    writer = SummaryWriter(log_dir)
    
    # CSV 로깅 설정
    import csv
    import datetime
    log_file = os.path.join(log_dir, 'training_log.csv')
    with open(log_file, 'w', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(['Episode', 'Score', 'Steps', 'Epsilon', 'Time'])
    
    # 학습 상태 추적
    train_stats = {
        'episode_rewards': [],
        'eval_scores': [],
        'losses': [],
        'epsilon_values': []
    }
    
    for episode in tqdm(range(start_episode, num_episodes)):
        policy_net.train()  # 학습 모드 설정
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = []
        
        while True:
            valid_actions = env.get_valid_actions()
            
            # epsilon-greedy 전략
            if random.random() < epsilon:
                action = random.choice(valid_actions)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = policy_net(state_tensor)
                    masked_q_values = q_values.clone()
                    masked_q_values[0] = float('-inf')
                    masked_q_values[0, valid_actions] = q_values[0, valid_actions]
                    action = masked_q_values[0].argmax().item()
            
            # 환경과 상호작용
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            
            # 경험 저장
            memory.push(state, action, reward, next_state, done)
            
            # 배치 학습
            if len(memory) > batch_size and total_steps % update_frequency == 0:
                batch, indices = memory.sample(batch_size)
                if batch:
                    # 배치 데이터 준비
                    states, actions, rewards, next_states, dones = process_batch(batch)
                    
                    # 현재 Q-value 계산
                    current_q = policy_net(states)
                    current_q = current_q.gather(1, actions.unsqueeze(1))
                    
                    # 다음 상태의 Q-value 계산
                    with torch.no_grad():
                        next_q = target_net(next_states).max(1)[0].unsqueeze(1)
                        target_q = rewards + gamma * next_q * (1 - dones)
                    
                    # 손실 계산 및 역전파
                    loss = F.smooth_l1_loss(current_q, target_q)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    # TD 오차 계산 및 우선순위 업데이트
                    td_errors = torch.abs(current_q - target_q).detach().cpu().numpy()
                    memory.update_priorities(indices, td_errors.flatten() + memory.epsilon)
                    
                    episode_loss.append(loss.item())
            
            state = next_state
            total_steps += 1
            
            if done:
                break
        
        # 에피소드 통계 저장
        train_stats['episode_rewards'].append(episode_reward)
        train_stats['epsilon_values'].append(epsilon)
        if episode_loss:
            train_stats['losses'].append(np.mean(episode_loss))
        
        # 주기적인 평가 수행
        if episode % eval_interval == 0:
            mean_score, max_score = evaluate_model(env, policy_net, device)
            train_stats['eval_scores'].append(mean_score)
            
            writer.add_scalar('Evaluation/Mean Score', mean_score, episode)
            writer.add_scalar('Evaluation/Max Score', max_score, episode)
        
        # 로깅
        if episode % plot_interval == 0:
            writer.add_scalar('Training/Reward', episode_reward, episode)
            writer.add_scalar('Training/Epsilon', epsilon, episode)
            if episode_loss:
                writer.add_scalar('Training/Loss', np.mean(episode_loss), episode)
        
        # 타겟 네트워크 업데이트
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # 입실론 감소
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # 체크포인트 저장
        if episode % save_interval == 0:
            checkpoint_path = paths['checkpoint'].format(episode=episode)
            save_checkpoint(
                policy_net, optimizer, episode, epsilon,
                train_stats, checkpoint_path
            )
    
    # 학습 완료 후 최종 결과 저장
    save_final_results(
        policy_net, optimizer, num_episodes, epsilon,
        train_stats, log_dir
    )
    
    # 학습 완료 후 최고의 게임 플레이 기록
    print("\n학습 완료! 최고의 게임 플레이를 기록합니다...")
    policy_net.eval()
    render_dir = os.path.join(log_dir, 'final_gameplay')
    final_score, _ = render_best_episode(
        env,
        policy_net,
        device,
        save_dir=render_dir
    )
    print(f"최종 게임 점수: {final_score}")
    
    writer.close()
    return train_stats

def save_checkpoint(model, optimizer, episode, epsilon, stats, path):
    """체크포인트 저장"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode': episode,
        'epsilon': epsilon,
        'stats': stats
    }, path)

def save_final_results(model, optimizer, episodes, epsilon, stats, log_dir):
    """최종 결과 저장"""
    # 모델 저장
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episodes': episodes,
        'epsilon': epsilon,
        'stats': stats
    }, os.path.join(log_dir, 'final_model.pth'))
    
    # 학습 그래프 생성
    plt.figure(figsize=(15, 5))
    
    # 보상 그래프
    plt.subplot(131)
    plt.plot(stats['episode_rewards'])
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # 손실 그래프
    plt.subplot(132)
    plt.plot(stats['losses'])
    plt.title('Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    # 평가 점수 그래프
    plt.subplot(133)
    plt.plot(stats['eval_scores'])
    plt.title('Evaluation Scores')
    plt.xlabel('Evaluation')
    plt.ylabel('Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'training_results.png'))
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--complexity', 
                       choices=['simple', 'medium', 'complex', 'transformer','tetris'], 
                       default='simple', 
                       help='모델 복잡도 선택')
    parser.add_argument('--episodes', type=int, default=10000, 
                       help='학습 에피소드 수')
    parser.add_argument('--resume', type=str,
                       help='체크포인트 파일 경로 (예: models/simple/checkpoints/tetris_model_ep1000.pth)')
    parser.add_argument('--pretrained', type=str,
                       help='Pretrained 모델 파일 경로 (예: models/simple/pretrain_best_model.pth)')
    args = parser.parse_args()
    
    train(
        num_episodes=args.episodes,
        model_complexity=args.complexity,
        resume_from=args.resume,
        pretrained_model=args.pretrained
    ) 