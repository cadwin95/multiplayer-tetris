import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from environment import TetrisEnv
from model import AlphaZeroPolicyValueNet
import argparse
import os
from train import MCTSNode, run_mcts

class GameVisualizer:
    def __init__(self, model_path, device='cpu', save_dir='game_visualization'):
        self.env = TetrisEnv()
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 모델 로드 (41개 액션으로 변경)
        self.model = AlphaZeroPolicyValueNet(obs_dim=214, num_actions=41)
        
        # 안전한 모델 로드
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA를 사용할 수 없어 CPU를 사용합니다.")
            device = 'cpu'
        elif device == 'mps' and not torch.backends.mps.is_available():
            print("MPS를 사용할 수 없어 CPU를 사용합니다.")
            device = 'cpu'
            
        try:
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint)
            print(f"모델을 {device} 디바이스에 성공적으로 로드했습니다.")
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {e}")
            print("CPU를 사용하여 다시 시도합니다.")
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint)
            device = 'cpu'
            
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # 시각화를 위한 색상 설정
        self.colors = {
            0: 'white',   # 빈 칸
            1: 'gray',    # 고정된 블록
            2: 'red',     # 현재 블록
            3: 'green'    # 다음 블록
        }
        
        
    
    def get_action_from_mcts(self, state, num_simulations=20, max_depth=2):
        """MCTS를 사용하여 다음 행동 선택"""
        root = MCTSNode(state=state)
        actions, pi = run_mcts(root, self.env, self.model,
                             simulations=num_simulations,
                             max_depth=max_depth,
                             device=self.device)
        
        # 가장 높은 확률의 액션 선택
        best_idx = np.argmax(pi)
        return actions[best_idx]
    
    def render_board(self, board, current_piece=None):
        """게임 보드 시각화 (고스트 피스 제거)"""
        display_board = board.copy()
        
        # 현재 피스 표시
        if current_piece is not None:
            piece_shape = self.env.PIECES[current_piece['type']][0]  # 기본 회전 상태 사용
            pos = (0, 0)  # 기본 위치 사용
            for y, row in enumerate(piece_shape):
                for x, cell in enumerate(row):
                    if cell:
                        board_y = pos[1] + y
                        board_x = pos[0] + x
                        if 0 <= board_y < 20 and 0 <= board_x < 10:
                            display_board[board_y][board_x] = 2
        
        return display_board
    
    def play_game(self, num_simulations=20, max_depth=2, save_interval=1):
        """게임 플레이 및 시각화 (고스트 피스 제거)"""
        frames = []
        state = self.env.reset()
        done = False
        total_reward = 0
        step = 0
        
        while not done and step < 1000:  # 최대 1000스텝
            # 현재 상태 저장
            current_board = self.env.board.copy()
            current_piece = self.env.current_piece.copy()
            
            # MCTS로 액션 선택
            action = self.get_action_from_mcts(state, num_simulations, max_depth)
            
            # 프레임 저장
            if step % save_interval == 0:
                display_board = self.render_board(
                    current_board,
                    current_piece
                )
                frames.append({
                    'board': display_board,
                    'score': total_reward,
                    'action': action,
                    'step': step
                })
            
            # 액션 실행
            next_state, reward, done, _, _ = self.env.step(action)
            total_reward += reward
            state = next_state
            step += 1
        
        # 마지막 상태 저장
        display_board = self.render_board(
            self.env.board,
            self.env.current_piece
        )
        frames.append({
            'board': display_board,
            'score': total_reward,
            'action': None,
            'step': step
        })
        
        return frames, total_reward
    
    def save_visualization(self, frames, fps=2):
        """게임 플레이 시각화 저장"""
        print(f"\n게임 시각화 생성 중... ({len(frames)} 프레임)")
        
        # 액션 이름 업데이트 (hold 액션 추가)
        action_names = {
            **{i: f'Rot{i//10} X{i%10}' for i in range(40)},  # 기본 40개 액션
            40: 'Hold'  # hold 액션
        }
        
        # 이미지 생성
        figs = []
        for i, frame in enumerate(frames):
            print(f"\r프레임 생성 중: {i+1}/{len(frames)}", end="")
            
            fig = plt.figure(figsize=(12, 8))
            
            # 보드 상태 표시 (왼쪽)
            plt.subplot(1, 2, 1)
            plt.imshow(frame['board'], cmap='viridis')
            plt.title(f'Step: {frame["step"]}\nScore: {frame["score"]:.0f}')
            plt.grid(True)
            
            # 액션 정보 표시 (오른쪽)
            plt.subplot(1, 2, 2)
            if frame['action'] is not None:
                plt.text(0.5, 0.5, f'Action: {action_names[frame["action"]]}',
                        ha='center', va='center', fontsize=12)
            plt.axis('off')
            
            plt.tight_layout()
            figs.append(fig)
            
            # 개별 이미지 저장
            plt.savefig(f'{self.save_dir}/step_{i:03d}.png')
            plt.close(fig)
        
        # GIF 생성
        try:
            import imageio
            print("\nGIF 애니메이션 생성 중...")
            images = []
            for i in range(len(frames)):
                images.append(imageio.imread(f'{self.save_dir}/step_{i:03d}.png'))
            imageio.mimsave(f'{self.save_dir}/game.gif', images, fps=fps)
            print("GIF 생성 완료!")
        except ImportError:
            print("\nimageio를 찾을 수 없습니다. GIF 생성을 건너뜁니다.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True,
                       help='학습된 모델 파일 경로')
    parser.add_argument('--device', type=str, default='cpu',
                       help='실행할 디바이스 (cpu/cuda/mps)')
    parser.add_argument('--simulations', type=int, default=20,
                       help='MCTS 시뮬레이션 횟수')
    parser.add_argument('--depth', type=int, default=2,
                       help='MCTS 최대 깊이')
    parser.add_argument('--save-dir', type=str, default='game_visualization',
                       help='시각화 저장 디렉토리')
    parser.add_argument('--fps', type=int, default=2,
                       help='GIF 애니메이션 FPS')
    args = parser.parse_args()
    
    # 게임 시각화
    visualizer = GameVisualizer(
        model_path=args.model_path,
        device=args.device,
        save_dir=args.save_dir
    )
    
    frames, final_score = visualizer.play_game(
        num_simulations=args.simulations,
        max_depth=args.depth
    )
    
    print(f"\n게임 완료! 최종 점수: {final_score:.0f}")
    
    # 시각화 저장
    visualizer.save_visualization(frames, fps=args.fps)

if __name__ == "__main__":
    main() 