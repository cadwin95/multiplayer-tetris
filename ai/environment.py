import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, List
import random

class TetrisEnv(gym.Env):
    """테트리스 강화학습 환경"""
    
    def __init__(self):
        # 액션 스페이스 정의 (left, right, rotate, down, hardDrop, hold)
        self.action_space = spaces.Discrete(6)
        
        # 관찰 스페이스 정의 (보드 상태)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(20, 10), dtype=np.float32
        )
        
        # 피스 정의
        self.PIECES = {
            'I': [[[1, 1, 1, 1]], [[1], [1], [1], [1]]],
            'O': [[[1, 1], [1, 1]]],
            'T': [[[0, 1, 0], [1, 1, 1]], [[1, 0], [1, 1], [1, 0]]],
            'S': [[[0, 1, 1], [1, 1, 0]], [[1, 0], [1, 1], [0, 1]]],
            'Z': [[[1, 1, 0], [0, 1, 1]], [[0, 1], [1, 1], [1, 0]]],
            'J': [[[1, 0, 0], [1, 1, 1]], [[1, 1], [1, 0], [1, 0]]],
            'L': [[[0, 0, 1], [1, 1, 1]], [[1, 0], [1, 0], [1, 1]]]
        }
        
        # hold piece 관련 상태 추가
        self.hold_piece = None
        self.can_hold = True  # hold 사용 가능 여부
        
        self.reset()
    
    def _get_random_piece(self) -> Dict:
        piece_type = random.choice(list(self.PIECES.keys()))
        return {
            'type': piece_type,
            'rotation': 0,
            'position': [4, 0],  # 중앙 상단에서 시작
            'shape': self.PIECES[piece_type][0]
        }
    
    def _check_collision(self, piece: Dict, board: np.ndarray) -> bool:
        shape = piece['shape']
        pos = piece['position']
        
        for y in range(len(shape)):
            for x in range(len(shape[0])):
                if shape[y][x]:
                    board_y = pos[1] + y
                    board_x = pos[0] + x
                    
                    if (board_y >= 20 or board_x < 0 or board_x >= 10 or 
                        (board_y >= 0 and board[board_y][board_x])):
                        return True
        return False
    
    def _clear_lines(self) -> int:
        lines_cleared = 0
        y = 19
        while y >= 0:
            if np.all(self.board[y]):
                self.board[1:y+1] = self.board[0:y]
                self.board[0] = np.zeros(10)
                lines_cleared += 1
            else:
                y -= 1
        return lines_cleared
    
    def _calculate_reward(self, lines_cleared: int) -> float:
        # 게임 생존 보상 추가
        reward = 1  # 기본 생존 보상

        # 라인 클리어 보상 증가
        line_rewards = {1: 200, 2: 600, 3: 1000, 4: 2000}  # 보상 2배 증가
        reward += line_rewards.get(lines_cleared, 0)
        
        # 페널티 감소
        heights = [0] * 10
        for x in range(10):
            for y in range(20):
                if self.board[y][x]:
                    heights[x] = 20 - y
                    break
        
        max_height = max(heights)
        height_penalty = max_height   # 2에서 1로 감소
        
        # 구멍 페널티
        holes = 0
        for x in range(10):
            found_block = False
            for y in range(20):
                if self.board[y][x]:
                    found_block = True
                elif found_block:
                    holes += 1
        
        holes_penalty = holes * 5     # 10에서 5로 감소
        
        return reward - height_penalty - holes_penalty
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        self.board = np.zeros((20, 10), dtype=np.float32)
        self.current_piece = self._get_random_piece()
        self.next_piece = self._get_random_piece()
        self.hold_piece = None
        self.can_hold = True
        self.score = 0
        self.lines_cleared = 0
        
        return self._get_state(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        reward = 0
        done = False
        
        # hold piece 액션 추가
        if action == 5:  # hold
            if self.can_hold:
                if self.hold_piece is None:
                    self.hold_piece = {
                        'type': self.current_piece['type'],
                        'rotation': 0,
                        'position': [4, 0],
                        'shape': self.PIECES[self.current_piece['type']][0]
                    }
                    self.current_piece = self.next_piece
                    self.next_piece = self._get_random_piece()
                else:
                    self.current_piece, self.hold_piece = {
                        'type': self.hold_piece['type'],
                        'rotation': 0,
                        'position': [4, 0],
                        'shape': self.PIECES[self.hold_piece['type']][0]
                    }, {
                        'type': self.current_piece['type'],
                        'rotation': 0,
                        'position': [4, 0],
                        'shape': self.PIECES[self.current_piece['type']][0]
                    }
                self.can_hold = False
                return self._get_state(), 0, False, False, {
                    'score': self.score,
                    'lines': self.lines_cleared
                }
        
        # 액션 행
        if action == 0:  # left
            new_pos = [self.current_piece['position'][0] - 1, self.current_piece['position'][1]]
            if not self._check_collision({**self.current_piece, 'position': new_pos}, self.board):
                self.current_piece['position'] = new_pos
        
        elif action == 1:  # right
            new_pos = [self.current_piece['position'][0] + 1, self.current_piece['position'][1]]
            if not self._check_collision({**self.current_piece, 'position': new_pos}, self.board):
                self.current_piece['position'] = new_pos
        
        elif action == 2:  # rotate
            new_rotation = (self.current_piece['rotation'] + 1) % len(self.PIECES[self.current_piece['type']])
            new_shape = self.PIECES[self.current_piece['type']][new_rotation]
            if not self._check_collision({**self.current_piece, 'rotation': new_rotation, 'shape': new_shape}, self.board):
                self.current_piece['rotation'] = new_rotation
                self.current_piece['shape'] = new_shape
        
        elif action == 3:  # down
            new_pos = [self.current_piece['position'][0], self.current_piece['position'][1] + 1]
            if not self._check_collision({**self.current_piece, 'position': new_pos}, self.board):
                self.current_piece['position'] = new_pos
                reward += 1
            else:
                self._place_piece()
                lines_cleared = self._clear_lines()
                reward += self._calculate_reward(lines_cleared)
                self.lines_cleared += lines_cleared
                
                if self._is_game_over():
                    done = True
                    reward -= 50
                else:
                    self.current_piece = self.next_piece
                    self.next_piece = self._get_random_piece()
        
        elif action == 4:  # hardDrop
            while not self._check_collision({**self.current_piece, 
                'position': [self.current_piece['position'][0], self.current_piece['position'][1] + 1]}, self.board):
                self.current_piece['position'][1] += 1
                reward += 2
            
            self._place_piece()
            lines_cleared = self._clear_lines()
            reward += self._calculate_reward(lines_cleared)
            self.lines_cleared += lines_cleared
            
            if self._is_game_over():
                done = True
                reward -= 50
            else:
                self.current_piece = self.next_piece
                self.next_piece = self._get_random_piece()
        
        # 피스가 고정되면 hold 사용 가능하도록 초기화
        if action in [3, 4] and not self._check_collision({**self.current_piece, 
            'position': [self.current_piece['position'][0], self.current_piece['position'][1] + 1]}, self.board):
            self.can_hold = True
        
        return self._get_state(), reward, done, False, {
            'score': self.score,
            'lines': self.lines_cleared
        }
    
    def _place_piece(self):
        shape = self.current_piece['shape']
        pos = self.current_piece['position']
        
        for y in range(len(shape)):
            for x in range(len(shape[0])):
                if shape[y][x]:
                    self.board[pos[1] + y][pos[0] + x] = 1
    
    def _is_game_over(self) -> bool:
        return self._check_collision(self.current_piece, self.board)
    
    def _get_state(self) -> np.ndarray:
        # 현재 상태에 추가 정보를 포함
        board_state = self.board.copy()
        
        # 다음 피스 정보 추가
        next_piece_state = np.zeros((4, 4))
        next_shape = self.next_piece['shape']
        for y in range(len(next_shape)):
            for x in range(len(next_shape[0])):
                if next_shape[y][x]:
                    next_piece_state[y][x] = 1
        
        # hold 피스 정보 추가
        hold_piece_state = np.zeros((4, 4))
        if self.hold_piece:
            hold_shape = self.hold_piece['shape']
            for y in range(len(hold_shape)):
                for x in range(len(hold_shape[0])):
                    if hold_shape[y][x]:
                        hold_piece_state[y][x] = 1
        
        # 높이 프로필 계산
        heights = np.zeros(10)
        for x in range(10):
            for y in range(20):
                if board_state[y][x]:
                    heights[x] = 20 - y
                    break
        
        # 구멍 정보 계산
        holes = np.zeros(10)
        for x in range(10):
            found_block = False
            for y in range(20):
                if board_state[y][x]:
                    found_block = True
                elif found_block:
                    holes[x] += 1
        
        # 모든 정보를 하나의 상태로 결합
        return np.concatenate([
            board_state.flatten(),
            next_piece_state.flatten(),
            hold_piece_state.flatten(),  # hold 피스 상태 추가
            heights,
            holes,
            [float(self.can_hold)]  # hold 가능 여부 추가
        ])
