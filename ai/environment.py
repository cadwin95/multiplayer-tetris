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
        
        self.max_steps = 1000  # 최대 스텝 수 제한
        
        self.difficulty = 1.0
        self.auto_drop_counter = 0  # 자동 하강 카운터 추가
        self.action_counter = 0  # 액션 카운터 추가
        self.action_delay = 3    # 액션 사이의 지연 추가
        
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
        # 기본적으로 라인 클리어에만 집중
        if lines_cleared == 0:
            return 0  # 라인을 클리어하지 않으면 보상 없음
        
        # 라인 클리어 보상 (기하급수적 증가)
        line_rewards = {
            1: 100,    # 1줄: 기본 점수
            2: 300,    # 2줄: 3배
            3: 500,    # 3줄: 9배
            4: 700    # 테트리스: 27배
        }
        reward = line_rewards.get(lines_cleared, 0)
        
        # 게임 오버 페널티
        if self._is_game_over():
            reward -= 1000  # 게임 오버시 큰 페널티
        
        return reward
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        self.board = np.zeros((20, 10), dtype=np.float32)
        self.current_piece = self._get_random_piece()
        self.next_piece = self._get_random_piece()
        self.hold_piece = None
        self.can_hold = True
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        
        return self._get_state(), {}
    
    def _move_down(self) -> bool:
        """피스를 한 칸 아래로 이동. 성공하면 True, 실패하면 False 반환"""
        new_pos = [self.current_piece['position'][0], self.current_piece['position'][1] + 1]
        if not self._check_collision({**self.current_piece, 'position': new_pos}, self.board):
            self.current_piece['position'] = new_pos
            return True
        return False
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        reward = 0
        done = False
        
        # 최대 스텝 수 초과시 게임 종료
        self.steps += 1
        if self.steps >= self.max_steps:
            done = True
            reward -= 10
        
        # 난이도에 따른 자동 하강 (속도 감소)
        self.difficulty = min(2.0, 1.0 + self.steps / 1000)  # 500 -> 1000으로 변경
        self.auto_drop_counter += self.difficulty * 0.2  # 하강 속도 감소
        
        # 액션 카운터 증가
        self.action_counter += 1
        
        # 액션 지연이 지난 후에만 새로운 액션 처리
        if self.action_counter >= self.action_delay:
            self.action_counter = 0
            
            # 액션 처리
            if action == 5 and self.can_hold:  # hold
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
            elif action == 0:  # left
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
        
        # 자동 하강 처리
        if self.auto_drop_counter >= 1:
            self.auto_drop_counter = 0
            if not self._move_down():
                # 하강 실패시 피스 고정
                self._place_piece()
                lines_cleared = self._clear_lines()
                reward += self._calculate_reward(lines_cleared)
                self.lines_cleared += lines_cleared
                
                if self._is_game_over():
                    done = True
                    reward -= 1000
                else:
                    self.current_piece = self.next_piece
                    self.next_piece = self._get_random_piece()
        
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
