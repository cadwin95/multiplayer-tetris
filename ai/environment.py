import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, List
import random

class TetrisEnv(gym.Env):
    """테트리스 강화학습 환경"""
    
    def __init__(self):
        # 액션 스페이스 변경: 
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
        
        # hold piece 관찰 상태 추가
        self.hold_piece = None
        self.can_hold = True  # hold 사용 가능 여부
        
        self.max_pieces = 200  # 최대 블록 수로 변경
        self.pieces_placed = 0  # 배치된 블록 수 추적
        
        self.action_counter = 0  # 액션 카운터 추가
        
        self.last_actions = []  # 최근 행동 기록
        self.last_heights = []  # 최근 높이 기록
        self.oscillation_penalty = 0  # 좌우 진동 페널티
        self.last_clear = 0  # 연속 라인 클리어 추적용
        
        # 현재 블록에 대한 행동 제한 수정
        self.moves_left = 8  # 좌우 이동 가능 횟수 증가 (3 -> 8)
        self.rotates_left = 4  # 회전 가능 횟수 증가 (2 -> 4)
        self.must_drop = False
        
        self.recent_actions = []  # 최근 행동 기록 추가
        self.repetitive_penalty = -0.1  # 반복 행동에 대한 페널티 값
        
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
            if sum(self.board[y]) == 0:
                break
            if np.all(self.board[y]):
                self.board[1:y+1] = self.board[0:y]
                self.board[0] = np.zeros(10)
                lines_cleared += 1
            else:
                y -= 1
        return lines_cleared
    
    def _calculate_reward(self, lines_cleared: int) -> float:
        """매우 단순한 보상 체계: 오직 라인 클리어와 게임 오버만 고려"""
        if self._is_game_over():
            return -2  # 게임 오버 페널티
        
        if lines_cleared > 0:
            # 라인 클리어에 대해서만 큰 양의 보상
            return {
                1: 100,
                2: 400,
                3: 900,
                4: 1600
            }[lines_cleared]
        
        return 0  # 다른 �����든 경우에는 보상 없음
    
    def _count_holes(self) -> int:
        """보드에 있는 구멍의 수를 계산"""
        holes = 0
        for col in range(10):
            block_found = False
            for row in range(20):
                if self.board[row][col]:
                    block_found = True
                elif block_found:
                    holes += 1
        return holes
    
    def _calculate_bumpiness(self) -> float:
        """보드 표면의 울퉁불퉁한 정도를 계산"""
        heights = []
        for col in range(10):
            for row in range(20):
                if self.board[row][col]:
                    heights.append(20 - row)
                    break
            else:
                heights.append(0)
        
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        return bumpiness
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        self.board = np.zeros((20, 10), dtype=np.float32)
        self.current_piece = self._get_random_piece()
        self.next_piece = self._get_random_piece()
        self.hold_piece = None
        self.can_hold = True
        self.score = 0
        self.lines_cleared = 0
        self.pieces_placed = 0  # 블록 카운트 초기화
        self._reset_move_limits()  # 이동/회전 제한 초기화
        self.recent_actions = []  # 행동 기록 초기화
        return self._get_state(), {}
    
    def _evaluate_drop_position(self) -> float:
        """하드드롭 위치의 품질을 평가"""
        reward = 0
        
        # 1. 구멍 생성 페널티
        holes_before = self._count_holes()
        # 임시로 피스를 놓아보고 구멍 수 계산
        temp_board = self.board.copy()
        shape = self.current_piece['shape']
        pos = self.current_piece['position']
        for y in range(len(shape)):
            for x in range(len(shape[0])):
                if shape[y][x]:
                    temp_board[pos[1] + y][pos[0] + x] = 1
        holes_after = self._count_holes()
        reward -= (holes_after - holes_before) * 30  # 새로운 구멍당 -30
        
        # 2. 높이 기반 페널티
        max_height = self._get_max_height()
        if max_height > 15:  # 높이가 15 이상이면 페널티
            reward -= (max_height - 15) * 20
        
        # 3. 울퉁불퉁함 페널티
        bumpiness = self._calculate_bumpiness()
        reward -= bumpiness * 2
        
        # 4. 접촉면 보너스 (다른 블록이나 바닥과 맞닿은 면적)
        contacts = self._count_contacts()
        reward += contacts * 10
        
        return reward

    def _count_contacts(self) -> int:
        """현재 피스의 접촉면 수를 계산"""
        contacts = 0
        shape = self.current_piece['shape']
        pos = self.current_piece['position']
        
        # 각 블록 셀에 대해 검사
        for y in range(len(shape)):
            for x in range(len(shape[0])):
                if shape[y][x]:
                    board_y = pos[1] + y
                    board_x = pos[0] + x
                    
                    # 바닥 접촉
                    if board_y + 1 >= 20:
                        contacts += 1
                    # 아래쪽 블록과 접촉
                    elif board_y + 1 < 20 and self.board[board_y + 1][board_x]:
                        contacts += 1
                    # 왼쪽 블록과 접촉
                    if board_x - 1 >= 0 and self.board[board_y][board_x - 1]:
                        contacts += 1
                    # 오른쪽 블록과 접촉
                    if board_x + 1 < 10 and self.board[board_y][board_x + 1]:
                        contacts += 1
        
        return contacts

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        reward = 0
        done = False
        piece_placed = False
        
        # 최근 행동 기록 업데이트
        self.recent_actions.append(action)
        if len(self.recent_actions) > 5:
            self.recent_actions.pop(0)
        
        # 동일한 행동 5번 이상 반복 시 페널티
        if len(self.recent_actions) == 5 and len(set(self.recent_actions)) == 1:
            reward += self.repetitive_penalty
        
        # 강제 하드드롭이 필요한 경우
        if self.must_drop and action != 4:
            return self._get_state(), 0, False, False, {
                'score': self.score,
                'lines': self.lines_cleared
            }
        
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
                self._reset_move_limits()  # 새 블록에 대한 제한 초기화
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
        
        elif action in [0, 1]:  # left/right
            if self.moves_left > 0:
                new_pos = [
                    self.current_piece['position'][0] + (-1 if action == 0 else 1),
                    self.current_piece['position'][1]
                ]
                if not self._check_collision({**self.current_piece, 'position': new_pos}, self.board):
                    self.current_piece['position'] = new_pos
                    self.moves_left -= 1
                    
                    # 강제 하드드롭 조건 제거
                    # if self.moves_left == 0 and self.rotates_left == 0:
                    #     self.must_drop = True
        
        elif action == 2:  # rotate
            if self.rotates_left > 0:
                new_rotation = (self.current_piece['rotation'] + 1) % len(self.PIECES[self.current_piece['type']])
                new_shape = self.PIECES[self.current_piece['type']][new_rotation]
                if not self._check_collision({**self.current_piece, 'rotation': new_rotation, 'shape': new_shape}, self.board):
                    self.current_piece['rotation'] = new_rotation
                    self.current_piece['shape'] = new_shape
                    self.rotates_left -= 1
                    
                    # 강제 하드드롭 조건 제거
                    # if self.moves_left == 0 and self.rotates_left == 0:
                    #     self.must_drop = True
        
        elif action == 3:  # down
            new_pos = [self.current_piece['position'][0], self.current_piece['position'][1] + 1]
            if not self._check_collision({**self.current_piece, 'position': new_pos}, self.board):
                self.current_piece['position'] = new_pos
            else:
                self._place_piece()
                piece_placed = True
                self.pieces_placed += 1
        
        elif action == 4:  # hardDrop
            # 하드드롭 전 위치 평가
            drop_quality = self._evaluate_drop_position()
            
            # 블록의 크기(셀 수) 계산
            piece_size = sum(sum(row) for row in self.current_piece['shape'])
            block_reward = piece_size * 2  # 각 셀당 2점의 보상
            
            while not self._check_collision(
                {**self.current_piece, 'position': [
                    self.current_piece['position'][0],
                    self.current_piece['position'][1] + 1
                ]},
                self.board
            ):
                self.current_piece['position'][1] += 1
            
            self._place_piece()
            piece_placed = True
            self.pieces_placed += 1
            self.must_drop = False
            
            # 하드드롭 위치 품질과 블록 크기에 따른 보상 합산
            reward += drop_quality + block_reward
        
        # 블록이 고정되었을 때
        if piece_placed:
            lines_cleared = self._clear_lines()
            reward += self._calculate_reward(lines_cleared)
            
            if self._is_game_over():
                done = True
            else:
                self.current_piece = self.next_piece
                self.next_piece = self._get_random_piece()
                self.can_hold = True
                self._reset_move_limits()
        
        if self.pieces_placed >= self.max_pieces:
            done = True
        
        return self._get_state(), reward, done, False, {
            'score': self.score,
            'lines': self.lines_cleared,
            'moves_left': self.moves_left,
            'rotates_left': self.rotates_left
        }
    
    def _place_piece(self):
        shape = self.current_piece['shape']
        pos = self.current_piece['position']
        
        for y in range(len(shape)):
            for x in range(len(shape[0])):
                if shape[y][x]:
                    self.board[pos[1] + y][pos[0] + x] = 1
    
    def _is_game_over(self) -> bool:
        return self._check_collision(self.next_piece, self.board)
    
    def _get_state(self) -> np.ndarray:
        """단순화된 상태 표현"""
        # 1. 현재 보드 상태 (200)
        board_state = self.board.flatten()
        
        # 2. 현재 피스의 실제 모양과 위치 (20x10)
        current_piece_board = self.board.copy()
        shape = self.current_piece['shape']
        pos = self.current_piece['position']
        for y in range(len(shape)):
            for x in range(len(shape[0])):
                if shape[y][x]:
                    board_y = pos[1] + y
                    board_x = pos[0] + x
                    if 0 <= board_y < 20 and 0 <= board_x < 10:
                        current_piece_board[board_y][board_x] = 1
        
        # 3. 다음 피스의 타입 (7)
        next_piece = np.zeros(7)
        piece_types = ['I', 'O', 'T', 'S', 'Z', 'J', 'L']
        next_piece[piece_types.index(self.next_piece['type'])] = 1
        
        # 4. 현재 피스의 착지 위치 (20x10)
        landing_board = current_piece_board.copy()
        original_y = self.current_piece['position'][1]  # 원래 y 위치 저장
        
        # 착지 위치 계산
        while not self._check_collision({
            **self.current_piece, 
            'position': [self.current_piece['position'][0], self.current_piece['position'][1] + 1]
        }, self.board):
            self.current_piece['position'][1] += 1
            landing_board = np.zeros((20, 10))
            for y in range(len(shape)):
                for x in range(len(shape[0])):
                    if shape[y][x]:
                        board_y = self.current_piece['position'][1] + y
                        board_x = self.current_piece['position'][0] + x
                        if 0 <= board_y < 20 and 0 <= board_x < 10:
                            landing_board[board_y][board_x] = 1
        
        # 원래 위치로 복구
        self.current_piece['position'][1] = original_y

        return np.concatenate([
            board_state,                    # 현재 보드 (200)
            current_piece_board.flatten(),  # 현재 피스 위치 (200)
            landing_board.flatten(),        # 착지 위치 (200)
            next_piece                      # 다음 피스 (7)
        ]).astype(np.float32)
    
    def _get_max_height(self) -> int:
        """현재 보드의 최대 높이 계산"""
        for y in range(20):
            if np.any(self.board[y]):
                return 20 - y
        return 0
    
    def _reset_move_limits(self):
        """새 블록에 대한 이동/회전 제한 초기화"""
        self.moves_left = 8  # 좌우 이동 가능 횟수 증가
        self.rotates_left = 4  # 회전 가능 횟수 증가
        self.must_drop = False
    
    def get_valid_actions(self) -> List[int]:
        """현재 상태에서 가능한 액션들의 리스트를 반환합니다."""
        valid_actions = []
        
        # 강제 하드드롭이 필요한 경우
        if self.must_drop:
            return [4]  # 하드드롭만 가능
            
        # 기본 액션들 검사
        if self.moves_left > 0:
            # 왼쪽 이동 가능 여부 확인
            if not self._check_collision({
                **self.current_piece, 
                'position': [self.current_piece['position'][0] - 1, self.current_piece['position'][1]]
            }, self.board):
                valid_actions.append(0)
                
            # 오른쪽 이동 가능 여부 확인
            if not self._check_collision({
                **self.current_piece,
                'position': [self.current_piece['position'][0] + 1, self.current_piece['position'][1]]
            }, self.board):
                valid_actions.append(1)
        
        # 회전 가능 여부 확인
        if self.rotates_left > 0:
            new_rotation = (self.current_piece['rotation'] + 1) % len(self.PIECES[self.current_piece['type']])
            new_shape = self.PIECES[self.current_piece['type']][new_rotation]
            if not self._check_collision({
                **self.current_piece,
                'rotation': new_rotation,
                'shape': new_shape
            }, self.board):
                valid_actions.append(2)
        
        # 아래로 이동과 하드드롭은 항상 가능
        valid_actions.extend([3, 4])
        
        # hold 가능 여부 확인
        if self.can_hold:
            valid_actions.append(5)
            
        return valid_actions
