import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from typing import Optional, Dict, Tuple, List

class TetrisEnv(gym.Env):
<<<<<<< Updated upstream
    """테트리스 강화학습 환경 (수정된 보상 구조)"""
    
    def __init__(self):
        # 액션 스페이스: 6가지 (좌/우/회전/다운/하드드롭/홀드)
        self.action_space = spaces.Discrete(6)
        
        # 관찰 스페이스
=======
    """
    (rotation, x)로 최종 배치하는 방식 + Hold 기능 + Next Piece 정보 포함.
    - 액션 스페이스:
       0~39 : (회전, x좌표) 조합
       40   : Hold 액션
    - 관찰 스페이스:
       보드(20x10) + NextPiece(7 one-hot) + HoldPiece(7 one-hot)
       => shape = (200 + 7 + 7,) = (214,)
    """

    def __init__(self, max_steps=200, non_huristic=False):
        super().__init__()

        self.height = 20
        self.width = 10
        self.max_steps = max_steps
        self.non_huristic = non_huristic
        # 회전 * x좌표(최대) + Hold
        self.action_space = spaces.Discrete(41)

        # 관찰: board 20*10 + next piece 7 + hold piece 7 => 214
        # 값 범위는 일단 0~1 가정 (보드), piece는 one-hot
        obs_shape = (200 + 7 + 7,)
>>>>>>> Stashed changes
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
        )

        # 테트로미노 정의 (회전 형태 목록)
        self.PIECES = {
<<<<<<< Updated upstream
            'I': [[[1, 1, 1, 1]], [[1], [1], [1], [1]]],
            'O': [[[1, 1], [1, 1]]],
            'T': [[[0, 1, 0], [1, 1, 1]],
                  [[1, 0], [1, 1], [1, 0]]],
            'S': [[[0, 1, 1], [1, 1, 0]],
                  [[1, 0], [1, 1], [0, 1]]],
            'Z': [[[1, 1, 0], [0, 1, 1]],
                  [[0, 1], [1, 1], [1, 0]]],
            'J': [[[1, 0, 0], [1, 1, 1]],
                  [[1, 1], [1, 0], [1, 0]]],
            'L': [[[0, 0, 1], [1, 1, 1]],
                  [[1, 0], [1, 0], [1, 1]]]
        }
        
        # hold piece
        self.hold_piece = None
        self.can_hold = True
        
        self.max_pieces = 200
        self.pieces_placed = 0
        
        # 좌우 진동 페널티 관련
        self.last_actions = []
        self.last_heights = []
        self.oscillation_penalty = 0
        self.last_clear = 0
        
        # 이동/회전 제한
        self.moves_left = 8
        self.rotates_left = 4
        self.must_drop = False
        
        # 반복행동 페널티
        self.recent_actions = []
        self.repetitive_penalty = -0.1
        
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
                    
                    # 보드 범위 밖이거나 이미 블록이 있는 곳
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

    ### 수정된 부분 (1) : 보상 로직 함수 재정의 ###
    def _calculate_reward(self, lines_cleared: int) -> float:
        """
        - 라인 클리어 보너스
        - 게임 오버 시 큰 패널티
        - 높이/울퉁불퉁함/구멍 패널티
        """
        reward = 0.0

        # 1) 라인 클리어 보너스 (기존 Alpha / Tetris류 스코어 스타일)
        if lines_cleared == 1:
            reward += 100
        elif lines_cleared == 2:
            reward += 400
        elif lines_cleared == 3:
            reward += 900
        elif lines_cleared == 4:
            reward += 1600

        # 2) 게임오버면 큰 페널티
        if self._is_game_over():
            reward -= 50  # 예: -50

        # 3) 높이/울퉁불퉁함/구멍 패널티
        holes = self._count_holes()
        bumpiness = self._calculate_bumpiness()
        max_h = self._get_max_height()

        # 예시로 구멍 1개당 -1
        reward -= holes * 1.0

        # 최대 높이가 10을 넘어갈 때마다 (초과분 * -0.5)
        if max_h > 10:
            reward -= (max_h - 10) * 0.5

        # bumpiness에 비례해 -0.2
        reward -= bumpiness * 0.2

        return reward
    ### 수정 끝 ###

    def _count_holes(self) -> int:
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
=======
            'I': [
                [[1,1,1,1]],
                [[1],[1],[1],[1]]
            ],
            'O': [
                [[1,1],[1,1]]
            ],
            'T': [
                [[0,1,0],[1,1,1]],
                [[1,0],[1,1],[1,0]],
                [[1,1,1],[0,1,0]],
                [[0,1],[1,1],[0,1]]
            ],
            'S': [
                [[0,1,1],[1,1,0]],
                [[1,0],[1,1],[0,1]]
            ],
            'Z': [
                [[1,1,0],[0,1,1]],
                [[0,1],[1,1],[1,0]]
            ],
            'J': [
                [[1,0,0],[1,1,1]],
                [[1,1],[1,0],[1,0]],
                [[1,1,1],[0,0,1]],
                [[0,1],[0,1],[1,1]]
            ],
            'L': [
                [[0,0,1],[1,1,1]],
                [[1,0],[1,0],[1,1]],
                [[1,1,1],[1,0,0]],
                [[1,1],[0,1],[0,1]]
            ]
        }
        self.piece_types = list(self.PIECES.keys())  # ['I','O','T','S','Z','J','L']

        self.reset()

    def reset(self, seed:Optional[int]=None, return_info=False):
>>>>>>> Stashed changes
        super().reset(seed=seed)
        self.board = np.zeros((self.height, self.width), dtype=np.int8)
        self.steps_done = 0
        self.done = False

        # 현재 조각, 다음 조각, hold 조각
        self.current_piece = self._random_piece()
        self.next_piece = self._random_piece()
        self.hold_piece = None
<<<<<<< Updated upstream
        self.can_hold = True
        self.score = 0
        self.lines_cleared = 0
        self.pieces_placed = 0
        self._reset_move_limits()
        self.recent_actions = []
        
        return self._get_state(), {}

    def _evaluate_drop_position(self) -> float:
        """하드드롭 위치 평가 (사용자 임의 정의)."""
        reward = 0
        
        holes_before = self._count_holes()
        temp_board = self.board.copy()
        shape = self.current_piece['shape']
        pos = self.current_piece['position']
        for y in range(len(shape)):
            for x in range(len(shape[0])):
                if shape[y][x]:
                    temp_board[pos[1] + y][pos[0] + x] = 1
        holes_after = self._count_holes()
        reward -= (holes_after - holes_before) * 30  # 새로 생긴 구멍당 -30
        
        max_height = self._get_max_height()
        if max_height > 15:
            reward -= (max_height - 15) * 20
        
        bumpiness = self._calculate_bumpiness()
        reward -= bumpiness * 2
        
        contacts = self._count_contacts()
        reward += contacts * 10
        
        return reward

    def _count_contacts(self) -> int:
        contacts = 0
        shape = self.current_piece['shape']
        pos = self.current_piece['position']
        
        for y in range(len(shape)):
            for x in range(len(shape[0])):
                if shape[y][x]:
                    board_y = pos[1] + y
                    board_x = pos[0] + x
                    
                    # 바닥
                    if board_y + 1 >= 20:
                        contacts += 1
                    elif board_y + 1 < 20 and self.board[board_y + 1][board_x]:
                        contacts += 1
                    # 좌/우 접촉
                    if board_x - 1 >= 0 and self.board[board_y][board_x - 1]:
                        contacts += 1
                    if board_x + 1 < 10 and self.board[board_y][board_x + 1]:
                        contacts += 1
        
        return contacts
=======
        self.can_hold = True  # 배치(액션) 한 번 하기 전엔 hold 불가로 변경 가능

        obs = self._get_obs()
        if return_info:
            return obs, {}
        else:
            return obs

    def _random_piece(self) -> Dict:
        ptype = random.choice(self.piece_types)
        return {'type': ptype}

    def step(self, action:int):
        if self.done:
            return self._get_obs(), 0.0, True, False, {}

        # 유효한 action 목록 구하기
        valid_moves = self.get_valid_moves(self.current_piece['type'])
        # action=40 이면 hold
        # action < len(valid_moves)이면 (회전, x좌표)
        # 그 외 => 무효 액션 -> 큰 패널티 & 종료
        if action == 40:
            # Hold
            if not self.can_hold:
                # hold 불가능 상태
                reward = -1
                self.done = True
                return self._get_obs(), reward, True, False, {"info":"Hold not allowed now"}
            # hold 실행
            reward = self._do_hold()
        else:
            # (회전,x) 배치 액션
            if action >= len(valid_moves):
                # 무효 액션 => 패널티 & 종료
                reward = -1
                self.done = True
                return self._get_obs(), reward, True, False, {"info":"Invalid action index"}
            else:
                rot, xpos = valid_moves[action]
                reward = self._place_piece_and_get_reward(self.current_piece['type'], rot, xpos)
                # 라인 클리어
                lines = self._clear_lines()
                reward += self._line_clear_bonus(lines)

                # 다음 조각으로 교체
                self.current_piece = self.next_piece
                self.next_piece = self._random_piece()
                # hold 사용 가능(=새 조각 배치 전 상태)
                self.can_hold = True

        self.steps_done += 1

        # 새 조각을 놓을 수 있는지 검사 (게임오버 여부)
        if len(self.get_valid_moves(self.current_piece['type'])) == 0:
            reward -= 1
            self.done = True

        if self.steps_done >= self.max_steps:
            self.done = True

        obs = self._get_obs()
        return obs, reward, self.done, False, {}

    def _do_hold(self):
        """
        hold 액션 수행:
          - hold_piece가 없다면 => hold_piece=current_piece, current_piece=next_piece
          - hold_piece가 있다면 => swap(current_piece <-> hold_piece)
        """
        if self.hold_piece is None:
            self.hold_piece = {'type': self.current_piece['type']}
            self.current_piece = self.next_piece
            self.next_piece = self._random_piece()
        else:
            temp = {'type': self.current_piece['type']}
            self.current_piece = {'type': self.hold_piece['type']}
            self.hold_piece = temp
        # hold 직후에는 다시 hold 못 함 (이후 블록을 배치해야 가능)
        self.can_hold = False
        # 보상은 살짝 주거나 페널티를 주거나 등 자유
        reward = 0  # 예: 약간의 페널티
        return reward

    def get_valid_moves(self, piece_type:str) -> List[Tuple[int,int]]:
        """
        가능한 (rotation, x) 목록을 구해, 보드에 놓을 수 있는지 체크
        """
        valid = []
        rotations = len(self.PIECES[piece_type])  # 회전 가지수
        for r in range(rotations):
            shape = self.PIECES[piece_type][r]
            shape_w = len(shape[0])
            # x 범위
            for x in range(self.width - shape_w + 1):
                # simulate fall
                final_y = self._simulate_fall(shape, x)
                if final_y >= 0:  # 놓을 수 있음
                    valid.append((r, x))
        return valid

    def _simulate_fall(self, shape, x):
        """
        shape를 (x,y)에서 아래로 떨어뜨려 충돌 직전 y를 구함
        - y<0 이면 배치 불가
        """
        for start_y in range(-len(shape), self.height):
            if self._check_collision(shape, x, start_y+1):
                return start_y
        return -1  # 전부 충돌이면 -1

    def _check_collision(self, shape, x_pos, y_pos):
        """
        shape를 (x_pos,y_pos)에 놓을 때 보드와 충돌?
        """
        h = len(shape)
        w = len(shape[0])
        for r in range(h):
            for c in range(w):
                if shape[r][c] == 1:
                    bx = x_pos + c
                    by = y_pos + r
                    if bx<0 or bx>=self.width:
                        return True
                    if by<0:
                        return True
                    if by>=self.height:
                        return True
                    if self.board[by][bx] == 1:
                        return True
        return False

    def _place_piece_and_get_reward(self, ptype, rotation, x_pos):
        """
        (ptype, rotation, x_pos)에 따라 final_y 계산, 보드에 놓고
        구멍/높이/울퉁불퉁함 패널티, 게임오버 여부 등 계산
        """
        shape = self.PIECES[ptype][rotation]
        final_y = self._simulate_fall(shape, x_pos)
        if final_y < 0:
            # 놓을 수 없음
            return -50

        # 실제 배치
        for r in range(len(shape)):
            for c in range(len(shape[0])):
                if shape[r][c] == 1:
                    self.board[final_y+r][x_pos+c] = 1
        if self.non_huristic:
            return 0
        # 구멍/높이/울퉁불퉁함 패널티
        holes = self._count_holes()
        bump = self._bumpiness()
        mh = self._max_height()
>>>>>>> Stashed changes

        reward = 0
<<<<<<< Updated upstream
        done = False
        piece_placed = False
        
        # 반복 행동 패널티
        self.recent_actions.append(action)
        if len(self.recent_actions) > 5:
            self.recent_actions.pop(0)
        if len(self.recent_actions) == 5 and len(set(self.recent_actions)) == 1:
            reward += self.repetitive_penalty
        
        if self.must_drop and action != 4:
            return self._get_state(), 0, False, False, {
                'score': self.score,
                'lines': self.lines_cleared
            }
        
        # 액션 로직
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
                self._reset_move_limits()
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
        
        elif action == 2:  # rotate
            if self.rotates_left > 0:
                new_rotation = (self.current_piece['rotation'] + 1) % len(self.PIECES[self.current_piece['type']])
                new_shape = self.PIECES[self.current_piece['type']][new_rotation]
                if not self._check_collision({**self.current_piece, 'rotation': new_rotation, 'shape': new_shape}, self.board):
                    self.current_piece['rotation'] = new_rotation
                    self.current_piece['shape'] = new_shape
                    self.rotates_left -= 1
        
        elif action == 3:  # down
            new_pos = [self.current_piece['position'][0], self.current_piece['position'][1] + 1]
            if not self._check_collision({**self.current_piece, 'position': new_pos}, self.board):
                self.current_piece['position'] = new_pos
            else:
                self._place_piece()
                piece_placed = True
                self.pieces_placed += 1
        
        elif action == 4:  # hardDrop
            drop_quality = self._evaluate_drop_position()
            piece_size = sum(sum(row) for row in self.current_piece['shape'])
            block_reward = piece_size * 2
            
            # 실제 하드드롭
            while not self._check_collision(
                {**self.current_piece, 'position': [self.current_piece['position'][0],
                                                    self.current_piece['position'][1] + 1]},
                self.board
            ):
                self.current_piece['position'][1] += 1
            
            self._place_piece()
            piece_placed = True
            self.pieces_placed += 1
            self.must_drop = False
            
            reward += drop_quality + block_reward
        
        # 블록이 고정되었을 때
        if piece_placed:
            lines_cleared = self._clear_lines()

            ### 수정된 부분 (2) : _calculate_reward() 결합 ###
            shaped_reward = self._calculate_reward(lines_cleared)
            reward += shaped_reward
            ###########################################

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
        """단순화된 상태 표현 (기존 코드)"""
        board_state = self.board.flatten()
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
        
        next_piece = np.zeros(7)
        piece_types = ['I', 'O', 'T', 'S', 'Z', 'J', 'L']
        next_piece[piece_types.index(self.next_piece['type'])] = 1
        
        landing_board = current_piece_board.copy()
        original_y = self.current_piece['position'][1]
        
        while not self._check_collision({
            **self.current_piece,
            'position': [self.current_piece['position'][0], self.current_piece['position'][1] + 1]
        }, self.board):
            self.current_piece['position'][1] += 1
            landing_board = np.zeros((20, 10))
            for yy in range(len(shape)):
                for xx in range(len(shape[0])):
                    if shape[yy][xx]:
                        by = self.current_piece['position'][1] + yy
                        bx = self.current_piece['position'][0] + xx
                        if 0 <= by < 20 and 0 <= bx < 10:
                            landing_board[by][bx] = 1
        
        self.current_piece['position'][1] = original_y

        return np.concatenate([
            board_state,
            current_piece_board.flatten(),
            landing_board.flatten(),
            next_piece
        ]).astype(np.float32)
    
    def _get_max_height(self) -> int:
        for y in range(20):
            if np.any(self.board[y]):
                return 20 - y
        return 0
    
    def _reset_move_limits(self):
        self.moves_left = 8
        self.rotates_left = 4
        self.must_drop = False
    
    def get_valid_actions(self) -> List[int]:
        """현재 상태에서 가능한 액션들의 리스트"""
        valid_actions = []
        
        if self.must_drop:
            return [4]  # 하드드롭만 가능
            
        if self.moves_left > 0:
            # 왼쪽
            if not self._check_collision({
                **self.current_piece,
                'position': [self.current_piece['position'][0] - 1, self.current_piece['position'][1]]
            }, self.board):
                valid_actions.append(0)
            # 오른쪽
            if not self._check_collision({
                **self.current_piece,
                'position': [self.current_piece['position'][0] + 1, self.current_piece['position'][1]]
            }, self.board):
                valid_actions.append(1)
        
        # 회전
        if self.rotates_left > 0:
            new_rotation = (self.current_piece['rotation'] + 1) % len(self.PIECES[self.current_piece['type']])
            new_shape = self.PIECES[self.current_piece['type']][new_rotation]
            if not self._check_collision({
                **self.current_piece,
                'rotation': new_rotation,
                'shape': new_shape
            }, self.board):
                valid_actions.append(2)
        
        # 아래로 이동, 하드드롭
        valid_actions.extend([3, 4])
        
        # hold
        if self.can_hold:
            valid_actions.append(5)
            
        return valid_actions
=======
        reward -= holes * 1.0
        reward -= bump * 0.2
        if mh > 10:
            reward -= (mh - 10)*0.5

        return reward

    def _clear_lines(self):
        lines_cleared = 0
        for row in range(self.height):
            if np.all(self.board[row] == 1):
                lines_cleared += 1
                # 당기기
                for r2 in range(row, 0, -1):
                    self.board[r2] = self.board[r2-1]
                self.board[0] = np.zeros(self.width, dtype=np.int8)
        return lines_cleared

    def _line_clear_bonus(self, lines:int):
        # 예시: 1줄=100,2줄=400,3줄=900,4줄=1600
        if lines==1: return 100
        elif lines==2: return 400
        elif lines==3: return 900
        elif lines==4: return 1600
        return 0

    def _count_holes(self):
        holes = 0
        for col in range(self.width):
            seen_block = False
            for row in range(self.height):
                if self.board[row][col] == 1:
                    seen_block = True
                elif seen_block and self.board[row][col] == 0:
                    holes += 1
        return holes

    def _bumpiness(self):
        heights = []
        for c in range(self.width):
            h=0
            for r in range(self.height):
                if self.board[r][c]==1:
                    h = self.height-r
                    break
            heights.append(h)
        bump=0
        for i in range(len(heights)-1):
            bump += abs(heights[i]-heights[i+1])
        return bump

    def _max_height(self):
        for r in range(self.height):
            if np.any(self.board[r]==1):
                return self.height-r
        return 0

    def _get_obs(self):
        """
        보드(20*10=200) + next_piece(7 one-hot) + hold_piece(7 one-hot)
        => shape=(214,)
        """
        board_flat = self.board.flatten().astype(np.float32)

        # next piece one-hot
        next_onehot = np.zeros(7, dtype=np.float32)
        n_idx = self.piece_types.index(self.next_piece['type'])
        next_onehot[n_idx] = 1.0

        # hold piece
        hold_onehot = np.zeros(7, dtype=np.float32)
        if self.hold_piece is not None:
            h_idx = self.piece_types.index(self.hold_piece['type'])
            hold_onehot[h_idx] = 1.0

        return np.concatenate([board_flat, next_onehot, hold_onehot])
>>>>>>> Stashed changes
