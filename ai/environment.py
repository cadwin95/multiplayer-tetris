import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from typing import Optional, Dict, Tuple, List

class TetrisEnv(gym.Env):
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
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
        )

        # 테트로미노 정의 (회전 형태 목록)
        self.PIECES = {
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
        super().reset(seed=seed)
        self.board = np.zeros((self.height, self.width), dtype=np.int8)
        self.steps_done = 0
        self.done = False

        # 현재 조각, 다음 조각, hold 조각
        self.current_piece = self._random_piece()
        self.next_piece = self._random_piece()
        self.hold_piece = None
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

        reward = 0
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
