import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################################
# 1) 대표 DQN 모델 (MediumDQN)만 유지
################################################################################

class MediumDQN(nn.Module):
    """
    '중간 복잡도' DQN 모델.
    - 입력: (board 200) + (현재 피스 200) + (착지 위치 200) + (다음 피스 7) = 607차원
    - 출력: 6개 액션 Q값
    """
    def __init__(self):
        super().__init__()
        
        # 보드 상태를 처리
        self.board_net = nn.Sequential(
            nn.Linear(200, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # 현재 피스
        self.current_piece_net = nn.Sequential(
            nn.Linear(200, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 착지 위치
        self.landing_net = nn.Sequential(
            nn.Linear(200, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 다음 피스
        self.next_piece_net = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU()
        )
        
        # 결합 레이어
        self.combine_net = nn.Sequential(
            nn.Linear(128 + 64 + 64 + 32, 128),  
            nn.ReLU(),
            nn.Linear(128, 6)  # 6개 액션 Q값
        )
    
    def forward(self, x, valid_actions=None):
        # x shape: (batch, 607)
        board_state = x[:, :200]          # (batch, 200)
        current_piece = x[:, 200:400]     # (batch, 200)
        landing = x[:, 400:600]          # (batch, 200)
        next_piece = x[:, 600:]          # (batch, 7)
        
        board_features = self.board_net(board_state)          # (batch, 128)
        current_features = self.current_piece_net(current_piece)  # (batch, 64)
        landing_features = self.landing_net(landing)          # (batch, 64)
        next_features = self.next_piece_net(next_piece)       # (batch, 32)
        
        combined = torch.cat([board_features, current_features, 
                              landing_features, next_features], dim=1)
        q_values = self.combine_net(combined)  # (batch, 6)
        
        if valid_actions is not None:
            # 유효 액션만 뽑아보는 로직 (optional)
            if isinstance(valid_actions, list):
                valid_actions = torch.tensor(valid_actions, device=x.device)
            return q_values.index_select(1, valid_actions)
        
        return q_values


################################################################################
# 2) AlphaZero식 MCTS에 사용할 Policy-Value Network 예시
################################################################################

class AlphaZeroPolicyValueNet(nn.Module):
    """
    AlphaZero 스타일의 Policy+Value 네트워크.
    - 입력: 테트리스 상태를 일렬로 (예: 보드(200) + next piece(7) + hold piece(7) 등)
      -> 여기서는 간단히 214차원 예시 (보드(200) + next(7) + hold(7))
      -> 실제론 607 등 다른 형태도 가능.
    - 출력:
      (1) policy: 모든 액션에 대한 확률 (크기 num_actions)
      (2) value: 스칼라 (현재 상태의 가치, -1~1 범위 등)
    """
    def __init__(self, obs_dim=214, num_actions=41):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        
        # 간단한 2-layer MLP로 representation
        self.representation = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(128, num_actions)
            # 학습 시엔 cross-entropy 등 사용 -> logits 출력
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(128, 1),
            nn.Tanh()  # -1~1 범위 등으로 가정
        )
    
    def forward(self, x):
        """
        x: (batch, obs_dim)
        return: policy_logits (batch, num_actions), value (batch, 1)
        """
        feat = self.representation(x)         # (batch, 128)
        policy_logits = self.policy_head(feat) # (batch, num_actions)
        value = self.value_head(feat)          # (batch, 1)
        return policy_logits, value


################################################################################
# 3) 모델 생성 함수
################################################################################

def create_model(model_type='dqn'):
    """
    model_type:
      - 'dqn': MediumDQN (단일 DQN)
      - 'alphazero': AlphaZeroPolicyValueNet (정책+가치)
    """
    if model_type == 'dqn':
        model = MediumDQN()
        print("[INFO] Created MediumDQN model.")
    elif model_type == 'alphazero':
        # 예: 보드(200) + next(7) + hold(7) = 214차원, 액션 41개 (회전x4*x좌표10 + hold)
        model = AlphaZeroPolicyValueNet(obs_dim=214, num_actions=41)
        print("[INFO] Created AlphaZeroPolicyValueNet model.")
    else:
        raise ValueError("Unknown model_type. Use 'dqn' or 'alphazero'.")
    
    # 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable params: {total_params}")
    
    return model


################################################################################
# 4) 메인 실행 (예시)
################################################################################
if __name__ == "__main__":
    # 예시: DQN 모델 생성
    dqn_model = create_model('dqn')
    
    # 예시: AlphaZero Policy-Value 모델 생성
    az_model = create_model('alphazero')
    
    # 임의의 입력으로 테스트
    test_input_dqn = torch.randn(4, 607)  # batch=4, obs_dim=607
    q_values = dqn_model(test_input_dqn)
    print("DQN q_values shape:", q_values.shape)  # (4, 6)
    
    test_input_az = torch.randn(4, 214)   # board(200)+next(7)+hold(7)=214
    policy_logits, value = az_model(test_input_az)
    print("AlphaZero policy shape:", policy_logits.shape)  # (4, 41)
    print("AlphaZero value shape:", value.shape)           # (4, 1)
