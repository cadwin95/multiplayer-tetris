import torch
import torch.nn as nn

class SimpleDQN(nn.Module):
    """간단한 DQN 모델"""
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(607, 256),  # 607 = 보드(200) + 현재피스(200) + 착지위치(200) + 다음피스(7)
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # 6개 액션
        )
    
    def forward(self, x, valid_actions=None):
        q_values = self.network(x)
        
        if valid_actions is not None:
            # 유효한 액션에 대한 Q-value만 선택
            if isinstance(valid_actions, list):
                valid_actions = torch.tensor(valid_actions, device=x.device)
            return q_values.index_select(1, valid_actions)
        
        return q_values

class MediumDQN(nn.Module):
    """중간 복잡도의 DQN 모델"""
    def __init__(self):
        super().__init__()
        # 보드 상태를 처리하는 레이어
        self.board_net = nn.Sequential(
            nn.Linear(200, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # 현재 피스 상태를 처리하는 레이어
        self.current_piece_net = nn.Sequential(
            nn.Linear(200, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 착지 위치를 처리하는 레이어
        self.landing_net = nn.Sequential(
            nn.Linear(200, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 다음 피스 정보를 처리하는 레이어
        self.next_piece_net = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU()
        )
        
        # 결합 레이어
        self.combine_net = nn.Sequential(
            nn.Linear(128 + 64 + 64 + 32, 128),  # 288 = 128 + 64 + 64 + 32
            nn.ReLU(),
            nn.Linear(128, 6)  # 6개 액션
        )
    
    def forward(self, x, valid_actions=None):
        board_state = x[:, :200]
        current_piece = x[:, 200:400]
        landing = x[:, 400:600]
        next_piece = x[:, 600:]
        
        board_features = self.board_net(board_state)
        current_features = self.current_piece_net(current_piece)
        landing_features = self.landing_net(landing)
        next_features = self.next_piece_net(next_piece)
        
        combined = torch.cat([
            board_features, 
            current_features, 
            landing_features, 
            next_features
        ], dim=1)
        
        q_values = self.combine_net(combined)
        
        if valid_actions is not None:
            # 유효한 액션에 대한 Q-value만 선택
            if isinstance(valid_actions, list):
                valid_actions = torch.tensor(valid_actions, device=x.device)
            return q_values.index_select(1, valid_actions)
        
        return q_values

class ComplexDQN(nn.Module):
    """복잡한 DQN 모델 (CNN for board + MediumDQN structure)"""
    def __init__(self):
        super().__init__()
        
        # 보드 상태를 처리하는 CNN 레이어
        self.board_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 20 * 10, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # 현재 피스와 착지 위치를 처리하는 CNN
        self.piece_net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 20 * 10, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # 다음 피스 정보를 처리하는 레이어
        self.next_piece_net = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU()
        )
        
        # 결합 레이어
        self.combine_net = nn.Sequential(
            nn.Linear(128 + 64 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # 6개 액션
        )
    
    def forward(self, x, valid_actions=None):
        # 입력 재구성
        board = x[:, :200].view(-1, 1, 20, 10)
        current_piece = x[:, 200:400].view(-1, 1, 20, 10)
        landing = x[:, 400:600].view(-1, 1, 20, 10)
        next_piece = x[:, 600:]
        
        # 특징 추출
        board_features = self.board_net(board)
        piece_features = self.piece_net(torch.cat([current_piece, landing], dim=1))
        next_features = self.next_piece_net(next_piece)
        
        # 특징 결합
        combined = torch.cat([board_features, piece_features, next_features], dim=1)
        q_values = self.combine_net(combined)
        
        if valid_actions is not None:
            # 유효한 액션에 대한 Q-value만 선택
            if isinstance(valid_actions, list):
                valid_actions = torch.tensor(valid_actions, device=x.device)
            return q_values.index_select(1, valid_actions)
        
        return q_values

class TransformerDQN(nn.Module):
    """Transformer 기반 DQN 모델 - 보드의 공간 구조 인식 강화"""
    def __init__(self, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        
        # 보드의 각 행(row)을 토큰으로 처리
        self.row_embedding = nn.Linear(10, d_model)  # 각 행(10칸)을 임베딩
        self.position_embedding = nn.Parameter(torch.randn(1, 20, d_model))  # 20개 행의 위치 정보
        
        # 피스 정보 임베딩
        self.piece_embedding = nn.Linear(14, d_model)  # 현재(7) + 다음(7)
        
        # Transformer 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 출력 헤드 (Dueling DQN 구조)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        
        self.advantage_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 6)  # 6개 액션
        )
    
    def forward(self, x, valid_actions=None):
        batch_size = x.size(0)
        
        # 보드 상태를 20x10 형태로 변환
        board_state = x[:, :200].view(batch_size, 20, 10)
        piece_state = x[:, 200:]  # 14 = 현재(7) + 다음(7)
        
        # 각 행을 임베딩
        board_embed = self.row_embedding(board_state)  # [batch, 20, d_model]
        
        # 위치 임베딩 추가
        board_embed = board_embed + self.position_embedding
        
        # 피스 정보를 시퀀스에 추가
        piece_embed = self.piece_embedding(piece_state).unsqueeze(1)  # [batch, 1, d_model]
        
        # 전체 시퀀스 생성 (보드 행들 + 피스 정보)
        sequence = torch.cat([board_embed, piece_embed], dim=1)  # [batch, 21, d_model]
        
        # Transformer로 처리
        transformed = self.transformer(sequence)
        
        # 전체 시퀀스의 평균을 계산 (Global Average Pooling)
        pooled = transformed.mean(dim=1)
        
        # Dueling DQN 구조
        value = self.value_head(pooled)
        advantage = self.advantage_head(pooled)
        
        # Q값 계산: V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        if valid_actions is not None:
            # 유효한 액션에 대한 Q-value만 선택
            if isinstance(valid_actions, list):
                valid_actions = torch.tensor(valid_actions, device=x.device)
            return q_values.index_select(1, valid_actions)
        
        return q_values

class TetrisNet(nn.Module):
    """CNN 기반 테트리스 DQN 모델 - 3채널 이미지 처리 방식"""
    def __init__(self):
        super(TetrisNet, self).__init__()
        # CNN 레이어 (입력: 3채널 20x10)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=10, padding=1),
            nn.ReLU()
        )
        
        # 다음 피스 정보를 처리하는 레이어
        self.next_piece_features = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU()
        )
        
        # 결합 및 출력 레이어
        self.combine = nn.Sequential(
            nn.Linear(768, 512),  # CNN 출력 (64*5*3) + 다음 피스 특징 (64)
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )
        
    def forward(self, x, valid_actions=None):
        # 입력을 재구성 (607 = 200 + 200 + 200 + 7)
        board = x[:, :200].view(-1, 20, 10)
        current_piece = x[:, 200:400].view(-1, 20, 10)
        landing = x[:, 400:600].view(-1, 20, 10)
        next_piece = x[:, 600:]  # 7차원 벡터
        
        # 3채널 이미지로 구성 (batch_size, 3, 20, 10)
        board_state = torch.stack([board, current_piece, landing], dim=1)
        
        # CNN 처리
        conv_out = self.conv_layers(board_state)
        conv_out = conv_out.view(conv_out.size(0), -1)  # Flatten
        
        # 다음 피스 정보 처리
        next_f = self.next_piece_features(next_piece)
        
        # 특징 결합
        combined = torch.cat([conv_out, next_f], dim=1)
        
        # 최종 출력
        q_values = self.combine(combined)
        
        if valid_actions is not None:
            # 유효한 액션에 대한 Q-value만 선택
            if isinstance(valid_actions, list):
                valid_actions = torch.tensor(valid_actions, device=x.device)
            return q_values.index_select(1, valid_actions)
        
        return q_values

def create_model(complexity='simple'):
    """모델 복잡도에 따른 모델 생성"""
    if complexity == 'simple':
        model = SimpleDQN()
    elif complexity == 'medium':
        model = MediumDQN()
    elif complexity == 'transformer':
        model = TransformerDQN()
    elif complexity == 'tetris':  # 새로운 옵션 추가
        model = TetrisNet()
    else:  # 'complex'
        model = ComplexDQN()
    
    # 파라미터 수 계산 및 출력
    total_params = count_parameters(model)
    print(f"\n{'='*50}")
    print(f"Model Architecture: {complexity.upper()}")
    print(f"Total Trainable Parameters: {total_params:,}")
    print(f"{'='*50}\n")
    
    # 각 레이어별 파라미터 수 출력
    print("Parameters by layer:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:30s}: {param.numel():,}")
    print(f"{'='*50}\n")
    
    return model

def count_parameters(model):
    """모델의 학습 가능한 파라미터 수를 반환"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='테트리스 DQN 모델 파라미터 정�� 출력')
    parser.add_argument('--model', type=str, choices=['simple', 'medium', 'complex', 'transformer', 'tetris'],
                      default='simple', help='확인할 모델 종류 (default: simple)')
    parser.add_argument('--compare', action='store_true',
                      help='모든 모델의 파라미터 수를 비교')
    
    args = parser.parse_args()
    
    if args.compare:
        print("\n모델 파라미터 수 비교\n")
        models = {
            'simple': SimpleDQN(),
            'medium': MediumDQN(),
            'complex': ComplexDQN(),
            'transformer': TransformerDQN(),
            'tetris': TetrisNet()
        }
        
        # 결과를 저장할 리스트
        results = []
        
        # 각 모델의 파라미터 수 계산
        for name, model in models.items():
            params = count_parameters(model)
            results.append((name, params))
        
        # 파라미터 수에 따라 정렬
        results.sort(key=lambda x: x[1])
        
        # 결과 출력
        print(f"{'='*50}")
        print(f"{'Model':15s} | {'Parameters':>15s} | {'Relative Size':>15s}")
        print(f"{'-'*50}")
        
        min_params = results[0][1]  # 가장 작은 모델의 파라미터 수
        for name, params in results:
            relative = f"{params/min_params:.1f}x"
            print(f"{name:15s} | {params:15,d} | {relative:>15s}")
        
        print(f"{'='*50}\n")
    else:
        create_model(args.model)