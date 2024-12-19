import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        
        # 입력 크기: 253 (200 + 16 + 16 + 10 + 10 + 1)
        self.fc1 = nn.Linear(253, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 6)  # 6개 액션 (left, right, rotate, down, hardDrop, hold)
        
        # 드롭아웃 추가
        self.dropout = nn.Dropout(0.2)
        
        # 레이어 정규화 (배치 정규화 대신)
        self.ln1 = nn.LayerNorm(512)
        self.ln2 = nn.LayerNorm(256)
        self.ln3 = nn.LayerNorm(128)
    
    def forward(self, x):
        # 완전연결 레이어
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.dropout(x)
        return self.fc4(x)
