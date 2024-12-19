import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride, padding=kernel_size//2, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=2):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.use_res_connect = stride == 1 and in_channels == out_channels

        self.conv = nn.Sequential(
            # Pointwise
            ConvBlock(in_channels, hidden_dim, kernel_size=1),
            # Depthwise
            ConvBlock(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim),
            # Pointwise linear
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Efficient feature extraction
        self.features = nn.Sequential(
            ConvBlock(1, 16),                    # 1x20x10 -> 16x20x10
            InvertedResidual(16, 16),            # Skip connection
            nn.MaxPool2d(2, 2),                  # -> 16x10x5
            
            InvertedResidual(16, 32),            # -> 32x10x5
            InvertedResidual(32, 32),            # Skip connection
            nn.MaxPool2d(2, 2),                  # -> 32x5x2
            
            ConvBlock(32, 64, kernel_size=1),    # 1x1 conv for channel reduction
        )
        
        # Efficient processing for additional features
        self.additional_net = nn.Sequential(
            nn.Linear(53, 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Combined processing with residual connection
        feature_size = 64 * 5 * 2 + 32
        self.value_net = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1)  # Value stream
        )
        
        self.advantage_net = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 6)  # Advantage stream for 6 actions
        )
    
    def forward(self, x):
        # Split input
        board = x[:, :200].view(-1, 1, 20, 10)      # Board state
        additional = x[:, 200:]                      # Additional features
        
        # Process board
        board_features = self.features(board)
        board_features = board_features.view(board_features.size(0), -1)
        
        # Process additional features
        additional_features = self.additional_net(additional)
        
        # Combine features
        combined = torch.cat([board_features, additional_features], dim=1)
        
        # Dueling architecture
        value = self.value_net(combined)
        advantage = self.advantage_net(combined)
        
        # Q = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
