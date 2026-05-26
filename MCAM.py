# ------------------------------------------------------------------------------
# Written by Ziyang Gao (gzy@stud.tjut.edu.cn)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F



class MCAM(nn.Module):
    def __init__(self, k_size=3):
        super(MCAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2,       dilation=1, bias=False)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2 * 3,   dilation=3, bias=False)
        self.conv5 = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2 * 5,   dilation=5, bias=False)
        self.conv7 = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2 * 7,   dilation=7, bias=False)
        self.conv9 = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2 * 9,   dilation=9, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)                              # (B, C, 1, 1)
        y_transformed = y.squeeze(-1).transpose(-1, -2)   # (B, 1, C)

        y1 = self.conv1(y_transformed)
        y3 = self.conv3(y_transformed)
        y5 = self.conv5(y_transformed)
        y7 = self.conv7(y_transformed)
        y9 = self.conv9(y_transformed)

        y = y1 + y3 + y5 + y7 + y9                       # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)             # (B, C, 1, 1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


if __name__ == '__main__':
    x = torch.randn(2, 128, 64, 128)   
    block = MCAM(k_size=3)
    output = block(x)
    print(f"Input shape:     {x.shape}")   # (2, 128, 64, 128)
    print(f"Output shape:    {output.shape}")  # (2, 128, 64, 128)
    print(f"Parameter count: {sum(p.numel() for p in block.parameters())}")  # 15
