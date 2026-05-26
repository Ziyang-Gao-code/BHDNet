# ------------------------------------------------------------------------------
# Written by Ziyang Gao (gzy@stud.tjut.edu.cn)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )



class LargeKernelSpatialGate(nn.Module):
    def __init__(self, kernel_size=11):
        super(LargeKernelSpatialGate, self).__init__()
        self.compress = ChannelPool()


        padding_size = kernel_size // 2
        self.dw_conv = nn.Sequential(
            # 垂直条带卷积 (处理跨通道/跨高度交互)
            nn.Conv2d(
                2, 2, kernel_size=(kernel_size, 1), stride=1,
                padding=(padding_size, 0), groups=2, bias=False
            ),
            # 水平条带卷积 (处理跨宽度交互)
            nn.Conv2d(
                2, 2, kernel_size=(1, kernel_size), stride=1,
                padding=(0, padding_size), groups=2, bias=False
            )
        )

        self.bn1 = nn.BatchNorm2d(2)
        self.relu = nn.ReLU(inplace=True)

        self.pw_conv = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.dw_conv(x_compress)
        x_out = self.bn1(x_out)
        x_out = self.relu(x_out)
        x_out = self.pw_conv(x_out)
        x_out = self.bn2(x_out)

        return x_out  



class SE_Block(nn.Module):
    def __init__(self, inchannel, reduction_ratio=16):
        super(SE_Block, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        reduced_channels = max(1, inchannel // reduction_ratio)

        self.se = nn.Sequential(
            nn.Linear(inchannel, reduced_channels, bias=False),
            nn.ReLU(),
            nn.Linear(reduced_channels, inchannel, bias=False)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        pooled = self.gap(x).view(b, c)
        channel_logits = self.se(pooled).view(b, c, 1, 1)

        return channel_logits  # 直接返回未激活的得分 (Logits)



class CustomTripletAttention(nn.Module):
    def __init__(
            self,
            gate_channels,
            reduction_ratio=16,
            no_spatial=False,
    ):
        super(CustomTripletAttention, self).__init__()

        self.ChannelGateH = LargeKernelSpatialGate(kernel_size=11)
        self.ChannelGateW = LargeKernelSpatialGate(kernel_size=11)

        self.no_spatial = no_spatial
        if not no_spatial:
            self.se_block = SE_Block(gate_channels, reduction_ratio)

    def forward(self, x):
        # 分支 1: 高度-通道 (H-C) 交互
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        out1 = self.ChannelGateH(x_perm1)
        row = out1.permute(0, 2, 1, 3).contiguous()

        # 分支 2: 宽度-通道 (W-C) 交互
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        out2 = self.ChannelGateW(x_perm2)
        col = out2.permute(0, 3, 2, 1).contiguous()

        # 分支 3 & 广播融合
        if not self.no_spatial:
            cha = self.se_block(x)
            attn_weight = torch.sigmoid(row + col + cha)
        else:
            attn_weight = torch.sigmoid(row + col)

        return attn_weight



class BCFM(nn.Module):


    def __init__(self, dim, reduction=4):
        super(BCFM, self).__init__()
        self.weight_generator = CustomTripletAttention(dim, reduction)


        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y):
        """
        x: 语义分支特征 (Semantic Context)
        y: 细节分支特征 (Spatial Details)
        """
        # 1. 初始特征相加，聚合双分支的初步信息
        initial = x + y


        attn = self.weight_generator(initial)

        # 3. 动态特征融合 (互补式门控机制)
        # attn 倾向于 1 的区域使用语义特征 x
        # attn 倾向于 0 的区域使用细节特征 y
        fusion_feat = x * attn + y * (1 - attn)

        # 4. 残差连接增强信息流动，并进行最终的卷积投影
        result = initial + fusion_feat
        result = self.proj(result)

        return result


if __name__ == '__main__':
    # 模拟语义分割中的特征图输入
    x = torch.randn(2, 64, 128, 128)
    y = torch.randn(2, 64, 128, 128)
    block = BCFM(dim=64)  # 修正类名
    output = block(x, y)

    print(f"Input shape:     {x.shape}")
    print(f"Output shape:    {output.shape}")
    print(f"Parameter count: {sum(p.numel() for p in block.parameters())}")