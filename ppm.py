# ------------------------------------------------------------------------------
# Written by Ziyang Gao (gzy@stud.tjut.edu.cn)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

algc = False

class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(DAPPM, self).__init__()
        bn_mom = 0.1
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale0 = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.process1 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process2 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process3 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process4 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )        
        self.compression = nn.Sequential(
                                    BatchNorm(branch_planes * 5, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
                                    )
        self.shortcut = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                                    )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]        
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[3])))
       
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out 
    
class PAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(PAPPM, self).__init__()
        bn_mom = 0.1
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )

        self.scale0 = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        
        self.scale_process = nn.Sequential(
                                    BatchNorm(branch_planes*4, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes*4, branch_planes*4, kernel_size=3, padding=1, groups=4, bias=False),
                                    )

      
        self.compression = nn.Sequential(
                                    BatchNorm(branch_planes * 5, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
                                    )
        
        self.shortcut = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                                    )


    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]        
        scale_list = []

        x_ = self.scale0(x)
        scale_list.append(F.interpolate(self.scale1(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        scale_list.append(F.interpolate(self.scale2(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        scale_list.append(F.interpolate(self.scale3(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        scale_list.append(F.interpolate(self.scale4(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        
        scale_out = self.scale_process(torch.cat(scale_list, 1))
       
        out = self.compression(torch.cat([x_,scale_out], 1)) + self.shortcut(x)
        return out

class BFPPM_C(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(BFPPM_C, self).__init__()
        bn_mom = 0.1


        self.scale1 = nn.Sequential(
            # 1. 水平池化提取上下文
            nn.AvgPool2d(kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            BatchNorm(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            # 2. 中间标准 3x3 卷积 (平滑特征，消除混叠，维持通道数不变)
            nn.Conv2d(inplanes, branch_planes, kernel_size=3, padding=1, bias=False),
            # 3. 垂直池化提取上下文
            nn.AvgPool2d(kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
        )

        self.scale2 = nn.Sequential(
            # 1. 水平池化提取上下文
            nn.AvgPool2d(kernel_size=(1, 9), stride=(1, 4), padding=(0, 4)),
            BatchNorm(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            # 2. 中间标准 3x3 卷积
            nn.Conv2d(inplanes, branch_planes, kernel_size=3, padding=1, bias=False),
            # 3. 垂直池化提取上下文
            nn.AvgPool2d(kernel_size=(9, 1), stride=(4, 1), padding=(4, 0)),
        )

        self.scale3 = nn.Sequential(
            # 1. 水平池化提取上下文
            nn.AvgPool2d(kernel_size=(1, 17), stride=(1, 8), padding=(0, 8)),
            BatchNorm(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            # 2. 中间标准 3x3 卷积
            nn.Conv2d(inplanes, branch_planes, kernel_size=3, padding=1, bias=False),
            # 3. 垂直池化提取上下文
            nn.AvgPool2d(kernel_size=(17, 1), stride=(8, 1), padding=(8, 0)),
        )

        # --- 保持不变的全局/局部特征提取分支 ---
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            BatchNorm(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )

        self.scale0 = nn.Sequential(
            BatchNorm(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )

        # --- 保持不变的级联平滑处理分支 ---
        self.process1 = nn.Sequential(
            BatchNorm(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process2 = nn.Sequential(
            BatchNorm(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process3 = nn.Sequential(
            BatchNorm(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process4 = nn.Sequential(
            BatchNorm(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )

        # --- 保持不变的特征压缩与残差连接 ---
        self.compression = nn.Sequential(
            BatchNorm(branch_planes * 5, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            BatchNorm(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]

        algc = False

        # =======================================================
        # Step 1: 并行提取特征并统一上采样到相同分辨率
        # =======================================================
        # 顺序：从最精细(S0) 到 最粗糙/全局(S4)
        s0 = self.scale0(x)
        s1 = F.interpolate(self.scale1(x), size=[height, width], mode='bilinear', align_corners=algc)
        s2 = F.interpolate(self.scale2(x), size=[height, width], mode='bilinear', align_corners=algc)
        s3 = F.interpolate(self.scale3(x), size=[height, width], mode='bilinear', align_corners=algc)
        s4 = F.interpolate(self.scale4(x), size=[height, width], mode='bilinear', align_corners=algc)

        feats_up = [s0, s1, s2, s3, s4]
        num_scales = 5

        # =======================================================
        # Step 2: 零参数双向信息流 (全频段特征融合)
        # =======================================================
        # 路径 A: Coarse-to-Fine (全局 -> 局部)
        c2f_feats = [None] * num_scales
        accum = torch.zeros_like(feats_up[0])
        for i in range(num_scales - 1, -1, -1):  # 倒序: 4, 3, 2, 1, 0
            accum = accum + feats_up[i]
            c2f_feats[i] = accum

        # 路径 B: Fine-to-Coarse (局部 -> 全局)
        f2c_feats = [None] * num_scales
        accum = torch.zeros_like(feats_up[0])
        for i in range(num_scales):  # 正序: 0, 1, 2, 3, 4
            accum = accum + feats_up[i]
            f2c_feats[i] = accum

        # 合并双向流
        fused_feats = [c2f + f2c for c2f, f2c in zip(c2f_feats, f2c_feats)]

        # =======================================================
        # Step 3: 并行细化 (完美适配原版的 Process 模块)
        # =======================================================
        x_list = []
        x_list.append(fused_feats[0])
        x_list.append(self.process1(fused_feats[1]))
        x_list.append(self.process2(fused_feats[2]))
        x_list.append(self.process3(fused_feats[3]))
        x_list.append(self.process4(fused_feats[4]))

        # =======================================================
        # Step 4: 拼接降维与残差相加
        # =======================================================
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out


class BFPPM_D(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(BFPPM_D, self).__init__()
        bn_mom = 0.1

        self.scale1 = nn.Sequential(
            # 1. 水平池化提取上下文
            nn.AvgPool2d(kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            BatchNorm(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            # 2. 中间标准 3x3 卷积 (平滑特征，消除混叠，维持通道数不变)
            nn.Conv2d(inplanes, branch_planes, kernel_size=3, padding=1, bias=False),
            # 3. 垂直池化提取上下文
            nn.AvgPool2d(kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
        )

        self.scale2 = nn.Sequential(
            # 1. 水平池化提取上下文
            nn.AvgPool2d(kernel_size=(1, 9), stride=(1, 4), padding=(0, 4)),
            BatchNorm(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            # 2. 中间标准 3x3 卷积
            nn.Conv2d(inplanes, branch_planes, kernel_size=3, padding=1, bias=False),
            # 3. 垂直池化提取上下文
            nn.AvgPool2d(kernel_size=(9, 1), stride=(4, 1), padding=(4, 0)),
        )

        self.scale3 = nn.Sequential(
            # 1. 水平池化提取上下文
            nn.AvgPool2d(kernel_size=(1, 17), stride=(1, 8), padding=(0, 8)),
            BatchNorm(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            # 2. 中间标准 3x3 卷积
            nn.Conv2d(inplanes, branch_planes, kernel_size=3, padding=1, bias=False),
            # 3. 垂直池化提取上下文
            nn.AvgPool2d(kernel_size=(17, 1), stride=(8, 1), padding=(8, 0)),
        )

        # --- 保持不变的全局/局部特征提取分支 ---
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            BatchNorm(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )

        self.scale0 = nn.Sequential(
            BatchNorm(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )


        self.process1 = nn.Sequential(
            BatchNorm(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            # dilation=2, padding=2
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=2, dilation=2, bias=False),
        )

        self.process2 = nn.Sequential(
            BatchNorm(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            # dilation=3, padding=3
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=3, dilation=3, bias=False),
        )

        self.process3 = nn.Sequential(
            BatchNorm(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            # dilation=5, padding=5
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=5, dilation=5, bias=False),
        )

        self.process4 = nn.Sequential(
            BatchNorm(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            # dilation=7, padding=7 (对于全局/超大尺度特征，7 的空洞率能提供极致的宏观感受野)
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=7, dilation=7, bias=False),
        )

        # --- 保持不变的特征压缩与残差连接 ---
        self.compression = nn.Sequential(
            BatchNorm(branch_planes * 5, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            BatchNorm(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]


        algc = False

        # =======================================================
        # Step 1: 并行提取特征并统一上采样到相同分辨率
        # =======================================================
        # 顺序：从最精细(S0) 到 最粗糙/全局(S4)
        s0 = self.scale0(x)
        s1 = F.interpolate(self.scale1(x), size=[height, width], mode='bilinear', align_corners=algc)
        s2 = F.interpolate(self.scale2(x), size=[height, width], mode='bilinear', align_corners=algc)
        s3 = F.interpolate(self.scale3(x), size=[height, width], mode='bilinear', align_corners=algc)
        s4 = F.interpolate(self.scale4(x), size=[height, width], mode='bilinear', align_corners=algc)

        feats_up = [s0, s1, s2, s3, s4]
        num_scales = 5

        # =======================================================
        # Step 2: 零参数双向信息流 (全频段特征融合)
        # =======================================================
        # 路径 A: Coarse-to-Fine (全局 -> 局部)
        c2f_feats = [None] * num_scales
        accum = torch.zeros_like(feats_up[0])
        for i in range(num_scales - 1, -1, -1):  # 倒序: 4, 3, 2, 1, 0
            accum = accum + feats_up[i]
            c2f_feats[i] = accum

        # 路径 B: Fine-to-Coarse (局部 -> 全局)
        f2c_feats = [None] * num_scales
        accum = torch.zeros_like(feats_up[0])
        for i in range(num_scales):  # 正序: 0, 1, 2, 3, 4
            accum = accum + feats_up[i]
            f2c_feats[i] = accum

        # 合并双向流
        fused_feats = [c2f + f2c for c2f, f2c in zip(c2f_feats, f2c_feats)]

        # =======================================================
        # Step 3: 并行细化 (完美适配原版的 Process 模块)
        # =======================================================
        x_list = []
        x_list.append(fused_feats[0])
        x_list.append(self.process1(fused_feats[1]))
        x_list.append(self.process2(fused_feats[2]))
        x_list.append(self.process3(fused_feats[3]))
        x_list.append(self.process4(fused_feats[4]))

        # =======================================================
        # Step 4: 拼接降维与残差相加
        # =======================================================
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out




if __name__ == '__main__':
    import torch
    import torch.nn.functional as F


    input_tensor = torch.randn(2, 64 * 16, 16, 32)
    block = BFPPM_C(inplanes=64 * 16, branch_planes=96, outplanes=64 * 4)
    output = block(input_tensor)

    print(f"Input shape:     {input_tensor.shape}")   # (2, 1024, 16, 32)
    print(f"Output shape:    {output.shape}")          # (2, 256,  16, 32)
    print(f"Parameter count: {sum(p.numel() for p in block.parameters())}")


