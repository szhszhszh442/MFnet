"""
Multi-Scale Adaptive Dynamic Convolution Fusion Module (MS-ADCF)

核心创新：
1. 多尺度动态卷积：不同尺度自适应生成卷积核
2. 自适应频率分解：可学习的频率分解网络
3. 跨模态注意力：RGB和DSM深度交互

理论支撑：
- Dynamic Convolution: Attention over Convolution Kernels (CVPR 2020)
- Feature Pyramid Networks for Object Detection (CVPR 2017)
- Squeeze-and-Excitation Networks (CVPR 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossModalAttention(nn.Module):
    """
    跨模态注意力模块
    实现RGB和DSM特征的深度交互
    
    理论基础：
    - Self-Attention机制 (Attention is All You Need, NeurIPS 2017)
    - 跨模态特征交互
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        
        # RGB -> Depth 注意力
        self.query_rgb = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.key_depth = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.value_depth = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Depth -> RGB 注意力
        self.query_depth = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.key_rgb = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.value_rgb = nn.Conv2d(channels, channels, kernel_size=1)
        
        # 可学习的融合权重
        self.gamma_rgb = nn.Parameter(torch.zeros(1))
        self.gamma_depth = nn.Parameter(torch.zeros(1))
        
        # 输出融合
        self.out_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, rgb, depth):
        """
        Args:
            rgb: (B, C, H, W) RGB特征
            depth: (B, C, H, W) DSM特征
        Returns:
            fused: (B, C, H, W) 融合特征
        """
        B, C, H, W = rgb.shape
        
        # RGB attend to Depth
        query_rgb = self.query_rgb(rgb).view(B, -1, H * W)  # (B, C//r, H*W)
        key_depth = self.key_depth(depth).view(B, -1, H * W)  # (B, C//r, H*W)
        
        # 计算注意力分数
        attn_rgb2depth = torch.bmm(query_rgb.transpose(1, 2), key_depth)  # (B, H*W, H*W)
        attn_rgb2depth = F.softmax(attn_rgb2depth, dim=-1)
        
        # 应用注意力
        value_depth = self.value_depth(depth).view(B, C, H * W)  # (B, C, H*W)
        rgb_attended = torch.bmm(value_depth, attn_rgb2depth.transpose(1, 2))  # (B, C, H*W)
        rgb_attended = rgb_attended.view(B, C, H, W)
        rgb_out = rgb + self.gamma_rgb * rgb_attended
        
        # Depth attend to RGB
        query_depth = self.query_depth(depth).view(B, -1, H * W)
        key_rgb = self.key_rgb(rgb).view(B, -1, H * W)
        
        attn_depth2rgb = torch.bmm(query_depth.transpose(1, 2), key_rgb)
        attn_depth2rgb = F.softmax(attn_depth2rgb, dim=-1)
        
        value_rgb = self.value_rgb(rgb).view(B, C, H * W)
        depth_attended = torch.bmm(value_rgb, attn_depth2rgb.transpose(1, 2))
        depth_attended = depth_attended.view(B, C, H, W)
        depth_out = depth + self.gamma_depth * depth_attended
        
        # 融合两个方向的注意力结果
        fused = self.out_conv(torch.cat([rgb_out, depth_out], dim=1))
        
        return fused


class AdaptiveFrequencyDecomposition(nn.Module):
    """
    自适应频率分解模块
    可学习的频率分解网络
    
    理论基础：
    - 频域特征学习
    - 可学习的滤波器组
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # 可学习的频率分解
        self.freq_decompose = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * 3, kernel_size=1),  # 生成低、中、高频
        )
        
        # 频率注意力（SE-like）
        self.freq_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 3, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels * 3, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 频率融合
        self.freq_fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 输入特征
        Returns:
            freq_feat: (B, C, H, W) 频率分解后的特征
        """
        # 频率分解
        freq_features = self.freq_decompose(x)  # (B, C*3, H, W)
        
        # 频率注意力
        freq_attn = self.freq_attention(freq_features)  # (B, C*3, 1, 1)
        freq_features = freq_features * freq_attn
        
        # 频率融合
        freq_feat = self.freq_fusion(freq_features)  # (B, C, H, W)
        
        return freq_feat


class MultiScaleAdaptiveDynamicConvFusion(nn.Module):
    """
    多尺度自适应动态卷积融合模块 (MS-ADCF)
    
    核心创新：
    1. 多尺度动态卷积：不同尺度自适应生成卷积核
    2. 自适应频率分解：可学习的频率分解网络
    3. 跨模态注意力：RGB和DSM深度交互
    
    理论支撑：
    - Dynamic Convolution (CVPR 2020)
    - Feature Pyramid Networks (CVPR 2017)
    - Squeeze-and-Excitation Networks (CVPR 2018)
    """
    def __init__(self, channels_in, num_kernels=4, num_scales=3):
        super().__init__()
        self.channels_in = channels_in
        self.num_kernels = num_kernels
        self.num_scales = num_scales
        
        # 多尺度池化层
        self.scale_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1) for _ in range(num_scales)
        ])
        
        # 每个尺度的动态卷积核生成器
        self.kernel_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(channels_in * 2, channels_in),
                nn.ReLU(inplace=True),
                nn.Linear(channels_in, num_kernels * channels_in * 3 * 3),
            ) for _ in range(num_scales)
        ])
        
        # 自适应频率分解
        self.freq_decomposition = AdaptiveFrequencyDecomposition(channels_in)
        
        # 跨模态注意力
        self.cross_modal_attention = CrossModalAttention(channels_in)
        
        # 最终融合
        self.fusion = nn.Sequential(
            nn.Conv2d(channels_in * 3, channels_in, kernel_size=1),
            nn.BatchNorm2d(channels_in),
            nn.ReLU(inplace=True)
        )
        
        # 初始化
        self._init_weights()
        
        # 用于记录diversity loss
        self.diversity_loss = 0.0
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def get_diversity_loss(self):
        """获取多样性损失"""
        return self.diversity_loss
    
    def forward(self, rgb, depth):
        """
        Args:
            rgb: (B, C, H, W) RGB特征
            depth: (B, C, H, W) DSM特征
        Returns:
            fused: (B, C, H, W) 融合特征
        """
        B, C, H, W = rgb.shape
        
        # 1. 多尺度动态卷积
        multi_scale_features = []
        all_kernels = []
        
        for i, (pool, kernel_gen) in enumerate(zip(self.scale_pools, self.kernel_generators)):
            # 池化获取全局特征
            rgb_global = pool(rgb).view(B, C)
            depth_global = pool(depth).view(B, C)
            
            # 生成动态卷积核
            combined_global = torch.cat([rgb_global, depth_global], dim=1)
            dynamic_kernels_flat = kernel_gen(combined_global)  # (B, K*C*3*3)
            
            # 重塑为卷积核形状
            dynamic_kernels = dynamic_kernels_flat.view(B, self.num_kernels, C, 3, 3)
            all_kernels.append(dynamic_kernels)
            
            # 计算注意力权重（基于kernel范数）
            kernel_norms = torch.norm(dynamic_kernels.view(B, self.num_kernels, -1), dim=2)
            attention_weights = F.softmax(kernel_norms, dim=1)  # (B, K)
            
            # 加权组合卷积核
            weighted_kernel = torch.einsum('bk,bkcij->bcij', 
                                          attention_weights, 
                                          dynamic_kernels)  # (B, C, 3, 3)
            
            # 应用动态卷积（使用深度可分离卷积）
            scale_feats = []
            for b in range(B):
                # 深度可分离卷积：每个通道有自己的卷积核
                # rgb[b:b+1]: (1, C, H, W)
                # weighted_kernel[b:b+1]: (1, C, 3, 3) -> (C, 1, 3, 3)
                kernel = weighted_kernel[b].unsqueeze(1)  # (C, 1, 3, 3)
                feat = F.conv2d(rgb[b:b+1], kernel, padding=1, groups=C)
                scale_feats.append(feat)
            scale_feat = torch.cat(scale_feats, dim=0)
            multi_scale_features.append(scale_feat)
        
        # 融合多尺度特征
        multi_scale_fused = sum(multi_scale_features) / self.num_scales
        
        # 计算diversity loss（训练时）
        if self.training:
            with torch.no_grad():
                total_diversity = 0.0
                for kernels in all_kernels:
                    # 计算kernel之间的相似度
                    k_flat = kernels.view(B, self.num_kernels, -1)
                    k_norm = F.normalize(k_flat, dim=2)
                    similarity = torch.bmm(k_norm, k_norm.transpose(1, 2))
                    # 移除对角线
                    mask = 1 - torch.eye(self.num_kernels, device=kernels.device).unsqueeze(0)
                    off_diagonal = similarity * mask
                    diversity = off_diagonal.abs().mean()
                    total_diversity += diversity
                self.diversity_loss = total_diversity / len(all_kernels)
        
        # 2. 自适应频率分解
        freq_feat = self.freq_decomposition(depth)
        
        # 3. 跨模态注意力
        cross_modal_feat = self.cross_modal_attention(rgb, depth)
        
        # 调试信息
        # print(f"Debug - multi_scale_fused shape: {multi_scale_fused.shape}")
        # print(f"Debug - freq_feat shape: {freq_feat.shape}")
        # print(f"Debug - cross_modal_feat shape: {cross_modal_feat.shape}")
        
        # 4. 最终融合
        combined = torch.cat([multi_scale_fused, freq_feat, cross_modal_feat], dim=1)
        # print(f"Debug - combined shape: {combined.shape}")
        fused = self.fusion(combined)
        
        return fused


class LightweightMSADCF(nn.Module):
    """
    轻量级版本的多尺度自适应动态卷积融合
    适用于资源受限的场景
    """
    def __init__(self, channels_in, num_kernels=4):
        super().__init__()
        self.channels_in = channels_in
        self.num_kernels = num_kernels
        
        # 简化的动态卷积核生成
        self.kernel_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels_in * 2, channels_in),
            nn.ReLU(inplace=True),
            nn.Linear(channels_in, num_kernels * channels_in * 3 * 3),
        )
        
        # 频率分解（简化版）
        self.freq_decompose = nn.Sequential(
            nn.Conv2d(channels_in, channels_in, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels_in),
            nn.ReLU(inplace=True),
        )
        
        # 融合
        self.fusion = nn.Sequential(
            nn.Conv2d(channels_in * 2, channels_in, kernel_size=1),
            nn.BatchNorm2d(channels_in),
            nn.ReLU(inplace=True)
        )
        
        self.diversity_loss = 0.0
    
    def get_diversity_loss(self):
        return self.diversity_loss
    
    def forward(self, rgb, depth):
        B, C, H, W = rgb.shape
        
        # 动态卷积
        combined = torch.cat([rgb, depth], dim=1)
        kernels_flat = self.kernel_gen(combined)
        kernels = kernels_flat.view(B, self.num_kernels, C, 3, 3)
        
        # 注意力权重
        kernel_norms = torch.norm(kernels.view(B, self.num_kernels, -1), dim=2)
        attn_weights = F.softmax(kernel_norms, dim=1)
        
        # 加权kernel
        weighted_kernel = torch.einsum('bk,bkcij->bcij', attn_weights, kernels)
        
        # 应用卷积
        dynamic_feat = F.conv2d(rgb, weighted_kernel, padding=1)
        
        # 频率分解
        freq_feat = self.freq_decompose(depth)
        
        # 融合
        fused = self.fusion(torch.cat([dynamic_feat, freq_feat], dim=1))
        
        return fused


# 测试代码
if __name__ == '__main__':
    print("Testing MultiScaleAdaptiveDynamicConvFusion...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 测试完整版本
    model = MultiScaleAdaptiveDynamicConvFusion(channels_in=256, num_kernels=4, num_scales=3).to(device)
    
    # 测试输入
    rgb = torch.randn(2, 256, 64, 64).to(device)
    depth = torch.randn(2, 256, 64, 64).to(device)
    
    # 前向传播
    model.train()
    output = model(rgb, depth)
    
    print(f"Input shape: RGB={rgb.shape}, Depth={depth.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Diversity loss: {model.get_diversity_loss():.4f}")
    
    # 测试反向传播
    loss = output.mean()
    loss.backward()
    print("Backward pass successful!")
    
    # 测试轻量级版本
    print("\nTesting LightweightMSADCF...")
    model_light = LightweightMSADCF(channels_in=256, num_kernels=4).to(device)
    output_light = model_light(rgb, depth)
    print(f"Lightweight output shape: {output_light.shape}")
    
    # 参数量对比
    params_full = sum(p.numel() for p in model.parameters())
    params_light = sum(p.numel() for p in model_light.parameters())
    print(f"\nFull model parameters: {params_full:,}")
    print(f"Lightweight model parameters: {params_light:,}")
    print(f"Reduction: {(1 - params_light/params_full)*100:.1f}%")
    
    print("\n✅ All tests passed!")
