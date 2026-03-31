"""
Differentiable Prompt Generation Module (DPG)

核心创新：
1. 可微分采样：使用Gumbel-Softmax实现端到端训练
2. 自适应重要性学习：动态学习特征重要性
3. 多尺度特征聚合：捕获不同尺度的上下文信息

理论支撑：
- Categorical Reparameterization with Gumbel-Softmax (ICLR 2017)
- Attention-based Feature Selection (CVPR 2018)
- Differentiable Neural Architecture Search (ICLR 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GumbelSoftmaxSampler(nn.Module):
    """
    可微分采样模块
    使用Gumbel-Softmax技巧实现端到端训练
    
    理论基础：
    - Categorical Reparameterization with Gumbel-Softmax (ICLR 2017)
    - 允许离散采样操作可微分
    """
    def __init__(self, temperature=1.0, hard=False):
        super().__init__()
        self.temperature = temperature
        self.hard = hard
        
    def forward(self, logits):
        """
        Args:
            logits: (B, K) 未归一化的分数
        Returns:
            samples: (B, K) 采样结果（one-hot或soft）
        """
        # Gumbel噪声
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        
        # Gumbel-Softmax
        y_soft = F.softmax((logits + gumbel_noise) / self.temperature, dim=-1)
        
        if self.hard:
            # Straight-through estimator
            index = y_soft.max(dim=-1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            # 前向传播使用hard，反向传播使用soft
            return y_hard - y_soft.detach() + y_soft
        else:
            return y_soft
    
    def set_temperature(self, temperature):
        """设置温度参数"""
        self.temperature = temperature


class AdaptiveImportanceLearning(nn.Module):
    """
    自适应重要性学习模块
    动态学习特征的重要性权重
    
    理论基础：
    - Squeeze-and-Excitation Networks (CVPR 2018)
    - Attention-based Feature Selection
    """
    def __init__(self, channels_in, reduction=4):
        super().__init__()
        self.channels_in = channels_in
        
        # 重要性预测网络
        self.importance_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels_in * 2, channels_in // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels_in // reduction, channels_in),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels_in * 2, channels_in // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_in // reduction, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, rgb, depth):
        """
        Args:
            rgb: (B, C, H, W) RGB特征
            depth: (B, C, H, W) DSM特征
        Returns:
            importance_weights: (B, C) 通道重要性
            spatial_weights: (B, 1, H, W) 空间重要性
        """
        # 通道重要性
        combined = torch.cat([rgb, depth], dim=1)
        importance_weights = self.importance_net(combined)  # (B, C)
        
        # 空间重要性
        spatial_weights = self.spatial_attention(combined)  # (B, 1, H, W)
        
        return importance_weights, spatial_weights


class MultiScaleFeatureAggregation(nn.Module):
    """
    多尺度特征聚合模块
    捕获不同尺度的上下文信息
    
    理论基础：
    - Feature Pyramid Networks (CVPR 2017)
    - Pyramid Scene Parsing Network (CVPR 2017)
    """
    def __init__(self, channels_in, scales=[1, 2, 4, 8]):
        super().__init__()
        self.channels_in = channels_in
        self.scales = scales
        
        # 每个尺度的特征提取
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels_in, channels_in, kernel_size=3, padding=scale, dilation=scale),
                nn.BatchNorm2d(channels_in),
                nn.ReLU(inplace=True)
            ) for scale in scales
        ])
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(channels_in * len(scales), channels_in, kernel_size=1),
            nn.BatchNorm2d(channels_in),
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
            aggregated: (B, C, H, W) 聚合后的特征
        """
        multi_scale_features = []
        for scale_conv in self.scale_convs:
            multi_scale_features.append(scale_conv(x))
        
        # 拼接并融合
        concat = torch.cat(multi_scale_features, dim=1)
        aggregated = self.fusion(concat)
        
        return aggregated


class DifferentiablePromptGenerate(nn.Module):
    """
    可微分提示生成模块 (DPG)
    
    核心创新：
    1. 可微分采样：使用Gumbel-Softmax实现端到端训练
    2. 自适应重要性学习：动态学习特征重要性
    3. 多尺度特征聚合：捕获不同尺度的上下文信息
    
    理论支撑：
    - Categorical Reparameterization with Gumbel-Softmax (ICLR 2017)
    - Squeeze-and-Excitation Networks (CVPR 2018)
    - Feature Pyramid Networks (CVPR 2017)
    """
    def __init__(self, channels_in, num_prompts=10, embed_dim=256, temperature=1.0):
        super().__init__()
        self.channels_in = channels_in
        self.num_prompts = num_prompts
        self.embed_dim = embed_dim
        
        # 自适应重要性学习
        self.importance_learning = AdaptiveImportanceLearning(channels_in)
        
        # 多尺度特征聚合
        self.multi_scale_agg = MultiScaleFeatureAggregation(channels_in)
        
        # 提示生成网络
        self.prompt_generator = nn.Sequential(
            nn.Conv2d(channels_in, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # 提示选择网络
        self.prompt_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, num_prompts),
        )
        
        # 可微分采样器
        self.sampler = GumbelSoftmaxSampler(temperature=temperature, hard=False)
        
        # 提示嵌入库
        self.prompt_embeddings = nn.Parameter(
            torch.randn(1, num_prompts, embed_dim)
        )
        
        # 初始化
        self._init_weights()
        
        # 用于记录采样分布
        self.sampling_distribution = None
    
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
        
        # 初始化提示嵌入
        nn.init.normal_(self.prompt_embeddings, mean=0.0, std=0.02)
    
    def set_temperature(self, temperature):
        """设置采样温度"""
        self.sampler.set_temperature(temperature)
    
    def forward(self, rgb, depth):
        """
        Args:
            rgb: (B, C, H, W) RGB特征
            depth: (B, C, H, W) DSM特征
        Returns:
            prompt_emb: (B, num_prompts, embed_dim) 提示嵌入
            importance_weights: (B, C) 通道重要性
        """
        B, C, H, W = rgb.shape
        
        # 1. 自适应重要性学习
        importance_weights, spatial_weights = self.importance_learning(rgb, depth)
        
        # 应用重要性权重
        rgb_weighted = rgb * importance_weights.view(B, C, 1, 1) * spatial_weights
        depth_weighted = depth * importance_weights.view(B, C, 1, 1) * spatial_weights
        
        # 2. 多尺度特征聚合
        rgb_agg = self.multi_scale_agg(rgb_weighted)
        depth_agg = self.multi_scale_agg(depth_weighted)
        
        # 3. 特征融合
        fused = rgb_agg + depth_agg
        
        # 4. 提示生成
        prompt_features = self.prompt_generator(fused)  # (B, embed_dim, H, W)
        
        # 5. 提示选择
        selection_logits = self.prompt_selector(prompt_features)  # (B, num_prompts)
        
        # 6. 可微分采样
        sampling_probs = self.sampler(selection_logits)  # (B, num_prompts)
        self.sampling_distribution = sampling_probs
        
        # 7. 生成提示嵌入
        # 方法1：加权组合
        prompt_emb = torch.einsum('bn,bnd->bd', 
                                  sampling_probs, 
                                  self.prompt_embeddings.expand(B, -1, -1))
        prompt_emb = prompt_emb.unsqueeze(1)  # (B, 1, embed_dim)
        
        # 方法2：也可以生成多个提示
        # prompt_emb = torch.einsum('bn,bnd->bnd', 
        #                           sampling_probs, 
        #                           self.prompt_embeddings.expand(B, -1, -1))
        
        return prompt_emb, importance_weights


class LightweightDPG(nn.Module):
    """
    轻量级可微分提示生成模块
    适用于资源受限的场景
    """
    def __init__(self, channels_in, num_prompts=10, embed_dim=256):
        super().__init__()
        self.channels_in = channels_in
        self.num_prompts = num_prompts
        self.embed_dim = embed_dim
        
        # 简化的重要性学习
        self.importance_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels_in * 2, channels_in),
            nn.Sigmoid()
        )
        
        # 提示生成
        self.prompt_gen = nn.Sequential(
            nn.Conv2d(channels_in, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # 提示嵌入
        self.prompt_embeddings = nn.Parameter(torch.randn(1, num_prompts, embed_dim))
        
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
        
        nn.init.normal_(self.prompt_embeddings, mean=0.0, std=0.02)
    
    def forward(self, rgb, depth):
        B, C, H, W = rgb.shape
        
        # 重要性权重
        combined = torch.cat([rgb, depth], dim=1)
        importance = self.importance_net(combined)
        
        # 加权融合
        fused = (rgb + depth) * importance.view(B, C, 1, 1)
        
        # 生成提示
        prompt_feat = self.prompt_gen(fused)
        prompt_emb = F.adaptive_avg_pool2d(prompt_feat, 1).view(B, 1, self.embed_dim)
        
        return prompt_emb, importance


# 测试代码
if __name__ == '__main__':
    print("Testing DifferentiablePromptGenerate...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 测试完整版本
    model = DifferentiablePromptGenerate(
        channels_in=256,
        num_prompts=10,
        embed_dim=256,
        temperature=1.0
    ).to(device)
    
    # 测试输入
    rgb = torch.randn(2, 256, 64, 64).to(device)
    depth = torch.randn(2, 256, 64, 64).to(device)
    
    # 前向传播
    model.train()
    prompt_emb, importance = model(rgb, depth)
    
    print(f"Input shape: RGB={rgb.shape}, Depth={depth.shape}")
    print(f"Prompt embedding shape: {prompt_emb.shape}")
    print(f"Importance weights shape: {importance.shape}")
    print(f"Sampling distribution: {model.sampling_distribution[0]}")
    
    # 测试反向传播
    loss = prompt_emb.mean()
    loss.backward()
    print("Backward pass successful!")
    
    # 测试温度调节
    model.set_temperature(0.5)
    prompt_emb_low_temp, _ = model(rgb, depth)
    print(f"\nLow temperature sampling distribution: {model.sampling_distribution[0]}")
    
    # 测试轻量级版本
    print("\nTesting LightweightDPG...")
    model_light = LightweightDPG(channels_in=256, num_prompts=10, embed_dim=256).to(device)
    prompt_emb_light, importance_light = model_light(rgb, depth)
    print(f"Lightweight prompt embedding shape: {prompt_emb_light.shape}")
    
    # 参数量对比
    params_full = sum(p.numel() for p in model.parameters())
    params_light = sum(p.numel() for p in model_light.parameters())
    print(f"\nFull model parameters: {params_full:,}")
    print(f"Lightweight model parameters: {params_light:,}")
    print(f"Reduction: {(1 - params_light/params_full)*100:.1f}%")
    
    print("\n✅ All tests passed!")
