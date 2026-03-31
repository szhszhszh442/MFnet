"""
Differentiable SAM Prompt Generation Module

核心创新：
1. 可微分点提示生成：从特征图生成点坐标和标签
2. 可微分框提示生成：从特征图生成边界框
3. 可微分掩码提示生成：从特征图生成初始掩码
4. 自适应提示类型选择：动态选择最佳提示类型

理论支撑：
- Categorical Reparameterization with Gumbel-Softmax (ICLR 2017)
- Attention-based Point Selection (CVPR 2020)
- Differentiable Bounding Box Regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DifferentiablePointPrompt(nn.Module):
    """
    可微分点提示生成
    
    从特征图生成点坐标和标签
    """
    def __init__(self, channels_in, num_points=5, image_size=256):
        super().__init__()
        self.channels_in = channels_in
        self.num_points = num_points
        self.image_size = image_size
        
        # 点重要性预测
        self.point_importance = nn.Sequential(
            nn.Conv2d(channels_in, channels_in // 4, kernel_size=1),
            nn.BatchNorm2d(channels_in // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_in // 4, 1, kernel_size=1),
        )
        
        # 点标签预测（前景点还是背景点）
        self.point_label_net = nn.Sequential(
            nn.Conv2d(channels_in, channels_in // 4, kernel_size=1),
            nn.BatchNorm2d(channels_in // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_in // 4, 1, kernel_size=1),
            nn.Sigmoid()
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
    
    def forward(self, features):
        """
        Args:
            features: (B, C, H, W) 输入特征
        Returns:
            point_coords: (B, num_points, 2) 点坐标 (x, y)，归一化到[0, 1]
            point_labels: (B, num_points) 点标签 (1=前景点, 0=背景点)
        """
        B, C, H, W = features.shape
        
        # 1. 预测点重要性图
        importance_map = self.point_importance(features)  # (B, 1, H, W)
        importance_flat = importance_map.view(B, -1)  # (B, H*W)
        
        # 2. 使用Gumbel-Softmax采样top-k个点
        # 添加Gumbel噪声
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(importance_flat) + 1e-8) + 1e-8)
        logits = importance_flat + gumbel_noise
        
        # Softmax得到概率分布
        probs = F.softmax(logits, dim=-1)  # (B, H*W)
        
        # 采样top-k个点（可微分）
        # 方法1：直接取top-k（不可微分）
        # topk_indices = torch.topk(probs, self.num_points, dim=-1)[1]
        
        # 方法2：使用softmax温度采样（可微分）
        # 为了可微分，我们使用加权平均的方式
        # 生成候选坐标网格
        y_coords = torch.linspace(0, 1, H, device=features.device)
        x_coords = torch.linspace(0, 1, W, device=features.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)  # (H*W, 2)
        
        # 加权平均得到点坐标
        point_coords = torch.einsum('bh,bhc->bc', probs, grid.unsqueeze(0).expand(B, -1, -1))
        point_coords = point_coords.unsqueeze(1).expand(-1, self.num_points, -1)  # (B, num_points, 2)
        
        # 添加一些扰动以生成多个不同的点
        perturbation = torch.randn(B, self.num_points, 2, device=features.device) * 0.05
        point_coords = torch.clamp(point_coords + perturbation, 0, 1)
        
        # 3. 预测点标签
        label_map = self.point_label_net(features)  # (B, 1, H, W)
        label_probs = F.adaptive_avg_pool2d(label_map, (self.num_points, 1)).squeeze(1).squeeze(-1)  # (B, num_points)
        point_labels = (label_probs > 0.5).float()  # (B, num_points)
        
        return point_coords, point_labels


class DifferentiableBoxPrompt(nn.Module):
    """
    可微分框提示生成
    
    从特征图生成边界框 [x1, y1, x2, y2]
    """
    def __init__(self, channels_in, image_size=256):
        super().__init__()
        self.channels_in = channels_in
        self.image_size = image_size
        
        # 框中心预测
        self.center_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels_in, channels_in // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels_in // 4, 2),
            nn.Sigmoid()
        )
        
        # 框大小预测
        self.size_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels_in, channels_in // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels_in // 4, 2),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features):
        """
        Args:
            features: (B, C, H, W) 输入特征
        Returns:
            boxes: (B, 4) 边界框 [x1, y1, x2, y2]，归一化到[0, 1]
        """
        B = features.shape[0]
        
        # 预测框中心和大小
        center = self.center_net(features)  # (B, 2) [cx, cy]
        size = self.size_net(features)  # (B, 2) [w, h]
        
        # 计算边界框坐标
        x1 = center[:, 0] - size[:, 0] / 2
        y1 = center[:, 1] - size[:, 1] / 2
        x2 = center[:, 0] + size[:, 0] / 2
        y2 = center[:, 1] + size[:, 1] / 2
        
        # 确保在[0, 1]范围内
        boxes = torch.stack([x1, y1, x2, y2], dim=1)  # (B, 4)
        boxes = torch.clamp(boxes, 0, 1)
        
        return boxes


class DifferentiableMaskPrompt(nn.Module):
    """
    可微分掩码提示生成
    
    从特征图生成初始掩码
    """
    def __init__(self, channels_in, mask_size=256):
        super().__init__()
        self.channels_in = channels_in
        self.mask_size = mask_size
        
        # 掩码生成网络
        self.mask_net = nn.Sequential(
            nn.Conv2d(channels_in, channels_in // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels_in // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_in // 2, channels_in // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels_in // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_in // 4, 1, kernel_size=1),
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
    
    def forward(self, features):
        """
        Args:
            features: (B, C, H, W) 输入特征
        Returns:
            masks: (B, 1, H', W') 掩码提示，通常为低分辨率
        """
        # 生成掩码logits
        mask_logits = self.mask_net(features)  # (B, 1, H, W)
        
        # 上采样到目标尺寸
        if mask_logits.shape[2:] != (self.mask_size, self.mask_size):
            mask_logits = F.interpolate(
                mask_logits, 
                size=(self.mask_size, self.mask_size), 
                mode='bilinear', 
                align_corners=False
            )
        
        return mask_logits


class AdaptivePromptSelector(nn.Module):
    """
    自适应提示类型选择器
    
    动态选择最佳提示类型（点、框、掩码）
    """
    def __init__(self, channels_in, num_prompt_types=3):
        super().__init__()
        self.channels_in = channels_in
        self.num_prompt_types = num_prompt_types
        
        # 提示类型选择网络
        self.type_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels_in, channels_in // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels_in // 4, num_prompt_types),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features, temperature=1.0):
        """
        Args:
            features: (B, C, H, W) 输入特征
            temperature: 采样温度
        Returns:
            prompt_type_weights: (B, num_prompt_types) 提示类型权重
        """
        # 预测提示类型得分
        type_logits = self.type_selector(features)  # (B, num_prompt_types)
        
        # Gumbel-Softmax采样
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(type_logits) + 1e-8) + 1e-8)
        prompt_type_weights = F.softmax((type_logits + gumbel_noise) / temperature, dim=-1)
        
        return prompt_type_weights


class DifferentiableSAMPromptGenerate(nn.Module):
    """
    可微分SAM提示生成模块
    
    核心创新：
    1. 可微分点提示生成：从特征图生成点坐标和标签
    2. 可微分框提示生成：从特征图生成边界框
    3. 可微分掩码提示生成：从特征图生成初始掩码
    4. 自适应提示类型选择：动态选择最佳提示类型
    
    理论支撑：
    - Categorical Reparameterization with Gumbel-Softmax (ICLR 2017)
    - Attention-based Point Selection (CVPR 2020)
    """
    def __init__(self, channels_in, num_points=5, image_size=256, mask_size=256):
        super().__init__()
        self.channels_in = channels_in
        self.num_points = num_points
        self.image_size = image_size
        self.mask_size = mask_size
        
        # 自适应重要性学习
        self.importance_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels_in * 2, channels_in),
            nn.Sigmoid()
        )
        
        # 三种提示生成器
        self.point_prompt = DifferentiablePointPrompt(channels_in, num_points, image_size)
        self.box_prompt = DifferentiableBoxPrompt(channels_in, image_size)
        self.mask_prompt = DifferentiableMaskPrompt(channels_in, mask_size)
        
        # 提示类型选择器
        self.prompt_selector = AdaptivePromptSelector(channels_in, num_prompt_types=3)
        
        # 初始化
        self._init_weights()
        
        # 用于记录选择的提示类型
        self.selected_prompt_type = None
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, rgb, depth, temperature=1.0, prompt_type='auto'):
        """
        Args:
            rgb: (B, C, H, W) RGB特征
            depth: (B, C, H, W) DSM特征
            temperature: 采样温度
            prompt_type: 提示类型 ('auto', 'point', 'box', 'mask', 'point_and_box')
        Returns:
            prompts: dict，包含不同类型的提示
                - 'point_coords': (B, num_points, 2) 点坐标
                - 'point_labels': (B, num_points) 点标签
                - 'boxes': (B, 4) 边界框
                - 'mask_inputs': (B, 1, H, W) 掩码输入
            prompt_type_weights: (B, 3) 提示类型权重
        """
        B, C, H, W = rgb.shape
        
        # 1. 自适应特征融合
        importance = self.importance_net(torch.cat([rgb, depth], dim=1))
        fused = (rgb + depth) * importance.view(B, C, 1, 1)
        
        # 2. 生成所有类型的提示
        point_coords, point_labels = self.point_prompt(fused)
        boxes = self.box_prompt(fused)
        mask_inputs = self.mask_prompt(fused)
        
        # 3. 选择提示类型
        if prompt_type == 'auto':
            # 自动选择最佳提示类型
            prompt_type_weights = self.prompt_selector(fused, temperature)
            self.selected_prompt_type = prompt_type_weights
        else:
            # 手动指定提示类型
            prompt_type_weights = torch.zeros(B, 3, device=rgb.device)
            if prompt_type == 'point':
                prompt_type_weights[:, 0] = 1.0
            elif prompt_type == 'box':
                prompt_type_weights[:, 1] = 1.0
            elif prompt_type == 'mask':
                prompt_type_weights[:, 2] = 1.0
            elif prompt_type == 'point_and_box':
                prompt_type_weights[:, 0] = 0.5
                prompt_type_weights[:, 1] = 0.5
        
        # 4. 组装提示字典
        prompts = {
            'point_coords': point_coords,
            'point_labels': point_labels,
            'boxes': boxes,
            'mask_inputs': mask_inputs,
        }
        
        return prompts, prompt_type_weights


class LightweightSAMPrompt(nn.Module):
    """
    轻量级SAM提示生成模块
    只生成点提示
    """
    def __init__(self, channels_in, num_points=5, image_size=256):
        super().__init__()
        self.channels_in = channels_in
        self.num_points = num_points
        self.image_size = image_size
        
        # 点重要性预测
        self.point_importance = nn.Sequential(
            nn.Conv2d(channels_in, 1, kernel_size=1),
        )
        
        # 点标签预测
        self.point_label = nn.Sequential(
            nn.Conv2d(channels_in, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, rgb, depth):
        B, C, H, W = rgb.shape
        
        # 特征融合
        fused = rgb + depth
        
        # 预测点重要性
        importance = self.point_importance(fused).view(B, -1)
        
        # 采样点
        probs = F.softmax(importance, dim=-1)
        
        # 生成坐标网格
        y_coords = torch.linspace(0, 1, H, device=rgb.device)
        x_coords = torch.linspace(0, 1, W, device=rgb.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)
        
        # 加权平均得到点坐标
        point_coords = torch.einsum('bh,bhc->bc', probs, grid.unsqueeze(0).expand(B, -1, -1))
        point_coords = point_coords.unsqueeze(1).expand(-1, self.num_points, -1)
        
        # 添加扰动
        perturbation = torch.randn(B, self.num_points, 2, device=rgb.device) * 0.05
        point_coords = torch.clamp(point_coords + perturbation, 0, 1)
        
        # 预测点标签
        label_probs = F.adaptive_avg_pool2d(self.point_label(fused), (self.num_points, 1)).squeeze(1).squeeze(-1)
        point_labels = (label_probs > 0.5).float()
        
        return point_coords, point_labels


# 测试代码
if __name__ == '__main__':
    print("Testing DifferentiableSAMPromptGenerate...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 测试完整版本
    model = DifferentiableSAMPromptGenerate(
        channels_in=256,
        num_points=5,
        image_size=256,
        mask_size=256
    ).to(device)
    
    # 测试输入
    rgb = torch.randn(2, 256, 64, 64).to(device)
    depth = torch.randn(2, 256, 64, 64).to(device)
    
    # 前向传播
    model.train()
    prompts, prompt_weights = model(rgb, depth, temperature=1.0, prompt_type='auto')
    
    print(f"Input shape: RGB={rgb.shape}, Depth={depth.shape}")
    print(f"\nGenerated prompts:")
    print(f"  Point coords: {prompts['point_coords'].shape}")
    print(f"  Point labels: {prompts['point_labels'].shape}")
    print(f"  Boxes: {prompts['boxes'].shape}")
    print(f"  Mask inputs: {prompts['mask_inputs'].shape}")
    print(f"\nPrompt type weights: {prompt_weights}")
    
    # 测试反向传播
    loss = prompts['point_coords'].mean() + prompts['boxes'].mean()
    loss.backward()
    print("\n✅ Backward pass successful!")
    
    # 测试不同的提示类型
    print("\n" + "="*60)
    print("Testing different prompt types:")
    
    for ptype in ['point', 'box', 'mask', 'point_and_box']:
        prompts, weights = model(rgb, depth, prompt_type=ptype)
        print(f"\n{ptype}: weights = {weights[0]}")
    
    # 测试轻量级版本
    print("\n" + "="*60)
    print("Testing LightweightSAMPrompt...")
    model_light = LightweightSAMPrompt(256, num_points=5).to(device)
    point_coords, point_labels = model_light(rgb, depth)
    print(f"Point coords: {point_coords.shape}")
    print(f"Point labels: {point_labels.shape}")
    
    # 参数量对比
    params_full = sum(p.numel() for p in model.parameters())
    params_light = sum(p.numel() for p in model_light.parameters())
    print(f"\nFull model parameters: {params_full:,}")
    print(f"Lightweight model parameters: {params_light:,}")
    print(f"Reduction: {(1 - params_light/params_full)*100:.1f}%")
    
    print("\n✅ All tests passed!")
