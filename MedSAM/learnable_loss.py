import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableMultiTaskLoss(nn.Module):

    
    def __init__(self, num_classes, num_losses=5, initial_weights=None):
        super().__init__()
        self.num_classes = num_classes
        
        # 可学习的对数方差参数（使用对数保证正值）
        # sigma^2 越大，权重越小
        if initial_weights is None:
            initial_weights = [1.0, 1.0, 0.5, 0.5, 0.5]  # ce, dice, focal, tversky, boundary
        
        # 初始化为 log(sigma^2)
        # weight = 1 / (2 * sigma^2)
        # log_sigma2 = log(1 / (2 * weight))
        initial_log_sigma2 = []
        for w in initial_weights:
            sigma2 = 1.0 / (2.0 * w + 1e-6)
            initial_log_sigma2.append(torch.log(torch.tensor(sigma2)))
        
        self.log_sigma2 = nn.Parameter(torch.tensor(initial_log_sigma2), requires_grad=True)
        
        # Laplacian算子用于边界检测
        self.laplacian = nn.Parameter(
            torch.tensor([
                [0., 1., 0.],
                [1., -4., 1.],
                [0., 1., 0.]
            ], dtype=torch.float32).view(1, 1, 3, 3),
            requires_grad=False
        )
        
    def forward(self, pred, target):
        """
        前向传播
        自动计算各个损失并加权
        
        Args:
            pred: (B, C, H, W) 预测logits
            target: (B, H, W) 真实标签
        
        Returns:
            total_loss: 加权总损失
            losses: 各个损失的字典
        """
        losses = {}
        
        # 1. Cross Entropy Loss
        ce_loss = F.cross_entropy(pred, target)
        losses['ce_loss'] = ce_loss
        
        # 2. Dice Loss
        dice_loss = self.dice_loss(pred, target)
        losses['dice_loss'] = dice_loss
        
        # 3. Focal Loss
        focal_loss = self.focal_loss(pred, target)
        losses['focal_loss'] = focal_loss
        
        # 4. Tversky Loss
        tversky_loss = self.tversky_loss(pred, target)
        losses['tversky_loss'] = tversky_loss
        
        # 5. Boundary Loss
        boundary_loss = self.boundary_loss(pred, target)
        losses['boundary_loss'] = boundary_loss
        
        # 计算可学习权重
        # weight = 1 / (2 * sigma^2) = 1 / (2 * exp(log_sigma2))
        # 总损失 = sum(weight_i * loss_i) + 0.5 * sum(log_sigma2_i)
        # 第二项是正则化项，防止sigma过大
        precision = torch.exp(-self.log_sigma2)  # 1 / sigma^2
        
        total_loss = 0.0
        loss_values = [ce_loss, dice_loss, focal_loss, tversky_loss, boundary_loss]
        
        for i, (loss_val, prec) in enumerate(zip(loss_values, precision)):
            weight = 0.5 * prec  # 1 / (2 * sigma^2)
            total_loss = total_loss + weight * loss_val
        
        # 添加正则化项
        total_loss = total_loss + 0.5 * self.log_sigma2.sum()
        
        losses['total_loss'] = total_loss
        
        # 记录当前权重（用于监控）
        with torch.no_grad():
            weights = 0.5 * precision
            losses['weight_ce'] = weights[0]
            losses['weight_dice'] = weights[1]
            losses['weight_focal'] = weights[2]
            losses['weight_tversky'] = weights[3]
            losses['weight_boundary'] = weights[4]
        
        return total_loss, losses
    
    def dice_loss(self, pred, target):
        """Dice Loss - 优化mIoU"""
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + 1e-6) / (union + 1e-6)
        
        return 1 - dice.mean()
    
    def focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        """Focal Loss - 关注难样本"""
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    def tversky_loss(self, pred, target, alpha=0.3, beta=0.7):
        """Tversky Loss - 优化F1"""
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        
        TP = (pred * target_one_hot).sum(dim=(2, 3))
        FP = (pred * (1 - target_one_hot)).sum(dim=(2, 3))
        FN = ((1 - pred) * target_one_hot).sum(dim=(2, 3))
        
        tversky = (TP + 1e-6) / (TP + alpha * FP + beta * FN + 1e-6)
        
        return 1 - tversky.mean()
    
    def boundary_loss(self, pred, target):
        """Boundary Loss - 优化边界"""
        # 提取边界
        target_boundary = F.conv2d(
            target.unsqueeze(1).float(), 
            self.laplacian, 
            padding=1
        ).abs()
        target_boundary = (target_boundary > 0.5).float()
        
        # 预测边界
        pred_softmax = F.softmax(pred, dim=1)
        pred_boundary = F.conv2d(
            pred_softmax,
            self.laplacian.expand(self.num_classes, 1, 3, 3),
            padding=1,
            groups=self.num_classes
        ).abs()
        pred_boundary = pred_boundary.mean(dim=1, keepdim=True)
        pred_boundary = (pred_boundary > 0.5).float()
        
        return F.binary_cross_entropy(pred_boundary, target_boundary)


class AdaptiveWeightLoss(nn.Module):
    """
    自适应权重损失函数
    使用梯度标准化（GradNorm）自动调整权重
    
    基于论文:
    "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks" (ICML 2018)
    """
    
    def __init__(self, num_classes, num_losses=5, alpha=1.5):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha  # 控制权重调整速度
        
        # 初始权重
        self.weights = nn.Parameter(torch.ones(num_losses), requires_grad=True)
        
        # 记录初始损失（用于归一化）
        self.register_buffer('initial_losses', torch.ones(num_losses))
        self.initialized = False
        
        # Laplacian算子
        self.laplacian = nn.Parameter(
            torch.tensor([
                [0., 1., 0.],
                [1., -4., 1.],
                [0., 1., 0.]
            ], dtype=torch.float32).view(1, 1, 3, 3),
            requires_grad=False
        )
        
    def forward(self, pred, target):
        losses = {}
        
        # 计算各个损失
        ce_loss = F.cross_entropy(pred, target)
        dice_loss = self.dice_loss(pred, target)
        focal_loss = self.focal_loss(pred, target)
        tversky_loss = self.tversky_loss(pred, target)
        boundary_loss = self.boundary_loss(pred, target)
        
        loss_values = [ce_loss, dice_loss, focal_loss, tversky_loss, boundary_loss]
        loss_names = ['ce_loss', 'dice_loss', 'focal_loss', 'tversky_loss', 'boundary_loss']
        
        # 记录初始损失
        if not self.initialized:
            with torch.no_grad():
                for i, loss_val in enumerate(loss_values):
                    self.initial_losses[i] = loss_val.detach()
            self.initialized = True
        
        # 归一化权重
        weights_norm = F.softmax(self.weights, dim=0)
        
        # 计算加权总损失
        total_loss = 0.0
        for i, (loss_val, name, w) in enumerate(zip(loss_values, loss_names, weights_norm)):
            losses[name] = loss_val
            total_loss = total_loss + w * loss_val
        
        losses['total_loss'] = total_loss
        
        # 记录当前权重
        with torch.no_grad():
            for i, name in enumerate(loss_names):
                losses[f'weight_{name}'] = weights_norm[i]
        
        return total_loss, losses
    
    def update_weights(self, losses, shared_params):
        """
        使用GradNorm更新权重
        需要在backward之后调用
        
        Args:
            losses: 各个损失的列表
            shared_params: 共享参数的梯度
        """
        with torch.no_grad():
            # 计算每个损失的梯度范数
            grad_norms = []
            for loss in losses:
                grad = torch.autograd.grad(loss, shared_params, retain_graph=True)
                grad_norm = torch.norm(torch.cat([g.view(-1) for g in grad]))
                grad_norms.append(grad_norm)
            
            # 计算平均梯度范数
            avg_grad_norm = sum(grad_norms) / len(grad_norms)
            
            # 计算相对损失变化率
            loss_ratios = []
            for i, loss in enumerate(losses):
                loss_ratio = loss.detach() / self.initial_losses[i]
                loss_ratios.append(loss_ratio)
            
            avg_loss_ratio = sum(loss_ratios) / len(loss_ratios)
            
            # 计算目标梯度范数
            target_grad_norms = []
            for i, (grad_norm, loss_ratio) in enumerate(zip(grad_norms, loss_ratios)):
                inverse_train_rate = loss_ratio / avg_loss_ratio
                target = avg_grad_norm * (inverse_train_rate ** self.alpha)
                target_grad_norms.append(target)
            
            # 更新权重
            for i, (grad_norm, target) in enumerate(zip(grad_norms, target_grad_norms)):
                grad_weight = torch.abs(grad_norm - target)
                self.weights.data[i] += 0.01 * grad_weight
    
    def dice_loss(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice = (2. * intersection + 1e-6) / (union + 1e-6)
        return 1 - dice.mean()
    
    def focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        return (alpha * (1 - pt) ** gamma * ce_loss).mean()
    
    def tversky_loss(self, pred, target, alpha=0.3, beta=0.7):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        TP = (pred * target_one_hot).sum(dim=(2, 3))
        FP = (pred * (1 - target_one_hot)).sum(dim=(2, 3))
        FN = ((1 - pred) * target_one_hot).sum(dim=(2, 3))
        tversky = (TP + 1e-6) / (TP + alpha * FP + beta * FN + 1e-6)
        return 1 - tversky.mean()
    
    def boundary_loss(self, pred, target):
        target_boundary = F.conv2d(target.unsqueeze(1).float(), self.laplacian, padding=1).abs()
        target_boundary = (target_boundary > 0.5).float()
        pred_softmax = F.softmax(pred, dim=1)
        pred_boundary = F.conv2d(
            pred_softmax,
            self.laplacian.expand(self.num_classes, 1, 3, 3),
            padding=1,
            groups=self.num_classes
        ).abs()
        pred_boundary = pred_boundary.mean(dim=1, keepdim=True)
        pred_boundary = (pred_boundary > 0.5).float()
        return F.binary_cross_entropy(pred_boundary, target_boundary)


def test_learnable_loss():
    """测试可学习权重损失函数"""
    batch_size = 4
    num_classes = 6
    height, width = 256, 256
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    pred = torch.randn(batch_size, num_classes, height, width).to(device)
    target = torch.randint(0, num_classes, (batch_size, height, width)).to(device)
    
    print("\nTesting LearnableMultiTaskLoss...")
    criterion = LearnableMultiTaskLoss(num_classes).to(device)
    optimizer = torch.optim.SGD([pred.requires_grad_()], lr=0.01)
    
    # 模拟训练几步
    for step in range(3):
        optimizer.zero_grad()
        loss, losses = criterion(pred, target)
        loss.backward()
        optimizer.step()
        
        print(f"\nStep {step+1}:")
        print(f"  Total loss: {loss.item():.4f}")
        print(f"  Weights: CE={losses['weight_ce'].item():.3f}, "
              f"Dice={losses['weight_dice'].item():.3f}, "
              f"Focal={losses['weight_focal'].item():.3f}, "
              f"Tversky={losses['weight_tversky'].item():.3f}, "
              f"Boundary={losses['weight_boundary'].item():.3f}")
    
    print("\n" + "="*50)
    print("Testing AdaptiveWeightLoss...")
    criterion = AdaptiveWeightLoss(num_classes).to(device)
    
    for step in range(3):
        optimizer.zero_grad()
        loss, losses = criterion(pred, target)
        loss.backward()
        optimizer.step()
        
        print(f"\nStep {step+1}:")
        print(f"  Total loss: {loss.item():.4f}")
        print(f"  Weights: CE={losses['weight_ce_loss'].item():.3f}, "
              f"Dice={losses['weight_dice_loss'].item():.3f}, "
              f"Focal={losses['weight_focal_loss'].item():.3f}, "
              f"Tversky={losses['weight_tversky_loss'].item():.3f}, "
              f"Boundary={losses['weight_boundary_loss'].item():.3f}")


if __name__ == '__main__':
    test_learnable_loss()
