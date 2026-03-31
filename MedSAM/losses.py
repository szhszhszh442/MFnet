import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """
    多任务损失函数
    针对F1和mIoU指标优化
    
    包含:
    1. Cross Entropy Loss - 基础分类损失
    2. Dice Loss - 优化mIoU
    3. Tversky Loss - 优化F1分数
    4. Focal Loss - 关注难分类样本
    5. Boundary Loss - 优化边界
    """
    
    def __init__(self, num_classes, weights=None, 
                 ce_weight=1.0, dice_weight=1.0, focal_weight=0.5, 
                 tversky_weight=0.5, boundary_weight=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.weights = weights
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.boundary_weight = boundary_weight
        
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
        
        ce_loss = F.cross_entropy(pred, target, weight=self.weights)
        losses['ce_loss'] = ce_loss
        
        dice_loss = self.dice_loss(pred, target)
        losses['dice_loss'] = dice_loss
        
        focal_loss = self.focal_loss(pred, target)
        losses['focal_loss'] = focal_loss
        
        tversky_loss = self.tversky_loss(pred, target)
        losses['tversky_loss'] = tversky_loss
        
        boundary_loss = self.boundary_loss(pred, target)
        losses['boundary_loss'] = boundary_loss
        
        total_loss = (self.ce_weight * ce_loss + 
                      self.dice_weight * dice_loss +
                      self.focal_weight * focal_loss +
                      self.tversky_weight * tversky_loss +
                      self.boundary_weight * boundary_loss)
        
        losses['total_loss'] = total_loss
        
        return total_loss, losses
    
    def dice_loss(self, pred, target):
        """
        Dice Loss for mIoU optimization
        Dice = 2*TP / (2*TP + FP + FN)
        mIoU = TP / (TP + FP + FN)
        """
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + 1e-6) / (union + 1e-6)
        
        dice_loss = 1 - dice.mean()
        
        return dice_loss
    
    def focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        """
        Focal Loss for hard examples
        FL = -alpha * (1-pt)^gamma * log(pt)
        
        Args:
            alpha: weight for positive/negative examples
            gamma: focusing parameter (higher = more focus on hard examples)
        """
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        
        pt = torch.exp(-ce_loss)
        
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        
        return focal_loss.mean()
    
    def tversky_loss(self, pred, target, alpha=0.3, beta=0.7):
        """
        Tversky Loss for F1 optimization
        Tversky = TP / (TP + alpha*FP + beta*FN)
        
        F1 = 2*TP / (2*TP + FP + FN)
        
        When alpha + beta = 1, Tversky is similar to Dice
        When alpha < beta, we penalize FN more (improve recall)
        When alpha > beta, we penalize FP more (improve precision)
        
        Args:
            alpha: weight for false positives (FP)
            beta: weight for false negatives (FN)
        """
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        
        TP = (pred * target_one_hot).sum(dim=(2, 3))
        FP = (pred * (1 - target_one_hot)).sum(dim=(2, 3))
        FN = ((1 - pred) * target_one_hot).sum(dim=(2, 3))
        
        tversky = (TP + 1e-6) / (TP + alpha * FP + beta * FN + 1e-6)
        
        tversky_loss = 1 - tversky.mean()
        
        return tversky_loss
    
    def boundary_loss(self, pred, target):
        """
        Boundary Loss for edge optimization
        Uses Laplacian operator to detect boundaries
        """
        target_boundary = F.conv2d(
            target.unsqueeze(1).float(), 
            self.laplacian, 
            padding=1
        ).abs()
        
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
        
        boundary_loss = F.binary_cross_entropy(pred_boundary, target_boundary)
        
        return boundary_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for mIoU optimization
    """
    def __init__(self, num_classes, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice.mean()


class TverskyLoss(nn.Module):
    """
    Tversky Loss for F1 optimization
    """
    def __init__(self, num_classes, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        
        TP = (pred * target_one_hot).sum(dim=(2, 3))
        FP = (pred * (1 - target_one_hot)).sum(dim=(2, 3))
        FN = ((1 - pred) * target_one_hot).sum(dim=(2, 3))
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1 - tversky.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for hard examples
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class BoundaryLoss(nn.Module):
    """
    Boundary Loss for edge optimization
    """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        self.laplacian = nn.Parameter(
            torch.tensor([
                [0., 1., 0.],
                [1., -4., 1.],
                [0., 1., 0.]
            ], dtype=torch.float32).view(1, 1, 3, 3),
            requires_grad=False
        )
    
    def forward(self, pred, target):
        target_boundary = F.conv2d(
            target.unsqueeze(1).float(), 
            self.laplacian, 
            padding=1
        ).abs()
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


class CombinedLoss(nn.Module):
    """
    Combined Loss with configurable weights
    """
    def __init__(self, num_classes, weights=None, 
                 loss_configs=None):
        """
        Args:
            num_classes: number of classes
            weights: class weights for cross entropy
            loss_configs: dict of loss configurations
                e.g., {
                    'ce': {'weight': 1.0},
                    'dice': {'weight': 1.0},
                    'focal': {'weight': 0.5, 'alpha': 0.25, 'gamma': 2.0},
                    'tversky': {'weight': 0.5, 'alpha': 0.3, 'beta': 0.7},
                    'boundary': {'weight': 0.5}
                }
        """
        super().__init__()
        self.num_classes = num_classes
        self.weights = weights
        
        if loss_configs is None:
            loss_configs = {
                'ce': {'weight': 1.0},
                'dice': {'weight': 1.0},
                'focal': {'weight': 0.5},
                'tversky': {'weight': 0.5},
                'boundary': {'weight': 0.5}
            }
        
        self.loss_configs = loss_configs
        self.losses = nn.ModuleDict()
        
        if 'ce' in loss_configs:
            self.losses['ce'] = None
        
        if 'dice' in loss_configs:
            self.losses['dice'] = DiceLoss(num_classes)
        
        if 'focal' in loss_configs:
            config = loss_configs['focal']
            self.losses['focal'] = FocalLoss(
                alpha=config.get('alpha', 0.25),
                gamma=config.get('gamma', 2.0)
            )
        
        if 'tversky' in loss_configs:
            config = loss_configs['tversky']
            self.losses['tversky'] = TverskyLoss(
                num_classes,
                alpha=config.get('alpha', 0.3),
                beta=config.get('beta', 0.7)
            )
        
        if 'boundary' in loss_configs:
            self.losses['boundary'] = BoundaryLoss(num_classes)
    
    def forward(self, pred, target):
        losses = {}
        total_loss = 0.0
        
        if 'ce' in self.losses:
            ce_loss = F.cross_entropy(pred, target, weight=self.weights)
            losses['ce_loss'] = ce_loss
            total_loss += self.loss_configs['ce']['weight'] * ce_loss
        
        if 'dice' in self.losses:
            dice_loss = self.losses['dice'](pred, target)
            losses['dice_loss'] = dice_loss
            total_loss += self.loss_configs['dice']['weight'] * dice_loss
        
        if 'focal' in self.losses:
            focal_loss = self.losses['focal'](pred, target)
            losses['focal_loss'] = focal_loss
            total_loss += self.loss_configs['focal']['weight'] * focal_loss
        
        if 'tversky' in self.losses:
            tversky_loss = self.losses['tversky'](pred, target)
            losses['tversky_loss'] = tversky_loss
            total_loss += self.loss_configs['tversky']['weight'] * tversky_loss
        
        if 'boundary' in self.losses:
            boundary_loss = self.losses['boundary'](pred, target)
            losses['boundary_loss'] = boundary_loss
            total_loss += self.loss_configs['boundary']['weight'] * boundary_loss
        
        losses['total_loss'] = total_loss
        
        return total_loss, losses


def test_losses():
    """Test loss functions"""
    batch_size = 4
    num_classes = 6
    height, width = 256, 256
    
    pred = torch.randn(batch_size, num_classes, height, width)
    target = torch.randint(0, num_classes, (batch_size, height, width))
    
    print("Testing MultiTaskLoss...")
    criterion = MultiTaskLoss(num_classes)
    loss, losses = criterion(pred, target)
    print(f"Total loss: {loss.item():.4f}")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.4f}")
    
    print("\nTesting CombinedLoss...")
    criterion = CombinedLoss(num_classes)
    loss, losses = criterion(pred, target)
    print(f"Total loss: {loss.item():.4f}")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.4f}")
    
    print("\nTesting individual losses...")
    dice_loss = DiceLoss(num_classes)
    print(f"Dice loss: {dice_loss(pred, target).item():.4f}")
    
    tversky_loss = TverskyLoss(num_classes)
    print(f"Tversky loss: {tversky_loss(pred, target).item():.4f}")
    
    focal_loss = FocalLoss()
    print(f"Focal loss: {focal_loss(pred, target).item():.4f}")
    
    boundary_loss = BoundaryLoss(num_classes)
    print(f"Boundary loss: {boundary_loss(pred, target).item():.4f}")


if __name__ == '__main__':
    test_losses()
