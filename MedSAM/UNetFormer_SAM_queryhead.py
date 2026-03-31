import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
import cv2
import torch.autograd as autograd
from MedSAM.models.sam import sam_model_registry
import MedSAM.cfg as cfg
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from MedSAM.multi_scale_adaptive_fusion import MultiScaleAdaptiveDynamicConvFusion


class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
    
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalLocalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3*dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1,  padding=(window_size//2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size//2 - 1))

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.ws - 1
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape

        local = self.local2(x) + self.local1(x)

        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]

        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        out = out[:, :, :H, :W]

        return out

class Block(nn.Module):
    def __init__(self, dim=256, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class WF_single(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF_single, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.post_conv(x)
        return x
    
class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y
    
class DynamicConvFusion(nn.Module):
    def __init__(self, channels_in, num_kernels=8, kernel_size=3, activation=nn.ReLU(inplace=True)):
        super(DynamicConvFusion, self).__init__()
        self.channels_in = channels_in
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size

        self.depth_conv = nn.Sequential(
            nn.Conv2d(channels_in, channels_in, kernel_size=3, padding=1),
            activation
        )

        self.mlp = nn.Sequential(
            nn.Linear(channels_in, channels_in),
            activation,
            nn.Linear(channels_in, num_kernels)
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels_in * 2, channels_in, kernel_size=1),
            activation
        )

        self.hpf_conv = nn.Sequential(
            nn.Conv2d(channels_in, channels_in, kernel_size=3, padding=1),
            activation
        )

        self.pre_kernels = nn.Parameter(torch.randn(num_kernels, channels_in, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.pre_kernels, mode='fan_out', nonlinearity='relu')

        self.static_baseline = nn.Parameter(torch.randn(num_kernels, channels_in, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.static_baseline, mode='fan_out', nonlinearity='relu')

    def get_diversity_loss(self):
        kernels = self.pre_kernels.view(self.num_kernels, -1)
        kernels = F.normalize(kernels, dim=1)
        similarity = torch.mm(kernels, kernels.t())
        pairwise_cos = similarity
        off_diagonal = pairwise_cos - torch.eye(self.num_kernels, device=kernels.device)
        diversity_loss = off_diagonal.abs().mean()
        return diversity_loss

    def forward(self, rgb, depth):
        B, C, H, W = rgb.shape

        G = self.depth_conv(depth)
        g = F.adaptive_avg_pool2d(G, 1).view(B, C)

        alpha = self.mlp(g)
        alpha = F.softmax(alpha, dim=1)

        dyn_kernels = self.pre_kernels + self.static_baseline
        W_dyn = torch.einsum('bn,kcij->bcij', alpha, dyn_kernels)

        F_dyn = F.conv2d(rgb.reshape(1, B * C, H, W),
                         W_dyn.reshape(B * C, 1, self.kernel_size, self.kernel_size),
                         padding=self.kernel_size//2,
                         groups=B * C)

        F_dyn = F_dyn.view(B, C, H, W)

        H_dsm = depth - F.avg_pool2d(depth, kernel_size=3, stride=1, padding=1)
        H_dsm_feat = self.hpf_conv(H_dsm)

        F_out = F_dyn + H_dsm_feat

        out = self.fusion_conv(torch.cat([F_out, depth], dim=1))

        return out

  
class FeatureRefinementHead_single(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.pa = nn.Sequential(nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
                                nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                Conv(decode_channels, decode_channels//16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(decode_channels//16, decode_channels, kernel_size=1),
                                nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)

        return x
      
class FeatureRefinementHead(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.pa = nn.Sequential(nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
                                nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                Conv(decode_channels, decode_channels//16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(decode_channels//16, decode_channels, kernel_size=1),
                                nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)

        return x

class AuxHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        return feat


# ============================================================
# Query-Based 语义分割头 (新添加)
# ============================================================
class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats: int = 128):
        super().__init__()
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            torch.randn(2, num_pos_feats),
        )
    
    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        
        y_embed = y_embed / h
        x_embed = x_embed / w
        
        coords = torch.stack([x_embed, y_embed], dim=-1)
        pe = self._pe_encoding(coords)
        return pe.permute(2, 0, 1)
    
    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * torch.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)


class QueryBasedSegmentationHead(nn.Module):
    """
    Query-Based 语义分割头
    核心思想：
    1. 使用可学习的类别Query (类似DETR)
    2. 通过Transformer融合图像特征和Prompt信息
    3. 输出直接是语义分割结果
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 6,
        num_queries: int = None,
        num_heads: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries if num_queries is not None else num_classes
        self.embed_dim = embed_dim
        
        self.input_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.input_norm = nn.BatchNorm2d(embed_dim)
        self.input_act = nn.GELU()
        
        self.dense_proj = nn.Conv2d(256, embed_dim, kernel_size=1)
        
        self.pos_encoding = PositionEmbeddingRandom(embed_dim // 2)
        
        self.class_queries = nn.Embedding(self.num_queries, embed_dim)
        
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        
        self.sparse_proj = nn.Linear(256, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.image_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.query_transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        
        
        self.seg_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        )

        self.class_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )
        self.sparse_cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )
        self.class_query_norm = nn.LayerNorm(embed_dim)
        self.feat_proj = nn.Linear(embed_dim, embed_dim)
        self.feat_norm = nn.LayerNorm(embed_dim)
        self.class_token_proj = nn.Linear(embed_dim, 1)
        self.fuse_gate = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 4, num_classes, kernel_size=1),
            nn.Sigmoid()
        )
        self.dense_gate = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(
        self,
        fpn_features: torch.Tensor,
        sparse_prompt_emb: Optional[torch.Tensor] = None,
        dense_prompt_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = fpn_features.shape[0]
        H, W = fpn_features.shape[2], fpn_features.shape[3]

        src = self.input_proj(fpn_features)
        src = self.input_norm(src)
        src = self.input_act(src)

        if dense_prompt_emb is not None:
            dense_prompt_proj = self.dense_proj(dense_prompt_emb)
            gate = torch.sigmoid(self.dense_gate(dense_prompt_proj))
            src = src + gate * dense_prompt_proj

        pos_embed = self.pos_encoding(src.shape[-2:])
        pos_embed = pos_embed.permute(1, 2, 0).unsqueeze(0).expand(B, -1, -1, -1)

        src_flat = src.flatten(2).permute(0, 2, 1)
        pos_flat = pos_embed.flatten(2).permute(0, 2, 1)

        memory = self.image_transformer(src_flat + pos_flat)

        queries = self.class_queries.weight
        queries = self.query_proj(queries)
        queries = queries.unsqueeze(0).expand(B, -1, -1)

        if sparse_prompt_emb is not None:
            sparse_prompt_proj = self.sparse_proj(sparse_prompt_emb)
            attn_out, _ = self.sparse_cross_attn(
                query=queries,
                key=sparse_prompt_proj,
                value=sparse_prompt_proj
            )
            queries = queries + attn_out

        tgt = queries
        outputs = self.query_transformer(tgt, memory)

        src_for_seg = src_flat.permute(0, 2, 1).view(B, self.embed_dim, H, W)

        x = self.upsample1(src_for_seg)
        x = self.upsample2(x)
        

        class_queries = self.class_query_norm(outputs)
        feat_high_flat = x.flatten(2).permute(0, 2, 1)
        feat_high_flat = self.feat_proj(feat_high_flat)
        feat_high_flat = self.feat_norm(feat_high_flat)

        class_queries_for_seg = class_queries[:, :self.num_classes, :]
        mask_logits = torch.einsum("bqc,bkc->bqk", class_queries_for_seg, feat_high_flat)
        mask_logits = mask_logits.view(B, self.num_classes, x.shape[2], x.shape[3])

        seg_logits = self.seg_head(x)
        gate = self.fuse_gate(x)
        seg_logits = seg_logits * (1 - gate) + mask_logits * gate

        return seg_logits


# ============================================================
# 支持Prompt的Decoder (新添加)
# ============================================================
class PromptDecoder(nn.Module):
    """
    支持Prompt的Decoder
    结合了UNetFormer的原始Decoder结构和Query-Based方法
    """
    def __init__(
        self,
        encoder_channels=(64, 128, 256, 512),
        decode_channels=64,
        dropout=0.1,
        window_size=8,
        num_classes=6,
        use_query_based=True,
    ):
        super(PromptDecoder, self).__init__()
        
        self.use_query_based = use_query_based
        
        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.b4 = Block(dim=decode_channels, num_heads=8, window_size=window_size)

        self.b3 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p3 = WF(encoder_channels[-2], decode_channels)

        self.b2 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p2 = WF(encoder_channels[-3], decode_channels)

        self.p1 = FeatureRefinementHead(encoder_channels[-4], decode_channels)

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        
        if use_query_based:
            self.query_head = QueryBasedSegmentationHead(
                embed_dim=decode_channels,
                num_classes=num_classes,
                num_queries=100,
                num_heads=8,
                num_layers=2
            )
        
        self.init_weight()

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, res1, res2, res3, res4, h, w,
                sparse_prompt_emb=None, dense_prompt_emb=None):      

        
        x = self.b4(self.pre_conv(res4))

        x = self.p3(x, res3)
        x = self.b3(x)

        x = self.p2(x, res2)
        x = self.b2(x)

        x = self.p1(x, res1)
        if self.use_query_based and sparse_prompt_emb is not None:
            x = self.query_head(
                fpn_features=x,
                sparse_prompt_emb=sparse_prompt_emb,
                dense_prompt_emb=dense_prompt_emb
            )
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
            return x        
        x = self.segmentation_head(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        return x


class PromptGenerate(nn.Module):
    """
    Wavelet-Edge Structural Prompting Module (WESP-M)
    小波-结构边缘协同提示生成模块
    
    三个阶段：
    1. 频域特征解耦 (FD-D): 2D-DWT 分解
    2. 跨模态高频融合 (CM-HF): 高频能量融合
    3. 结构几何提示派生 (SG-PD): 生成 point/box/mask prompt
    """
    def __init__(self, num_points=10, threshold_ratio=0.5):
        super().__init__()
        self.num_points = num_points
        self.threshold_ratio = threshold_ratio
        
        self.mlpfreq = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        
    def dwt_2d(self, x):
        """
        2D 离散小波变换 (Haar wavelet)
        Args:
            x: (B, C, H, W)
        Returns:
            LL, (LH, HL, HH)
        """
        x01 = x[..., 0::2, :] / 2
        x02 = x[..., 1::2, :] / 2
        
        x_LL = x01[..., :, 0::2] + x02[..., :, 0::2]
        x_LH = x01[..., :, 0::2] - x02[..., :, 0::2]
        x_HL = x01[..., :, 1::2] + x02[..., :, 1::2]
        x_HH = x01[..., :, 1::2] - x02[..., :, 1::2]
        
        return x_LL, (x_LH, x_HL, x_HH)
    
    def compute_high_freq_energy(self, LL, LH, HL, HH):
        """
        计算高频能量图
        E = sqrt(LH^2 + HL^2 + HH^2)
        """
        energy = torch.sqrt(LH ** 2 + HL ** 2 + HH ** 2 + 1e-8)
        return energy
    
    def gaussian_smooth(self, x, kernel_size=5, sigma=1.0):
        """
        高斯平滑
        """
        B, C, H, W = x.size()
        kernel = self._gaussian_kernel(kernel_size, sigma, device=x.device)
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(C, 1, 1, 1)
        
        padding = kernel_size // 2
        x_smooth = F.conv2d(x, kernel, padding=padding, groups=C)
        return x_smooth
    
    def _gaussian_kernel(self, kernel_size, sigma, device):
        """
        生成高斯核
        """
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=device)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel
    
    def find_local_maxima(self, x, num_points):
        """
        基于空间NMS的局部极大值搜索，找到空间分布的显著点
        使用向量化操作和torch.cdist避免GPU-CPU同步
        输出归一化坐标 [0,1]，方便后续映射到任意尺寸
        Args:
            x: (B, C, H, W) - C 可以是 1 或 3 通道
        Returns:
            coords: (B, num_points, 2), 归一化坐标 (x=W/W=1, y=H/H=1)
            labels: (B, num_points), point_labels (全1表示前景)
        """
        if x.size(1) > 1:
            x = x.mean(dim=1, keepdim=True)

        B, _, H, W = x.size()
        x_flat = x.view(B, -1)

        coords = torch.zeros(B, num_points, 2, device=x.device, dtype=torch.float)
        labels = torch.ones(B, num_points, device=x.device, dtype=torch.long)

        min_distance = max(H, W) // 20
        min_distance = max(min_distance, 20)

        for b in range(B):
            saliency = x_flat[b]

            max_pool = F.max_pool2d(x[b], kernel_size=3, stride=1, padding=1)
            peaks = (x[b] == max_pool) & (x[b] > 0)

            peak_mask = peaks.squeeze(0)
            peak_values = saliency[peak_mask.view(-1)]

            if peak_values.numel() == 0:
                _, topk_idx_local = torch.topk(saliency, min(num_points, saliency.numel()), dim=0)
                selected_indices = topk_idx_local[:num_points]
            else:
                peak_indices = torch.where(peak_mask.view(-1))[0]
                num_peaks = peak_indices.shape[0]

                sorted_idx = torch.argsort(peak_values, descending=True)
                sorted_peaks = peak_indices[sorted_idx]

                peak_coords = torch.zeros(num_peaks, 2, device=x.device, dtype=torch.float)
                peak_coords[:, 0] = sorted_peaks % W
                peak_coords[:, 1] = torch.div(sorted_peaks, W, rounding_mode='trunc')

                dist_matrix = torch.cdist(peak_coords, peak_coords, p=2)

                available = torch.ones(num_peaks, dtype=torch.bool, device=x.device)
                chosen_indices = []

                for current_pos in range(num_peaks):
                    if len(chosen_indices) >= num_points:
                        break
                    if available[current_pos]:
                        chosen_indices.append(current_pos)
                        nearby = dist_matrix[current_pos] < min_distance
                        nearby[:current_pos + 1] = False
                        available[nearby] = False

                selected_peaks = sorted_peaks[torch.tensor(chosen_indices, device=x.device)]

                if selected_peaks.shape[0] < num_points:
                    remaining_needed = num_points - selected_peaks.shape[0]
                    remaining_indices = torch.argsort(saliency, descending=True)
                    mask_to_exclude = torch.zeros(saliency.numel(), dtype=torch.bool, device=saliency.device)
                    mask_to_exclude[sorted_peaks] = True
                    remaining_indices = remaining_indices[~mask_to_exclude[remaining_indices]]
                    selected_indices = torch.cat([selected_peaks, remaining_indices[:remaining_needed]])
                else:
                    selected_indices = selected_peaks[:num_points]

            if selected_indices.numel() < num_points:
                if selected_indices.numel() == 0:
                    selected_indices = torch.zeros(num_points, dtype=torch.long, device=x.device)
                else:
                    repeats_needed = torch.div(num_points + selected_indices.numel() - 1, selected_indices.numel(), rounding_mode='trunc')
                    selected_indices = selected_indices.repeat(repeats_needed)[:num_points]

            for i in range(num_points):
                idx = selected_indices[i]
                h_idx = torch.div(idx, W, rounding_mode='trunc')
                w_idx = idx - h_idx * W
                coords[b, i, 0] = w_idx / W
                coords[b, i, 1] = h_idx / H

        return coords, labels
    
    def find_largest_component_bbox(self, binary_mask, min_area=100):
        """
        找到最大连通区域的边界框 (纯PyTorch版本)
        使用BFS-like方法标记连通分量
        Args:
            binary_mask: (H, W) 二值掩码
            min_area: 最小面积阈值
        Returns:
            (x_min, y_min, x_max, y_max) 或默认值
        """
        H, W = binary_mask.shape

        labels = torch.zeros((H, W), dtype=torch.long, device=binary_mask.device)
        current_label = 1

        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for i in range(H):
            for j in range(W):
                if binary_mask[i, j] > 0.5 and labels[i, j] == 0:
                    stack = [(i, j)]
                    labels[stack[0]] = current_label
                    min_r, max_r, min_c, max_c = i, i, j, j

                    while stack:
                        r, c = stack.pop()
                        min_r = min(min_r, r)
                        max_r = max(max_r, r)
                        min_c = min(min_c, c)
                        max_c = max(max_c, c)

                        for dr, dc in dirs:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < H and 0 <= nc < W:
                                if binary_mask[nr, nc] > 0.5 and labels[nr, nc] == 0:
                                    labels[nr, nc] = current_label
                                    stack.append((nr, nc))

                    area = (max_r - min_r + 1) * (max_c - min_c + 1)
                    if area < min_area:
                        labels[labels == current_label] = 0
                    else:
                        current_label += 1

        if current_label == 1:
            return torch.tensor([0.0, 0.0, float(W - 1), float(H - 1)], device=binary_mask.device)

        label_counts = torch.bincount(labels.flatten().long())
        label_counts[0] = 0
        largest_label = label_counts.argmax().item()

        component_mask = (labels == largest_label)
        rows = torch.any(component_mask, dim=1)
        cols = torch.any(component_mask, dim=0)
        rmin, rmax = torch.where(rows)[0][[0, -1]]
        cmin, cmax = torch.where(cols)[0][[0, -1]]

        return torch.tensor([float(cmin), float(rmin), float(cmax), float(rmax)], device=binary_mask.device)

    def compute_boxes_from_mask(self, mask, min_area=100):
        """
        从二值掩码提取最小外接矩形框 (纯PyTorch版本)
        找到最大连通区域的边界框
        Args:
            mask: (B, 1, H, W)
        Returns:
            boxes: (B, 4) 或 None
        """
        B, _, H, W = mask.size()
        threshold = self.threshold_ratio
        binary = (mask > threshold).float()

        boxes_list = []
        for b in range(B):
            bbox = self.find_largest_component_bbox(binary[b, 0], min_area=min_area)
            boxes_list.append(bbox)

        boxes_tensor = torch.stack(boxes_list)
        return boxes_tensor
    
    def forward(self, rgb_image, depth_image):
        """
        Args:
            rgb_image: RGB 图像 (B, 3, H, W)
            depth_image: 深度图像 (B, 1, H, W) 或 (B, H, W)
        Returns:
            point_coords: 点坐标 (B, N, 2)
            point_labels: 点标签 (B, N)
            boxes: 边界框 (B, 4)
            masks: 掩码 (B, 1, H, W)
        """
        if depth_image.dim() == 3:
            depth_image = depth_image.unsqueeze(1)
        
        if depth_image.size(1) == 1 and depth_image.size(2) != rgb_image.size(2):
            depth_image = F.interpolate(depth_image, size=(rgb_image.size(2), rgb_image.size(3)), 
                                        mode='bilinear', align_corners=False)
        
        rgb_gray = rgb_image.mean(dim=1, keepdim=True)
        
        rgb_LL, (rgb_LH, rgb_HL, rgb_HH) = self.dwt_2d(rgb_gray)
        depth_LL, (depth_LH, depth_HL, depth_HH) = self.dwt_2d(depth_image)
        
        E_rgb = self.compute_high_freq_energy(rgb_LL, rgb_LH, rgb_HL, rgb_HH)
        E_depth = self.compute_high_freq_energy(depth_LL, depth_LH, depth_HL, depth_HH)
        
        freq_features = torch.cat([rgb_LH, rgb_HL, rgb_HH, depth_LH, depth_HL, depth_HH], dim=1)
        
        del rgb_LL, depth_LL
        
        E_fused = E_rgb * E_depth
        
        del E_rgb, E_depth
        
        E_fused_up = F.interpolate(E_fused, size=(rgb_image.size(2), rgb_image.size(3)), 
                                   mode='bilinear', align_corners=False)
        
        del E_fused
        
        E_smooth = self.gaussian_smooth(E_fused_up, kernel_size=5, sigma=2.0)
        
        saliency_map = E_smooth / (E_smooth.max() + 1e-8)
        
        del E_smooth
        
        point_coords, point_labels = self.find_local_maxima(saliency_map, self.num_points)

        threshold = saliency_map.max() * self.threshold_ratio
        binary_mask = (saliency_map > threshold).float()
        
        del saliency_map
        
        # boxes = self.compute_boxes_from_mask(binary_mask)
        
        freq_features_up = F.interpolate(freq_features, size=(rgb_image.size(2), rgb_image.size(3)), 
                                          mode='bilinear', align_corners=False)
        
        del freq_features
        
        masks = torch.sigmoid(self.mlpfreq(freq_features_up))
        
        return point_coords, point_labels, masks


class Decoder_single(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6):
        super(Decoder_single, self).__init__()

        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.b4 = Block(dim=decode_channels, num_heads=8, window_size=window_size)

        self.b3 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p3 = WF_single(encoder_channels[-2], decode_channels)

        self.b2 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p2 = WF_single(encoder_channels[-3], decode_channels)

        self.p1 = FeatureRefinementHead_single(encoder_channels[-4], decode_channels)

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()

    def forward(self, res4, h, w):
        x = self.b4(self.pre_conv(res4))
        x = self.p3(x)
        x = self.b3(x)

        x = self.p2(x)
        x = self.b2(x)

        x = self.p1(x)
        
        x = self.segmentation_head(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6):
        super(Decoder, self).__init__()

        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.b4 = Block(dim=decode_channels, num_heads=8, window_size=window_size)

        self.b3 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p3 = WF(encoder_channels[-2], decode_channels)

        self.b2 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p2 = WF(encoder_channels[-3], decode_channels)

        self.p1 = FeatureRefinementHead(encoder_channels[-4], decode_channels)

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w):
        x = self.b4(self.pre_conv(res4))
        x = self.p3(x, res3)
        x = self.b3(x)

        x = self.p2(x, res2)
        x = self.b2(x)

        x = self.p1(x, res1)
        
        x = self.segmentation_head(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def draw_features(feature, savename=''):
    H = W = 256
    visualize = F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=False)
    visualize = visualize.detach().cpu().numpy()
    visualize = np.mean(visualize, axis=1).reshape(H, W)
    visualize = (((visualize - np.min(visualize)) / (np.max(visualize) - np.min(visualize))) * 255).astype(np.uint8)
    savedir = savename
    visualize = cv2.applyColorMap(visualize, cv2.COLORMAP_JET)
    cv2.imwrite(savedir, visualize)

class UNetFormer(nn.Module):
    def __init__(self,
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6,
                 use_prompt_decoder=True
                 ):
        super().__init__()
        args = cfg.parse_args()
        self.sam = sam_model_registry["vit_l"](args,checkpoint='weights/sam_vit_l_0b3195.pth')
        self.image_encoder = self.sam.image_encoder
        self.prompt_encoder = self.sam.prompt_encoder
        encoder_channels = (256, 256, 256, 256)

        self.inter_embed_dim = self.image_encoder.embed_dim

        self.fpn1x = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            Norm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
        )
        self.fpn2x = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
        )
        self.fpn3x = nn.Identity()
        self.fpn4x = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fpn_interp_conv = nn.Conv2d(256, 256, kernel_size=1)
        self.neck_interp = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        self.inter_proj = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        self.inter_neck = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        self.inter_proj0 = nn.Conv2d(self.inter_embed_dim, 256, kernel_size=1)
        self.inter_proj1 = nn.Conv2d(self.inter_embed_dim, 256, kernel_size=1)
        self.inter_proj2 = nn.Conv2d(self.inter_embed_dim, 256, kernel_size=1)

        self.lateral_conv = nn.Conv2d(256, 256, kernel_size=1)
        self.fpnfusion_conv1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        self.fpnfusion_conv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        self.fpnfusion_conv3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        self.fpn_out_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        
        self.fpn1y = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
        )
        self.fpn2y = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
        )
        self.fpn3y = nn.Identity()
        self.fpn4y = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fusion1 = MultiScaleAdaptiveDynamicConvFusion(
            channels_in=encoder_channels[0],
            num_kernels=4,
            num_scales=3
        )
        self.fusion2 = MultiScaleAdaptiveDynamicConvFusion(encoder_channels[1])
        self.fusion3 = MultiScaleAdaptiveDynamicConvFusion(encoder_channels[2])
        self.fusion4 = MultiScaleAdaptiveDynamicConvFusion(encoder_channels[3])
        for n, value in self.image_encoder.named_parameters():
            if 'lora_' not in n:
                value.requires_grad = False
            else:
                value.requires_grad = True

        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

        if use_prompt_decoder:

            self.decoder = PromptDecoder(encoder_channels, decode_channels, dropout, window_size, num_classes)
            self.prompt_generate = PromptGenerate()
        else:
            self.decoder = Decoder(encoder_channels, decode_channels, dropout, window_size, num_classes)
            self.prompt_generate = None

    def forward(self, x, y, mode='Train',
                point_coords=None, point_labels=None,
                boxes=None, masks=None):
        
        h, w = x.size()[-2:]
        y = torch.unsqueeze(y, dim=1).repeat(1,3,1,1)
        encoder_outputs = self.image_encoder(x, y)

        if len(encoder_outputs) == 3:
            deepx, deepy, intermediates = encoder_outputs
            int_x0, int_y0, H_out, W_out = intermediates[0]
            int_x1, int_y1, _, _ = intermediates[1]
            int_x2, int_y2, _, _ = intermediates[2]

            int_x0 = int_x0.permute(0, 3, 1, 2)
            int_x1 = int_x1.permute(0, 3, 1, 2)
            int_x2 = int_x2.permute(0, 3, 1, 2)
            int_y0 = int_y0.permute(0, 3, 1, 2)
            int_y1 = int_y1.permute(0, 3, 1, 2)
            int_y2 = int_y2.permute(0, 3, 1, 2)

            int_x0 = self.inter_proj0(int_x0)
            int_x1 = self.inter_proj1(int_x1)
            int_x2 = self.inter_proj2(int_x2)
            int_x = (int_x0 + int_x1 + int_x2) / 3
            int_x = self.inter_neck(int_x)

            int_y0 = self.inter_proj0(int_y0)
            int_y1 = self.inter_proj1(int_y1)
            int_y2 = self.inter_proj2(int_y2)
            int_y = (int_y0 + int_y1 + int_y2) / 3
            int_y = self.inter_neck(int_y)
        else:
            deepx, deepy = encoder_outputs
            int_x = int_y = None

        deepx = self.lateral_conv(deepx)
        deepy = self.lateral_conv(deepy)

        if int_x is not None:
            deepx = deepx + int_x
            deepy = deepy + int_y

        res1x = self.fpn1x(deepx)
        res2x = self.fpn2x(deepx)
        res3x = self.fpn3x(deepx)
        res4x = self.fpn4x(deepx)
        res1y = self.fpn1y(deepy)
        res2y = self.fpn2y(deepy)
        res3y = self.fpn3y(deepy)
        res4y = self.fpn4y(deepy)
        res1 = self.fusion1(res1x, res1y)
        res2 = self.fusion2(res2x, res2y)
        res3 = self.fusion3(res3x, res3y)
        res4 = self.fusion4(res4x, res4y)
        
        sparse_emb = None
        dense_emb = None
        if point_coords is not None or boxes is not None or masks is not None:
            sparse_emb, dense_emb = self.prompt_encoder(
                points=(point_coords, point_labels) if point_coords is not None else None,
                boxes=boxes,
                masks=masks
            )
        elif self.prompt_generate is not None:
            h_now, w_now = x.size()[-2:]

            point_coords_gen, point_labels_gen, masks_gen = self.prompt_generate(x, y)

            if point_coords_gen is not None or masks_gen is not None:
                point_coords_mapped = point_coords_gen * torch.tensor([1024, 1024], device=x.device)
                sparse_emb, dense_emb = self.prompt_encoder(
                    points=(point_coords_mapped, point_labels_gen) if point_coords_gen is not None else None,
                    boxes=None,
                    masks=masks_gen
                )
        
        x = self.decoder(res1, res2, res3, res4, h, w,
                        sparse_prompt_emb=sparse_emb,
                        dense_prompt_emb=dense_emb)
        
        return x
