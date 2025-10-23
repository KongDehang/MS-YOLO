import torch
from torch import nn, Tensor, LongTensor
from torch.nn import init
import torch.nn.functional as F
import torchvision

import math
import numpy as np

__all__ = ['BandAttention', 'BandAttentionV2', 'BandAttentionV3', 'SpectralSpatialAttention', 'AdaptiveBandAttention']


class BandAttention(nn.Module):
    """
    Band Attention module for multispectral/hyperspectral image processing.
    
    Applies channel-wise attention to emphasize important spectral bands,
    similar to SENet but optimized for spectral data.
    
    Args:
        in_channels (int): Number of input channels (spectral bands).
        reduction (int): Channel reduction ratio for bottleneck. Default: 4
        
    Example:
        >>> attn = BandAttention(in_channels=64)
        >>> x = torch.randn(2, 64, 80, 80)
        >>> out = attn(x)  # Shape: (2, 64, 80, 80)
    """
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Use max(1, ...) to avoid division by zero for small channels
        hidden_channels = max(1, in_channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Apply band attention to input feature map.
        
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            Tensor: Attention-weighted output of shape (B, C, H, W)
        """
        b, c, h, w = x.shape
        # Squeeze: Global Average Pooling
        y = self.avg_pool(x).view(b, c)
        # Excitation: Learn channel-wise weights
        y = self.fc(y).view(b, c, 1, 1)
        # Scale: Apply attention weights
        return x * y.expand_as(x)


class BandAttentionV2(nn.Module):
    """
    Enhanced Band Attention with both average and max pooling.
    
    Combines both average-pooled and max-pooled features for more robust
    channel attention, similar to CBAM channel attention.
    
    Args:
        in_channels (int): Number of input channels.
        reduction (int): Channel reduction ratio. Default: 4
        
    Example:
        >>> attn = BandAttentionV2(in_channels=64)
        >>> x = torch.randn(2, 64, 80, 80)
        >>> out = attn(x)
    """
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden_channels = max(1, in_channels // reduction)
        
        # Shared MLP for both pooling paths
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Apply enhanced band attention with dual pooling.
        
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            Tensor: Attention-weighted output of shape (B, C, H, W)
        """
        b, c, h, w = x.shape
        
        # Average-pool path
        avg_pool = self.avg_pool(x).view(b, c)
        avg_out = self.fc(avg_pool)
        
        # Max-pool path
        max_pool = self.max_pool(x).view(b, c)
        max_out = self.fc(max_pool)
        
        # Combine and apply sigmoid
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        return x * out.expand_as(x)


class BandAttentionV3(nn.Module):
    """
    🚀 INNOVATION: BandAttention + Lightweight Spatial Attention
    
    Combines dual-pooling channel attention with efficient spatial attention
    for multispectral image processing. Unlike CBAM, uses depthwise separable
    convolution for spatial modeling to reduce parameters.
    
    Key Innovations:
        1. Dual-path channel attention (Avg + Max pooling)
        2. Depthwise separable spatial attention (lighter than CBAM's 7x7 conv)
        3. Parallel fusion instead of sequential (channel & spatial computed together)
    
    Args:
        in_channels (int): Number of input channels (spectral bands).
        reduction (int): Channel reduction ratio. Default: 4
        spatial_kernel (int): Spatial attention kernel size. Default: 5
        
    Example:
        >>> attn = BandAttentionV3(in_channels=64)
        >>> x = torch.randn(2, 64, 80, 80)
        >>> out = attn(x)
    """
    def __init__(self, in_channels, reduction=4, spatial_kernel=5):
        super().__init__()
        
        # === Channel Attention (Same as V2) ===
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden_channels = max(1, in_channels // reduction)
        
        self.channel_fc = nn.Sequential(
            nn.Linear(in_channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels, bias=False)
        )
        
        # === Spatial Attention (Innovation: Depthwise Separable Conv) ===
        # Lighter than CBAM's standard 7x7 conv
        padding = spatial_kernel // 2
        self.spatial_conv = nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(2, 2, kernel_size=spatial_kernel, padding=padding, groups=2, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            # Pointwise convolution
            nn.Conv2d(2, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1)
        )
        
        self.channel_sigmoid = nn.Sigmoid()
        self.spatial_sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Apply channel and spatial attention in parallel.
        
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            Tensor: Attention-weighted output of shape (B, C, H, W)
        """
        b, c, h, w = x.shape
        
        # === Channel Attention Path ===
        avg_pool_c = self.avg_pool(x).view(b, c)
        max_pool_c = self.max_pool(x).view(b, c)
        
        avg_out = self.channel_fc(avg_pool_c)
        max_out = self.channel_fc(max_pool_c)
        
        channel_att = self.channel_sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        # === Spatial Attention Path ===
        # Cross-channel pooling
        avg_pool_s = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_pool_s, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        
        spatial_input = torch.cat([avg_pool_s, max_pool_s], dim=1)  # (B, 2, H, W)
        spatial_att = self.spatial_sigmoid(self.spatial_conv(spatial_input))  # (B, 1, H, W)
        
        # === Fusion: Apply both attentions ===
        out = x * channel_att * spatial_att
        
        return out


class SpectralSpatialAttention(nn.Module):
    """
    🚀 INNOVATION: Spectral-Spatial Collaborative Attention
    
    Designed specifically for multispectral/hyperspectral imaging.
    Unlike CBAM which treats all channels equally, this module groups
    spectral bands and applies group-wise spatial attention.
    
    Key Innovations:
        1. Spectral band grouping (neighboring bands often correlated)
        2. Group-wise spatial attention (each group gets its own spatial weights)
        3. Learnable fusion of group attentions
    
    Args:
        in_channels (int): Number of input channels (must be divisible by num_groups).
        num_groups (int): Number of spectral band groups. Default: 4
        reduction (int): Channel reduction ratio. Default: 4
        
    Example:
        >>> attn = SpectralSpatialAttention(in_channels=64, num_groups=4)
        >>> x = torch.randn(2, 64, 80, 80)
        >>> out = attn(x)
    """
    def __init__(self, in_channels, num_groups=4, reduction=4):
        super().__init__()
        assert in_channels % num_groups == 0, "in_channels must be divisible by num_groups"
        
        self.num_groups = num_groups
        self.channels_per_group = in_channels // num_groups
        
        # Global channel attention
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        hidden_channels = max(1, in_channels // reduction)
        self.global_fc = nn.Sequential(
            nn.Linear(in_channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels, bias=False),
            nn.Sigmoid()
        )
        
        # Group-wise spatial attention
        self.group_spatial = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
                nn.Sigmoid()
            ) for _ in range(num_groups)
        ])
        
        # Learnable fusion weights for groups
        self.fusion_weights = nn.Parameter(torch.ones(num_groups) / num_groups)
    
    def forward(self, x):
        """
        Apply spectral-spatial collaborative attention.
        
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            Tensor: Attention-weighted output of shape (B, C, H, W)
        """
        b, c, h, w = x.shape
        
        # === Global Channel Attention ===
        global_att = self.global_pool(x).view(b, c)
        global_att = self.global_fc(global_att).view(b, c, 1, 1)
        x_channel = x * global_att
        
        # === Group-wise Spatial Attention ===
        x_groups = torch.chunk(x_channel, self.num_groups, dim=1)
        out_groups = []
        
        for i, x_group in enumerate(x_groups):
            # Spatial pooling for this group
            avg_pool = torch.mean(x_group, dim=1, keepdim=True)
            max_pool, _ = torch.max(x_group, dim=1, keepdim=True)
            spatial_input = torch.cat([avg_pool, max_pool], dim=1)
            
            # Group-specific spatial attention
            spatial_att = self.group_spatial[i](spatial_input)
            out_groups.append(x_group * spatial_att * self.fusion_weights[i])
        
        # Concatenate groups
        out = torch.cat(out_groups, dim=1)
        
        return out


class AdaptiveBandAttention(nn.Module):
    """
    🚀 INNOVATION: Adaptive Band Attention with Dynamic Reduction
    
    Automatically adjusts the reduction ratio based on input feature statistics.
    Useful when different layers/stages have different channel importance patterns.
    
    Key Innovations:
        1. Dynamic reduction ratio (adapts to feature complexity)
        2. Multi-scale channel aggregation (uses multiple pooling scales)
        3. Soft gating instead of hard Sigmoid (smoother gradient flow)
    
    Args:
        in_channels (int): Number of input channels.
        base_reduction (int): Base reduction ratio. Default: 4
        adaptive (bool): Enable adaptive reduction. Default: True
        
    Example:
        >>> attn = AdaptiveBandAttention(in_channels=64)
        >>> x = torch.randn(2, 64, 80, 80)
        >>> out = attn(x)
    """
    def __init__(self, in_channels, base_reduction=4, adaptive=True):
        super().__init__()
        self.adaptive = adaptive
        self.base_reduction = base_reduction
        
        # Multi-scale pooling
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveAvgPool2d(2),
            nn.AdaptiveMaxPool2d(1)
        ])
        
        # Base MLP (will be dynamically adjusted if adaptive=True)
        hidden_channels = max(1, in_channels // base_reduction)
        self.fc1 = nn.Linear(in_channels * 3, hidden_channels, bias=False)  # *3 for multi-scale
        self.fc2 = nn.Linear(hidden_channels, in_channels, bias=False)
        
        # Adaptive reduction controller (learns optimal reduction)
        if adaptive:
            self.reduction_controller = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_channels, 1),
                nn.Sigmoid()
            )
        
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()  # Soft gating: output in [-1, 1], then shift to [0, 2]
    
    def forward(self, x):
        """
        Apply adaptive band attention with dynamic reduction.
        
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            Tensor: Attention-weighted output of shape (B, C, H, W)
        """
        b, c, h, w = x.shape
        
        # === Multi-scale Feature Aggregation ===
        features = []
        
        # Scale 1: Global pooling (1x1)
        feat1 = self.pools[0](x).view(b, c)
        features.append(feat1)
        
        # Scale 2: Spatial pooling (2x2) then global
        feat2 = self.pools[1](x).view(b, c, -1).mean(dim=2)
        features.append(feat2)
        
        # Scale 3: Max pooling
        feat3 = self.pools[2](x).view(b, c)
        features.append(feat3)
        
        # Concatenate multi-scale features
        multi_scale = torch.cat(features, dim=1)  # (B, C*3)
        
        # === Adaptive Reduction (Optional) ===
        if self.adaptive:
            # Learn a scaling factor for the bottleneck
            reduction_scale = self.reduction_controller(x)  # (B, 1)
            # Apply to hidden layer (dynamic capacity)
            hidden = self.fc1(multi_scale)
            hidden = hidden * (reduction_scale + 0.5)  # Scale in [0.5, 1.5]
            hidden = self.relu(hidden)
        else:
            hidden = self.relu(self.fc1(multi_scale))
        
        # === Attention Generation ===
        att = self.fc2(hidden)
        
        # Soft gating: Tanh output [-1, 1] -> shift to [0, 2] for soft scaling
        att = self.tanh(att) + 1.0  # Now in [0, 2]
        att = att.view(b, c, 1, 1)
        
        # === Apply Attention ===
        out = x * att
        
        return out
