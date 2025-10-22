from torch import nn

__all__ = ["BandAttention", "BandAttentionV2"]


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
            nn.Sigmoid(),
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
            nn.Linear(hidden_channels, in_channels, bias=False),
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
