import torch
import torch.nn as nn
from einops import rearrange
from timm.models.efficientvit import EfficientViT

class VideoEncoder(BaseEncoder):
    """
    Encodeur vidéo 3D avec EfficientViT optimisé
    Features:
    - Temporal Patch Embedding
    - 3D Hyena Operators
    - Spatio-Temporal Pooling
    """
    
    def __init__(self, 
                 in_channels=3,
                 dim=768,
                 depth=12,
                 num_heads=8,
                 kernel_size=(3,3,3),
                 temporal_stride=2,
                 **kwargs):
        super().__init__(input_dim=in_channels, output_dim=dim)
        
        # Configuration EfficientViT 3D
        self.backbone = EfficientViT(
            in_chans=in_channels,
            depths=[depth//3, depth//3, depth//3],
            dim=dim,
            num_heads=num_heads,
            window_size=(8, 8, 8),
            mlp_ratio=4.0,
            qkv_bias=True,
            use_3d=True
        )
        
        # Temporal processing
        self.temporal_hyena = nn.Sequential(*[
            HyenaOperator3D(
                dim=dim,
                order=2,
                kernel_size=kernel_size,
                temporal_stride=temporal_stride
            ) for _ in range(3)
        ])
        
        # Pooling spatio-temporel
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.backbone(x)
        x = self.temporal_hyena(x)
        x = self.pool(x)
        return x.squeeze()

class HyenaOperator3D(nn.Module):
    """Opérateur Hyena 3D pour traitement spatio-temporel efficace"""
    def __init__(self, dim, order=2, kernel_size=(3,3,3), temporal_stride=2):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim, kernel_size, stride=(temporal_stride,1,1), padding=(1,1,1), groups=dim)
        self.gate = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim)
        )
        
    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.gate(x)
        x = x.permute(0, 4, 1, 2, 3)
        return x + residual
