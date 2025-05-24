import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub
from mobile_vit import MobileViTv3
from coremltools.converters.mil import register_torch_op
from coremltools.models.neural_engine import NeuralEngine
from typing import Dict, Any, Optional, Tuple, Union, List

from .base import BaseEncoder

class ImageEncoder(BaseEncoder):
    """
    Encodeur pour les données images utilisant une architecture CNN.
    
    Cette implémentation utilise une architecture de type ResNet simplifiée
    pour extraire des caractéristiques d'images.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 512,
        depth: int = 16,
        num_heads: int = 8,
        kernel_size: Tuple[int, int] = (3, 3),
        use_neural_engine: bool = True,
        **kwargs
    ):
        """
        Initialise l'encodeur d'images.
        
        Args:
            in_channels: Nombre de canaux en entrée (3 pour RGB, 1 pour niveaux de gris)
            dim: Dimension de sortie de l'encodeur
            depth: Profondeur du modèle
            num_heads: Nombre de têtes pour l'attention
            kernel_size: Taille du noyau pour les convolutions
            use_neural_engine: Si True, utilise l'Apple Neural Engine pour l'accélération
        """
        super().__init__(input_dim=in_channels, output_dim=dim, **kwargs)
        
        # Configuration MobileViTv3
        self.backbone = MobileViTv3(
            in_channels=in_channels,
            dims=[dim//4, dim//2, dim],
            depths=[depth//4, depth//2, depth],
            num_heads=num_heads,
            expand_ratio=4,
            kernel_sizes=[3,5,7],
            use_activation_checkpointing=True
        )
        
        # Optimisations Apple Neural Engine
        if use_neural_engine:
            self.neural_engine = NeuralEngine(
                input_shape=(in_channels, 256, 256),
                compute_units=NeuralEngine.ComputeUnit.ALL,
                optimize_for='performance'
            )
        
        # Quantization INT8
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        # Dynamic token merging
        self.token_merging = nn.Sequential(
            nn.Conv2d(dim, dim//8, 1),
            nn.GELU(),
            nn.Conv2d(dim//8, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """
        Passe avant de l'encodeur d'images.
        
        Args:
            x: Tenseur d'entrée de forme (batch_size, channels, height, width)
            
        Returns:
            Dictionnaire contenant la sortie encodée et d'autres informations utiles
        """
        x = self.quant(x)
        
        # Neural Engine acceleration
        if hasattr(self, 'neural_engine'):
            x = self.neural_engine(x)
        
        # MobileViTv3 backbone
        features = self.backbone(x)
        
        # Dynamic token pruning
        mask = self.token_merging(features)
        features = features * mask
        
        output = self.dequant(features.mean([-2,-1]))
        
        return {
            'output': output,
            'features': [features],  # Features intermédiaires pour d'éventuelles pertes multi-échelles
            'spatial_features': features  # Avant le pooling global
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration de l'encodeur d'images."""
        config = super().get_config()
        config.update({
            'dim': self.backbone.dims[-1],
            'depth': self.backbone.depths[-1],
            'num_heads': self.backbone.num_heads,
            'use_neural_engine': hasattr(self, 'neural_engine'),
        })
        return config
