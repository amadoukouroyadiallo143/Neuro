import torch
import torch.nn as nn
from einops import rearrange
from flash_attn.modules.mha import FlashCrossAttention
from hyena.hyena import HyenaOperator
from s3pool import S3Pool
from typing import Dict, Any, Optional

from .base import BaseEncoder

class TextEncoder(BaseEncoder):
    """
    Encodeur pour les données textuelles.
    
    Cet encodeur utilise une architecture Transformer pour encoder du texte.
    """
    
    def __init__(
        self,
        vocab_size: int,
        dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ff_mult: int = 4,
        hyena_order: int = 2,
        window_size: int = 128,
        group_size: int = 64,
        use_flash: bool = True,
        **kwargs
    ):
        """
        Initialise l'encodeur de texte.
        
        Args:
            vocab_size: Taille du vocabulaire
            dim: Dimension des couches cachées
            depth: Nombre de couches du modèle
            num_heads: Nombre de têtes d'attention
            ff_mult: Multiplicateur pour la dimension des couches feed-forward
            hyena_order: Ordre de l'opérateur Hyena
            window_size: Taille de la fenêtre pour l'opérateur Hyena
            group_size: Taille du groupe pour le pooling S3
            use_flash: Utilisation de l'attention Flash
        """
        super().__init__(input_dim=dim, output_dim=dim, **kwargs)
        
        self.emb = nn.Embedding(vocab_size, dim)
        self.s3_pool = S3Pool(dim, pool_size=3, group_size=group_size)
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                HyenaOperator(
                    dim=dim,
                    order=hyena_order,
                    filter_order=window_size,
                    num_in_blocks=1,
                    num_out_blocks=1,
                    drop_prob=0.1
                ) if i % 2 == 0 else 
                FlashCrossAttention(
                    embed_dim=dim,
                    num_heads=num_heads,
                    causal=False
                ),
                nn.LayerNorm(dim),
                nn.GELU(),
                S3Pool(dim, pool_size=2, group_size=group_size)
            ) for i in range(depth)
        ])
        
        # Quantization dynamique
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]:
        """
        Passe avant de l'encodeur de texte.
        
        Args:
            input_ids: Indices des tokens d'entrée (batch_size, seq_len)
            attention_mask: Masque d'attention (batch_size, seq_len)
            
        Returns:
            Dictionnaire contenant la sortie encodée et d'autres informations utiles
        """
        x = self.quant(self.emb(input_ids))
        for layer in self.layers:
            x = layer(x)
        return {
            'output': self.dequant(x[:, 0]),
            'attention_mask': attention_mask,
            'embeddings': x
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration de l'encodeur de texte."""
        config = super().get_config()
        config.update({
            'vocab_size': self.emb.num_embeddings,
            'dim': self.emb.embedding_dim,
            'depth': len(self.layers),
            'num_heads': self.layers[0][1].num_heads,
            'ff_mult': 4,
            'hyena_order': self.layers[0][0].order,
            'window_size': self.layers[0][0].filter_order,
            'group_size': self.s3_pool.group_size,
            'use_flash': True,
        })
        return config
