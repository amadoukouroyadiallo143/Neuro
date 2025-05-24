import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union, Tuple
from flash_attn.ops.fused_dense import FusedDense
from flash_attn.modules.mha import FlashCrossAttention
from deepseek_v2 import DeepseekMoE

from .base import BaseDecoder

class MultimodalDecoder(BaseDecoder):
    """
    Décodeur multimodal capable de gérer plusieurs modalités de sortie.
    
    Ce décodeur combine plusieurs sous-décodeurs spécialisés (texte, image, etc.)
    et les coordonne pour produire des sorties multimodales.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        expert_dim: int = 2048,
        num_experts: int = 8,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        **kwargs
    ):
        """
        Initialise le décodeur multimodal.
        
        Args:
            input_dim: Dimension des entrées du décodeur
            expert_dim: Dimension des experts
            num_experts: Nombre d'experts
            top_k: Nombre d'experts à sélectionner
            capacity_factor: Facteur de capacité pour le routage
        """
        # La dimension de sortie est déterminée par la dimension d'entrée
        output_dim = input_dim
        super().__init__(input_dim=input_dim, output_dim=output_dim, **kwargs)
        
        # Routing avec Load Balancing
        self.router = nn.Linear(input_dim, num_experts, bias=False)
        
        # Experts parallèles (MoE)
        self.experts = DeepseekMoE(
            input_dim,
            expert_dim,
            num_experts=num_experts,
            top_k=top_k,
            capacity_factor=capacity_factor,
            fused=True
        )
        
        # Projection finale avec Flash Attention
        self.proj = FusedDense(input_dim, input_dim)
        
        # Kernel optimisé pour FlashDecoding
        self.register_buffer('decoding_cache', torch.zeros(1, 1, input_dim))
        
    def forward(
        self, 
        inputs: torch.Tensor, 
        encoder_outputs: Optional[Dict[str, Any]] = None,
        decoder_inputs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Passe avant du décodeur multimodal.
        
        Args:
            inputs: Tenseur d'entrée de forme (batch_size, input_dim) ou (batch_size, seq_len, input_dim)
            encoder_outputs: Sorties de l'encodeur (optionnel)
            decoder_inputs: Entrées spécifiques pour chaque décodeur (optionnel)
            
        Returns:
            Dictionnaire contenant les sorties de chaque décodeur et la sortie fusionnée
        """
        batch_size = inputs.size(0)
        
        # Dictionnaire pour stocker les sorties de chaque décodeur
        outputs = {}
        
        # Routing intelligent
        logits = self.router(inputs)
        weights = torch.softmax(logits, dim=-1)
        
        # Exécution parallèle des experts
        expert_outputs = self.experts(inputs)
        
        # Combinaison dynamique
        x = torch.einsum('bne,bned->bd', weights, expert_outputs)
        
        # Fusion contextuelle
        if encoder_outputs:
            x = FlashCrossAttention(x, encoder_outputs)
        
        # Projection finale
        outputs['fused_output'] = self.proj(x)
        
        return outputs
    
    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration du décodeur multimodal."""
        config = super().get_config()
        config.update({
            'expert_dim': self.experts.expert_dim,
            'num_experts': self.experts.num_experts,
            'top_k': self.experts.top_k,
            'capacity_factor': self.experts.capacity_factor,
        })
        
        return config
