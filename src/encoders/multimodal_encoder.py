import torch
import torch.nn as nn
from typing import Dict
from .base import BaseEncoder

class MultimodalEncoder(BaseEncoder):
    """
    Encodeur multimodal unifié avec :
    - Fusion cross-attention
    - Dynamic Gating
    - Memory-augmented processing
    """
    
    def __init__(self, 
                 modality_dims: Dict[str, int],
                 hidden_dim=768,
                 num_experts=4,
                 **kwargs):
        super().__init__(input_dim=sum(modality_dims.values()), output_dim=hidden_dim)
        
        # Projections par modalité
        self.projections = nn.ModuleDict({
            mod: nn.Linear(dim, hidden_dim)
            for mod, dim in modality_dims.items()
        })
        
        # Experts de fusion
        self.fusion_experts = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_dim, 8, dim_feedforward=3072)
            for _ in range(num_experts)
        ])
        
        # Mémoire persistante
        self.memory = nn.Parameter(torch.randn(1024, hidden_dim))
        
        # Dynamic Gating
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, inputs: Dict[str, torch.Tensor]):
        # Projection des modalités
        projected = [proj(inputs[mod]) for mod, proj in self.projections.items()]
        
        # Concaténation
        x = torch.stack(projected, dim=1)
        
        # Ajout mémoire
        mem = self.memory.unsqueeze(0).expand(x.size(0), -1, -1)
        x = torch.cat([x, mem], dim=1)
        
        # Calcul des gates
        gates = self.gate(x.mean(1))
        
        # Application des experts
        expert_outputs = []
        for i, expert in enumerate(self.fusion_experts):
            expert_outputs.append(expert(x) * gates[:, i].unsqueeze(-1))
        
        # Combinaison
        x = sum(expert_outputs)
        
        return {
            'output': x[:, 0],  # Retourne le token [CLS]
            'modality_embeddings': x
        }
