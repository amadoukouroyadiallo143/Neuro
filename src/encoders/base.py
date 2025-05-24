from typing import Dict, Any, Optional, Union, List
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseEncoder(nn.Module, ABC):
    """
    Classe de base abstraite pour tous les encodeurs
    
    Nouveautés :
    - Support multi-input/multi-modal
    - Méthode de configuration avancée
    - Hooks pour l'observabilité
    """
    
    def __init__(self, 
                 input_dim: Union[int, Dict[str, int]],
                 output_dim: int,
                 supports_multi_input: bool = False,
                 **kwargs):
        super().__init__()
        
        if isinstance(input_dim, dict):
            self.input_dims = input_dim
            self.supports_multi_input = True
        else:
            self.input_dim = input_dim
            self.supports_multi_input = supports_multi_input
            
        self.output_dim = output_dim
        
        # Hooks d'observabilité
        self.register_buffer('latent_stats', torch.zeros(3, output_dim))
        
    @abstractmethod
    def forward(self, 
                x: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                **kwargs) -> Dict[str, Any]:
        """
        Forward pass de l'encodeur
        
        Args:
            x: Input tensor ou dict de tensors pour multi-modal
            **kwargs: Arguments spécifiques à l'implémentation
        
        Returns:
            Dict contenant :
            - 'output': Tensor de sortie principal
            - 'aux_outputs': Sorties auxiliaires
            - 'attention': Masques d'attention (optionnel)
        """
        pass
    
    def update_latent_stats(self, x):
        """Track mean/std des activations pour l'analyse"""
        self.latent_stats[0] += x.mean(dim=0).detach()
        self.latent_stats[1] += x.std(dim=0).detach()
        self.latent_stats[2] += 1
    
    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration complète"""
        config = {
            'encoder_type': self.__class__.__name__,
            'input_config': self.input_dims if self.supports_multi_input else self.input_dim,
            'output_dim': self.output_dim,
            'supports_multi_input': self.supports_multi_input
        }
        return config
    
    def reset_stats(self):
        """Réinitialise les statistiques de tracking"""
        self.latent_stats.zero_()
