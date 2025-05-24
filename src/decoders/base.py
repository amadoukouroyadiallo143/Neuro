from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

class BaseDecoder(nn.Module, ABC):
    """
    Classe de base pour tous les décodeurs du projet Neuro.
    
    Cette classe abstraite définit l'interface commune que tous les décodeurs
    doivent implémenter.
    """
    
    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        """
        Initialise le décodeur de base.
        
        Args:
            input_dim: Dimension des entrées du décodeur
            output_dim: Dimension de sortie souhaitée
            **kwargs: Arguments additionnels spécifiques à l'implémentation
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    @abstractmethod
    def forward(
        self, 
        inputs: torch.Tensor, 
        encoder_outputs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Passe avant du décodeur.
        
        Args:
            inputs: Tenseur d'entrée de forme (batch_size, seq_len, input_dim)
            encoder_outputs: Sorties de l'encodeur (optionnel)
            **kwargs: Arguments additionnels spécifiques à l'implémentation
            
        Returns:
            Dictionnaire contenant au moins la sortie décodée sous la clé 'output'
            et potentiellement d'autres informations utiles
        """
        pass
    
    def get_output_dim(self) -> int:
        """Retourne la dimension de sortie du décodeur."""
        return self.output_dim
    
    def get_config(self) -> Dict[str, Any]:
        """
        Retourne la configuration du décodeur sous forme de dictionnaire.
        
        Cette méthode peut être surchargée par les classes filles pour inclure
        des paramètres spécifiques.
        """
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'decoder_type': self.__class__.__name__
        }
