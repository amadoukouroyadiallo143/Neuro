import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union

from .base import BaseEncoder

class AudioEncoder(BaseEncoder):
    """
    Encodeur pour les données audio.
    
    Cet encodeur utilise une architecture inspirée de Wav2Vec 2.0 pour extraire
    des caractéristiques à partir de signaux audio bruts.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        conv_layers: int = 7,
        conv_channels: int = 512,
        kernel_sizes: Optional[list] = None,
        strides: Optional[list] = None,
        output_dim: int = 768,
        **kwargs
    ):
        """
        Initialise l'encodeur audio.
        
        Args:
            in_channels: Nombre de canaux d'entrée (1 pour mono, 2 pour stéréo)
            conv_layers: Nombre de couches convolutives
            conv_channels: Nombre de canaux pour les couches convolutives
            kernel_sizes: Liste des tailles de noyaux pour chaque couche
            strides: Liste des pas pour chaque couche
            output_dim: Dimension de sortie de l'encodeur
        """
        super().__init__(input_dim=in_channels, output_dim=output_dim, **kwargs)
        
        self.in_channels = in_channels
        self.conv_layers = conv_layers
        self.conv_channels = conv_channels
        
        # Configuration par défaut des noyaux et des pas
        if kernel_sizes is None:
            kernel_sizes = [10] + [3] * (conv_layers - 1)  # Premier noyau plus grand
        if strides is None:
            strides = [5] + [2] * (conv_layers - 1)  # Premier pas plus grand
            
        assert len(kernel_sizes) == conv_layers, "kernel_sizes doit avoir une taille égale à conv_layers"
        assert len(strides) == conv_layers, "strides doit avoir une taille égale à conv_layers"
        
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        
        # Couches convolutives
        self.conv_layers = nn.ModuleList()
        
        in_dim = in_channels
        for i in range(conv_layers):
            padding = kernel_sizes[i] // 2  # Padding pour maintenir la dimension temporelle
            
            conv = nn.Sequential(
                nn.Conv1d(
                    in_dim,
                    conv_channels,
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=padding,
                    bias=False
                ),
                nn.Dropout(0.1),
                nn.GELU(),
                nn.LayerNorm(conv_channels),
            )
            
            self.conv_layers.append(conv)
            in_dim = conv_channels
        
        # Projection vers l'espace de sortie
        self.projection = nn.Linear(conv_channels, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]:
        """
        Passe avant de l'encodeur audio.
        
        Args:
            x: Tenseur d'entrée de forme (batch_size, channels, seq_len)
            lengths: Longueurs réelles des séquences (pour le masquage)
            
        Returns:
            Dictionnaire contenant la sortie encodée et d'autres informations utiles
        """
        # x: (batch_size, channels, seq_len)
        batch_size, _, seq_len = x.size()
        
        # Passage à travers les couches convolutives
        features = []
        conv_input = x
        
        for i, conv in enumerate(self.conv_layers):
            conv_output = conv(conv_input)
            features.append(conv_output)
            conv_input = conv_output
        
        # Dernière couche de features
        last_features = features[-1]  # (batch_size, conv_channels, seq_len')
        
        # Permuter pour la projection linéaire: (batch_size, seq_len', conv_channels)
        last_features = last_features.permute(0, 2, 1)
        
        # Projection vers l'espace de sortie
        output = self.projection(last_features)  # (batch_size, seq_len', output_dim)
        output = self.layer_norm(output)
        
        # Calculer les longueurs de séquence après chaque couche
        if lengths is not None:
            for stride in self.strides:
                lengths = ((lengths.float() - 1) / stride + 1).floor().long()
        
        return {
            'output': output,  # (batch_size, seq_len', output_dim)
            'features': features,  # Toutes les représentations intermédiaires
            'lengths': lengths,  # Longueurs mises à jour
            'attention_mask': self._create_attention_mask(batch_size, output.size(1), lengths) if lengths is not None else None
        }
    
    def _create_attention_mask(self, batch_size: int, seq_len: int, lengths: torch.Tensor) -> torch.Tensor:
        """Crée un masque d'attention basé sur les longueurs de séquence."""
        # Créer un masque de forme (batch_size, seq_len)
        # 1 pour les tokens valides, 0 pour le padding
        mask = torch.arange(seq_len, device=lengths.device).expand(batch_size, seq_len) < lengths.unsqueeze(1)
        return mask.float()
    
    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration de l'encodeur audio."""
        config = super().get_config()
        config.update({
            'in_channels': self.in_channels,
            'conv_layers': len(self.conv_layers),
            'conv_channels': self.conv_channels,
            'kernel_sizes': self.kernel_sizes,
            'strides': self.strides,
        })
        return config
