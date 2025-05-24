import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union

from .base import BaseDecoder

class ImageDecoder(BaseDecoder):
    """
    Décodeur pour la génération d'images.
    
    Ce décodeur utilise une architecture de type CNN avec des couches de déconvolution
    pour générer des images à partir des représentations internes du modèle.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_channels: int = 3,
        base_channels: int = 512,
        num_layers: int = 4,
        output_size: Tuple[int, int] = (256, 256),
        **kwargs
    ):
        """
        Initialise le décodeur d'images.
        
        Args:
            input_dim: Dimension des entrées du décodeur
            output_channels: Nombre de canaux de sortie (3 pour RGB, 1 pour niveaux de gris)
            base_channels: Nombre de canaux de base pour les couches déconvolutives
            num_layers: Nombre de couches de déconvolution
            output_size: Taille de sortie de l'image (hauteur, largeur)
        """
        super().__init__(input_dim=input_dim, output_dim=output_channels, **kwargs)
        
        self.output_channels = output_channels
        self.base_channels = base_channels
        self.num_layers = max(2, num_layers)  # Au moins 2 couches
        self.output_size = output_size
        
        # Calcul des dimensions intermédiaires
        # On part d'une petite grille (ex: 8x8) et on upscale progressivement
        self.initial_size = (output_size[0] // (2 ** (num_layers - 1)), 
                           output_size[1] // (2 ** (num_layers - 1)))
        
        # Couche initiale pour projeter l'entrée dans l'espace des features
        self.projection = nn.Sequential(
            nn.Linear(input_dim, base_channels * self.initial_size[0] * self.initial_size[1] // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Blocs de déconvolution
        self.deconv_blocks = nn.ModuleList()
        
        in_planes = base_channels // 2
        current_size = (self.initial_size[0] * 2, self.initial_size[1] * 2)
        
        for i in range(num_layers):
            out_planes = max(base_channels // (2 ** (i + 1)), output_channels * 2)
            
            # Dernière couche : nombre de canaux de sortie
            if i == num_layers - 1:
                out_planes = output_channels
            
            block = self._make_deconv_block(
                in_planes,
                out_planes,
                use_batchnorm=(i < num_layers - 1)  # Pas de batch norm sur la dernière couche
            )
            
            self.deconv_blocks.append(block)
            in_planes = out_planes
        
        # Couche de sortie (activation finale)
        self.output_activation = nn.Tanh()  # Pour des images dans [-1, 1]
    
    def _make_deconv_block(
        self, 
        in_planes: int, 
        out_planes: int, 
        use_batchnorm: bool = True
    ) -> nn.Module:
        """Crée un bloc de déconvolution avec upsample."""
        layers = [
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        ]
        
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_planes))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        encoder_outputs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Passe avant du décodeur d'images.
        
        Args:
            inputs: Tenseur d'entrée de forme (batch_size, input_dim)
            encoder_outputs: Sorties de l'encodeur (optionnel)
            
        Returns:
            Dictionnaire contenant l'image générée et d'autres informations utiles
        """
        batch_size = inputs.size(0)
        
        # Projection de l'entrée
        x = self.projection(inputs)
        
        # Remodeler en grille spatiale (batch_size, channels, height, width)
        x = x.view(batch_size, -1, self.initial_size[0] // 2, self.initial_size[1] // 2)
        
        # Passer à travers les blocs de déconvolution
        features = []
        for i, block in enumerate(self.deconv_blocks):
            x = block(x)
            features.append(x)  # Stocker les features intermédiaires
        
        # Appliquer l'activation finale
        output = self.output_activation(x)
        
        return {
            'output': output,  # (batch_size, output_channels, height, width)
            'features': features,  # Features intermédiaires pour d'éventuelles pertes multi-échelles
            'latent_representation': inputs  # Représentation latente d'origine
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration du décodeur d'images."""
        config = super().get_config()
        config.update({
            'output_channels': self.output_channels,
            'base_channels': self.base_channels,
            'num_layers': self.num_layers,
            'output_size': self.output_size,
            'initial_size': self.initial_size,
        })
        return config
