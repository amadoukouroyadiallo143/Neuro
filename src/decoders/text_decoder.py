import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

from .base import BaseDecoder

class TextDecoder(BaseDecoder):
    """
    Décodeur pour la génération de texte.
    
    Ce décodeur utilise une architecture de type Transformer pour générer du texte
    à partir des représentations internes du modèle.
    """
    
    def __init__(
        self,
        vocab_size: int,
        max_length: int = 512,
        hidden_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 6,
        dropout: float = 0.1,
        **kwargs
    ):
        """
        Initialise le décodeur de texte.
        
        Args:
            vocab_size: Taille du vocabulaire de sortie
            max_length: Longueur maximale des séquences générées
            hidden_dim: Dimension des couches cachées
            num_heads: Nombre de têtes d'attention
            num_layers: Nombre de couches du décodeur
            dropout: Taux de dropout
        """
        super().__init__(input_dim=hidden_dim, output_dim=vocab_size, **kwargs)
        
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        # Couche d'embedding pour les tokens de sortie
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        
        # Couches du décodeur Transformer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Couche de sortie
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout et normalisation
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Initialisation des poids
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialise les poids du modèle."""
        init_range = 0.1
        self.token_embedding.weight.data.uniform_(-init_range, init_range)
        self.position_embedding.weight.data.uniform_(-init_range, init_range)
        self.output_projection.bias.data.zero_()
        self.output_projection.weight.data.uniform_(-init_range, init_range)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        encoder_outputs: Optional[Dict[str, Any]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Passe avant du décodeur de texte.
        
        Args:
            input_ids: Indices des tokens d'entrée (batch_size, tgt_seq_len)
            encoder_outputs: Sorties de l'encodeur (doit contenir 'output')
            attention_mask: Masque d'attention pour le décodeur (batch_size, tgt_seq_len)
            encoder_attention_mask: Masque d'attention pour l'encodeur (batch_size, src_seq_len)
            
        Returns:
            Dictionnaire contenant les logits de sortie et d'autres informations utiles
        """
        batch_size, tgt_seq_len = input_ids.size()
        
        # Créer des embeddings pour les tokens d'entrée
        token_embeddings = self.token_embedding(input_ids)  # (batch_size, tgt_seq_len, hidden_dim)
        
        # Ajouter les embeddings de position
        positions = torch.arange(tgt_seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embedding(positions)
        
        # Somme des embeddings
        x = token_embeddings + position_embeddings
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        # Créer le masque causal pour le décodeur
        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len).to(x.device)
        
        # Vérifier si on a des sorties d'encodeur
        memory = None
        if encoder_outputs is not None and 'output' in encoder_outputs:
            memory = encoder_outputs['output']
        
        # Passage à travers le décodeur
        if memory is not None:
            # Décodage avec attention croisée sur les sorties de l'encodeur
            decoder_output = self.transformer_decoder(
                tgt=x,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=(1 - attention_mask).bool() if attention_mask is not None else None,
                memory_key_padding_mask=(1 - encoder_attention_mask).bool() if encoder_attention_mask is not None else None
            )
        else:
            # Décodage sans attention croisée (génération autonome)
            decoder_output = self.transformer_decoder(
                tgt=x,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=(1 - attention_mask).bool() if attention_mask is not None else None
            )
        
        # Projection vers l'espace de sortie
        logits = self.output_projection(decoder_output)  # (batch_size, tgt_seq_len, vocab_size)
        
        return {
            'logits': logits,
            'hidden_states': decoder_output,
            'attention_weights': None  # Peut être rempli avec les poids d'attention si nécessaire
        }
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Génère un masque carré pour le décalage temporel."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Dict[str, Any]] = None,
        max_length: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Génère du texte de manière auto-régressive.
        
        Args:
            input_ids: Indices de départ (batch_size, start_seq_len)
            encoder_outputs: Sorties de l'encodeur
            max_length: Longueur maximale de la séquence générée
            **kwargs: Arguments additionnels pour la génération
            
        Returns:
            Dictionnaire contenant les IDs générés et d'autres informations
        """
        raise NotImplementedError("La génération auto-régressive n'est pas encore implémentée.")
    
    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration du décodeur de texte."""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'dropout': self.dropout_rate,
        })
        return config
