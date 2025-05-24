"""
Module principal pour les décodeurs du projet Neuro.

Ce package contient les implémentations des différents décodeurs pour la génération
de sorties multi-modales à partir des représentations internes du modèle.
"""

from .base import BaseDecoder
from .text_decoder import TextDecoder
from .image_decoder import ImageDecoder
from .multimodal_decoder import MultimodalDecoder

__all__ = [
    'BaseDecoder',
    'TextDecoder',
    'ImageDecoder',
    'MultimodalDecoder',
]
