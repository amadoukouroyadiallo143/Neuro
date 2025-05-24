import torch
import torch.nn as nn
from einops import rearrange
from flash_attn.modules.mha import FlashCrossAttention

class VideoDecoder(BaseDecoder):
    """
    Décodeur vidéo avec architecture ConvLSTM optimisée
    Features:
    - Temporal Upsampling Blocks
    - 3D Flash Attention
    - Dynamic Frame Prediction
    """
    
    def __init__(self, input_dim=512, output_frames=16, output_size=(128,128), **kwargs):
        super().__init__(input_dim=input_dim, output_dim=output_frames*3*output_size[0]*output_size[1])
        
        # Initial projection
        self.proj = nn.Linear(input_dim, 1024)
        
        # ConvLSTM layers
        self.lstm = nn.ConvLSTM(
            input_size=1024,
            hidden_size=512,
            kernel_size=(3,3),
            num_layers=4,
            batch_first=True,
            projection_dim=256
        )
        
        # Upsampling blocks
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=(1,4,4), stride=(1,2,2)),
            nn.GELU(),
            nn.ConvTranspose3d(128, 64, kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1)),
            nn.GELU(),
            nn.ConvTranspose3d(64, 3, kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1))
        )
        
        # Temporal attention
        self.temp_attn = FlashCrossAttention(embed_dim=1024, num_heads=8)

    def forward(self, x, context=None):
        batch_size = x.size(0)
        
        # Projection initiale
        x = self.proj(x)
        
        # Ajout de la dimension temporelle
        x = x.unsqueeze(1)
        
        # ConvLSTM
        hidden = None
        outputs = []
        for _ in range(self.output_frames):
            x, hidden = self.lstm(x, hidden)
            outputs.append(x)
        
        # Concaténation temporelle
        x = torch.cat(outputs, dim=1)
        
        # Attention contextuelle
        if context is not None:
            x = self.temp_attn(x, context)
        
        # Upsampling 3D
        x = rearrange(x, 'b t c h w -> b c t h w')
        x = self.upsample(x)
        
        return {
            'output': x,
            'hidden_states': hidden
        }
