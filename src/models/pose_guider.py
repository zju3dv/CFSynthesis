from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from diffusers.models.modeling_utils import ModelMixin
from .transformer_3d import Transformer3DModel
from .motion_module import zero_module
from .resnet import InflatedConv3d
import torch

class PoseGuider(ModelMixin):
    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 64, 128),
        attention_num_heads: int = 8
    ):
        super().__init__()
        self.conv_in = InflatedConv3d(
            conditioning_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(
                InflatedConv3d(channel_in, channel_in, kernel_size=3, padding=1)
            )
            self.blocks.append(
                InflatedConv3d(
                    channel_in, channel_out, kernel_size=3, padding=1, stride=2
                )
            )

        self.attention=Transformer3DModel(
            attention_num_heads, 
            channel_out // attention_num_heads, 
            channel_out, norm_num_groups=32, unet_use_cross_frame_attention=False, unet_use_temporal_attention=False
        )

        self.conv_out = zero_module(
            InflatedConv3d(
                block_out_channels[-1],
                conditioning_embedding_channels,
                kernel_size=3,
                padding=1,
            )
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        # embedding = self.attention(embedding).sample
        embedding = self.conv_out(embedding)

        return embedding
    
# torch.save(embedding.squeeze(0).permute(1, 0, 2, 3)[0], "before.pt")
# torch.save(embedding.squeeze(0).permute(1, 0, 2, 3)[0], "after.pt")