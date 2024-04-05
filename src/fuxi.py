import torch
from torch import nn


class FuXi(torch.nn.Module):
    def __init__(self, input_var, latitutdes, longitudes, channels, transformer_block_count):
        self.layers = torch.nn.ModuleList()
        self.layers.append(CubeEmbedding(channels))
        self.layers.append(torch.nn.Conv2d(
            in_channels=,
            out_channels=,
            kernel_size=,
            stride=,
            padding=,
            dilation=,
        )

                           for _ in range(transformer_block_count):
        self.layers.append()


class SpaceTimeCubeEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpaceTimeCubeEmbedding, self).__init__()
        # 3D Convolution layer with kernel size and stride of (2, 4, 4)
        # This will reduce the temporal dimension by a factor of 2 and spatial dimensions by a factor of 4
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        # Applying 3D convolution
        x = self.conv3d(x)
        # Permuting the dimensions to apply LayerNorm correctly
        x_permuted = x.permute(1, 2, 3 ,0)  # Move the channel dimension to the end for LayerNorm
        # Applying layer normalization
        x_normalized = self.layer_norm(x_permuted)
        # Permuting back to the original dimension order
        x_out = x_normalized.permute(3, 0, 1, 2)
        return x_out
