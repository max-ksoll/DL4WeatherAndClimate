import torch
from torch import nn


class FuXi(torch.nn.Module):
    def __init__(self, input_var, latitutdes, longitudes, channels, transformer_block_count):
        self.layers = torch.nn.ModuleList()
        self.layers.append(SpaceTimeCubeEmbedding(channels))


class SpaceTimeCubeEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpaceTimeCubeEmbedding, self).__init__()
        self.layers = torch.nn.ModuleList(

        )
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x_permuted = x.permute(1, 0, 2, 3)  # Move the channel dimension to the end for LayerNorm
        x = self.conv3d(x_permuted)
        x = x.permute(1, 2, 3, 0)  # Move the channel dimension to the end for LayerNorm
        x_normalized = self.layer_norm(x)
        x_out = x_normalized.permute(0, 3, 1, 2)
        return x_out


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

        self.residual_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(2, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(2, out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        out = self.conv1(x)
        residual = self.residual_block(out)
        out = out + residual
        return out


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        # Scale the data size back up
        self.upsample = nn.ConvTranspose2d(
            in_channels * 2,
            in_channels * 2,
            kernel_size=2, stride=2)

        # Adjusting channels after concatenation
        self.adjust_channels = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)

        # Residual block similar to DownBlock but with adjusted channels due to skip connection
        self.residual_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(2, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(2, out_channels),
            nn.SiLU()
        )

    def forward(self, x, skip_connection):
        x = torch.cat([x, skip_connection], dim=1)
        print(x.shape)
        x = self.upsample(x)
        print(x.shape)
        x = self.adjust_channels(x)
        print(x.shape)
        residual = self.residual_block(x)
        x = x + residual
        return x


class UTransformer(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList([

        ])
