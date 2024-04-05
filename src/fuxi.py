import torch
from torchvision.models.swin_transformer import SwinTransformerBlockV2
from torch import nn


class FuXi(torch.nn.Module):
    def __init__(self, input_var, channels, transformer_block_count, lat, long, heads=8):
        super(FuXi, self).__init__()
        self.layers = torch.nn.ModuleList([
            SpaceTimeCubeEmbedding(input_var, channels),
            UTransformer(transformer_block_count, channels, input_var, heads, lat, long)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def training_step(self, batch) -> torch.Tensor:
        raise NotImplementedError()


class SpaceTimeCubeEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpaceTimeCubeEmbedding, self).__init__()
        self.layers = torch.nn.ModuleList(

        )
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x_permuted = x.permute(0, 2, 1, 3, 4)  # Move the channel dimension to the end for LayerNorm
        x = self.conv3d(x_permuted)
        x = x.permute(0, 2, 3, 4, 1)  # Move the channel dimension to the end for LayerNorm
        x_normalized = self.layer_norm(x)
        x_out = x_normalized.permute(0, 1, 4, 2, 3)
        x_out = torch.squeeze(x_out, dim=1)
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
        x = self.upsample(x)
        x = self.adjust_channels(x)
        residual = self.residual_block(x)
        x = x + residual
        return x


class UTransformer(torch.nn.Module):
    def __init__(self, layers, in_channels, out_channels, heads, lat, long):
        super().__init__()
        self.downblock = DownBlock(in_channels, in_channels)
        window_size = [8, 8]
        self.attentionblock = [
            SwinTransformerBlockV2(
                dim=in_channels,
                num_heads=heads,
                window_size=window_size,
                shift_size=[0 if i % 2 == 0 else w // 2 for w in window_size],
            ) for i in range(layers)
        ]
        self.upblock = UpBlock(in_channels, in_channels)
        self.fc = torch.nn.Linear(in_channels * (lat // 4) * (long // 4), out_channels * lat * long)
        self.lat = lat
        self.long = long

    def forward(self, x):
        bs = x.shape[0]
        down = self.downblock(x)
        x = down
        x = torch.permute(x, (0, 2, 3, 1))
        for block in self.attentionblock:
            x = block(x)
        x = torch.permute(x, (0, 3, 1, 2))
        x = self.upblock(x, down)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return torch.reshape(x, (bs, -1, self.lat, self.long))
