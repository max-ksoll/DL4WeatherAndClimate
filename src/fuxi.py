import torch


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


class CubeEmbedding(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Conv3d(
                in_channels=2,
                out_channels=channels,
                kernel_size=(2, 4, 4),
                stride=(2, 4, 4),
                padding=0,
                dilation=1,
            ),
            torch.nn.LayerNorm(channels)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
