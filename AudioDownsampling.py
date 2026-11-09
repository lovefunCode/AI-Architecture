import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioDownsampler(nn.Module):
    def __init__(self, num_mel_bins = 128, conv_channels = 64, d_model = 256):
        super().__init__()

        # Conv2d 1: [B, 1, T, F] --> [B, 64, T/2, F/2]
        self.conv2d1 = nn.Conv2d(
            in_channels=1,
            out_channels=conv_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )

       # Conv2d 2: [B, 64, T/2, F/2] --> [B, 64, T/4, F/4]
        self.conv2d2 = nn.Conv2d(
            in_channels=conv_channels,
            out_channels=conv_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )

        # Conv2d 3: [B, 64, T/4, T/4] --> [B, 64, T/8, T/8]
        self.conv2d3 = nn.Conv2d(
            in_channels=conv_channels,
            out_channels = conv_channels,
            kernel_size= 3,
            stride=2,
            padding=1
        )

        # calculate frequency dimension after 3 convolutions
        freq_dim  = num_mel_bins

        for _ in range(3):
            freq_dim = (freq_dim + 1) // 2

        # Flattened channels and frequency --> Transformer hidden dimension
        self.conv_out = nn.Linear(conv_channels * freq_dim, d_model, bias= True)

    def forward(self, x):
        # input: [B, T, F]
        # Add channel dimension: [B, 1, T, F]
        x = x.unsqueeze(1)

        x = F.gelu(self.conv2d1(x))
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))

        B, C, T, Freq = x.shape

        # Rearrange: [B, C, T, Freq] --> [B, T, C, Freq]
        x = x.permute(0, 2, 1, 3).contiguous()

        # Flatten C and freq: [B, T, C * freq]
        x = x.reshape(B, T, C*Freq)

        # Project to Transformer dimension: [B, T, d_model]
        x = self.conv_out(x)
        return x


#  Test
if __name__ == "__main__":
    model = AudioDownsampler(
        num_mel_bins=128,
        conv_channels= 64,
        d_model=256
    )

    # samples, 1000 time frames, 128 Mel bins
    x = torch.randn(2, 1000, 128)

    output = model(x)

    print('Input shape:', x.shape)
    print("Output shapes:", output.shape)


# python3 audioDownsample.py

# Input shape: torch.Size([2, 1000, 128])
# Output shapes: torch.Size([2, 125, 256])
