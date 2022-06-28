from torch import nn

ndf = 512

# torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
# bias=True, padding_mode='zeros', device=None, dtype=None)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        assert(ndf % 16 == 0 and ndf >= 16)

        self.discrim = nn.Sequential(
            nn.Conv1d(1, (ndf // 16), 4, 2, 0, 1),
            nn.LeakyReLU(0.2, inplace=True),

            # Shape: 1 x 128; One channel by 128 elements
            self._block(1, (ndf // 16), 4, 2, 0, 1),

            #Shape: ndf//16 x 64
            self._block((ndf // 16), (ndf // 8), 4, 2, 0, 1),

            # Shape: ndf//8 x 32
            self._block((ndf // 8), (ndf // 4), 4, 2, 0, 1),

            # Shape: ndf//4 x 16
            self._block((ndf // 4), (ndf // 2), 4, 2, 0, 1),

            # Shape: ndf//2 x 8
            self._block((ndf // 2), (ndf), 4, 2, 0, 1),

            # Shape: ndf x 4
            nn.Conv1d(ndf, 1, 4, 2, 0, 1, bias=False),
        )

    # Repeatable Layer Block
    def _block(self, channels_in, channels_out, kernel_size, stride, padding, dilation):
        return nn.Sequential(
            nn.Conv1d(channels_in, channels_out, kernel_size, stride, padding, dilation, bias=False),
            nn.BatchNorm1d(channels_out),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.2),
        )

    def forward(self, t):
        return self.discrim(t)

# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.gen = nn.Sequential(
#             # torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
#             # bias=True, padding_mode='zeros', device=None, dtype=None)
#         )
#
#     def forward(self, t):
#         return self.gen(t)