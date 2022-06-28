from torch import nn

CHANNELS = 1

FEATURE_MAPS = 4

LATENT_SIZE = 100

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discrim = nn.Sequential(
            # torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
            # bias=True, padding_mode='zeros', device=None, dtype=None)
            nn.Conv1d(CHANNELS, FEATURE_MAPS, 7, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(FEATURE_MAPS, FEATURE_MAPS * 2, 7, 2, 1, bias=False),
            nn.BatchNorm1d(FEATURE_MAPS * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(FEATURE_MAPS * 2, FEATURE_MAPS * 4, 7, 2, 1, bias=False),
            nn.BatchNorm1d(FEATURE_MAPS * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(FEATURE_MAPS * 4, FEATURE_MAPS * 8, 7, 2, 1, bias=False),
            nn.BatchNorm1d(FEATURE_MAPS * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(FEATURE_MAPS * 4, 1, 7, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, t):
        return self.discrim(t)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
            # bias=True, padding_mode='zeros', device=None, dtype=None)

        )

    def forward(self, t):
        return self.gen(t)