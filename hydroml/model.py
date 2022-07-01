from torch import nn

class Discriminator(nn.Module):
    def __init__(self, ndf):
        super(Discriminator, self).__init__()

        assert(ndf % 16 == 0 and ndf >= 16)

        self.ndf = ndf

        self.input_layer = self._block(1, (ndf // 16), 4, 2, 2, 1)

        self.layer1 = self._block((ndf // 16), (ndf // 8), 4, 2, 2, 1)

        self.layer2 = self._block((ndf // 8), (ndf // 4), 4, 2, 2, 1)

        self.layer3 = self._block((ndf // 4), (ndf // 2), 4, 2, 2, 1)

        self.layer4 = self._block((ndf // 2), (ndf), 4, 2, 2, 1)

        self.output_layer = self._output_block()

    # Repeatable Layer Block
    def _block(self, channels_in, channels_out, kernel_size, stride, padding, dilation):
        return nn.Sequential(
            nn.Conv1d(channels_in, channels_out, kernel_size, stride, padding, dilation, bias=False),
            nn.BatchNorm1d(channels_out),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.2),
        )

    def _output_block(self):
        # Shape: ndf x 4 - > 1
        return nn.Sequential(
            nn.Conv1d(self.ndf, 1, 4, 2, 0, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, t):
        t = self.input_layer(t)
        t = self.layer1(t)
        t = self.layer2(t)
        t = self.layer3(t)
        t = self.layer4(t)
        return self.output_layer(t)

class Generator(nn.Module):
    def __init__(self, ngf):
        super(Generator, self).__init__()

        assert(ngf % 16 == 0 and ngf >= 16)

        self.ngf = ngf

        # Input is 512 channels and we ConvTranspose down to 1 output channel with 128 elements.
        self.input_layer = self._block(ngf, (ngf // 2), 4, 2, 2, 1)

        self.layer1 = self._block((ngf // 2), (ngf // 4), 4, 2, 2, 1)

        self.layer2 = self._block((ngf // 4), (ngf // 8), 4, 2, 2, 1)

        self.layer3 = self._block((ngf // 8), (ngf // 16), 4, 2, 2, 1)

        self.layer4 = self._block((ngf // 16), 1, 4, 2, 2, 1)

        self.output_layer = self._output_block()

    # Repeatable Layer Block
    def _block(self, channels_in, channels_out, kernel_size, stride, padding, dilation):
        return nn.Sequential(
            nn.Conv1d(channels_in, channels_out, kernel_size, stride, padding, dilation, bias=False),
            nn.BatchNorm1d(channels_out),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.2),
        )

    def _output_block(self):
        return nn.Sequential(
            # ConvTranspose up to 128 elements.
            nn.Conv1d( 1, 1, 4, 2, 0, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, t):
        t = self.input_layer(t)
        t = self.layer1(t)
        t = self.layer2(t)
        t = self.layer3(t)
        t = self.layer4(t)
        return self.output_layer(t)