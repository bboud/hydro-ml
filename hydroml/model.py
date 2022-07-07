from torch import nn

class Simple(nn.Module):
    def __init__(self):
        super(Simple, self).__init__()
        self.input_layer = self._block(1, 2, 73, 1, "same", 1)
        self.second_layer = self._block(2, 2, 145, 1, "same", 1)
        self.third_layers = self._block(2, 2, 145, 1, "same", 1)
        self.fourth_layers = self._block(2, 1, 73, 1, "same", 1)


    def _block(self, channels_in, channels_out, kernel_size, stride, padding, dilation):
        if padding == "same":
            padding = (kernel_size-1)//2
        return nn.Sequential(
            nn.Conv1d(channels_in, channels_out, kernel_size, stride, padding, dilation, bias=False),
        )

    def forward(self, t):
        t = self.input_layer(t)
        t = self.second_layer(t)
        t = self.third_layers(t)
        t = self.fourth_layers(t)
        return t