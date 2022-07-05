from torch import nn

class Simple(nn.Module):
    def __init__(self):
        super(Simple, self).__init__()
        self.input_layer = self._block(1, 1, 33, 1, "same", 1)
        self.second_layer = self._block(1, 1, 33, 1, "same", 1)

    def _block(self, channels_in, channels_out, kernel_size, stride, padding, dilation):
        if padding == "same":
            padding = (kernel_size-1)//2
        return nn.Sequential(
            nn.Conv1d(channels_in, channels_out, kernel_size, stride, padding, dilation, bias=False),
        )

    def forward(self, t):
        t = self.input_layer(t)
        t = self.second_layer(t)
        return t