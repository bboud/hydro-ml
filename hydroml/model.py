from torch import nn

class Simple(nn.Module):
    def __init__(self):
        super(Simple, self).__init__()
        self.input_layer = self._block(1, 2, 73, 1, "same", 1)
        self.second_layer = self._block(2, 2, 145, 1, "same", 1)
        self.third_layer = self._block(2, 2, 145, 1, "same", 1)
        self.fourth_layer = self._block(2, 1, 73, 1, "same", 1)


    def _block(self, channels_in, channels_out, kernel_size, stride, padding, dilation):
        if padding == "same":
            padding = (kernel_size-1)//2
        return nn.Sequential(
            nn.Conv1d(channels_in, channels_out, kernel_size, stride, padding, dilation, bias=False),
        )

    def forward(self, t):
        t = self.input_layer(t)
        t = self.second_layer(t)
        t = self.third_layer(t)
        t = self.fourth_layer(t)
        return t

class DEModel(nn.Module):
    def __init__(self):
        super(DEModel, self).__init__()
        self.linear = nn.Linear(64, 141)

    def forward(self, t):
        t = self.linear(t)
        return t