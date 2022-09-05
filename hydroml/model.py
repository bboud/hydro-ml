from torch import nn

class BaryonModel(nn.Module):
    def __init__(self):
        super(BaryonModel, self).__init__()
        self.linear = nn.Linear(141, 141, bias=True)

    def forward(self, t):
        t = self.linear(t)
        return t

class DELinearModel(nn.Module):
    def __init__(self):
        super(DELinearModel, self).__init__()
        self.linear = nn.Linear(500, 141, bias=True)

    def forward(self, t):
        t = self.linear(t)
        return t

class DEModel(nn.Module):
    def __init__(self):
        super(DEModel, self).__init__()

        ichannels = 1

        self.conv1 = nn.Conv1d(in_channels= ichannels, out_channels= 1, kernel_size=16, stride=1, bias=True)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels= 1, kernel_size=8, stride=1, bias=True)

        self.lin1 = nn.Linear(42, 9)

    def forward(self, x):
        r = nn.ReLU()

        x = r(self.conv1(x))
        #x = r(self.conv2(x))
        x = self.lin1(x)

        return x