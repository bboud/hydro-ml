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
        self.linear = nn.Linear(64, 141, bias=True)

    def forward(self, t):
        t = self.linear(t)
        return t

class DEConvolutionModel(nn.Module):
    def __init__(self):
        super(DEConvolutionModel, self).__init__()

        ichannels = 1

        self.conv1 = nn.Conv1d(in_channels= ichannels, out_channels= 1, kernel_size=17, stride=1, padding='same', bias=False)
        self.lin1 = nn.Linear(462, 400, bias=False)
        self.lin2 = nn.Linear(400, 200, bias=False)
        self.lin3 = nn.Linear(200, 139, bias=False)
        self.lin4 = nn.Linear(139, 139, bias=False)

        self.r = nn.ReLU()

    def forward(self, x):
        x = self.lin1(x)
        x = self.r(self.conv1(x))

        x = self.lin2(x)
        x = self.lin3(x)
        x = self.lin4(x)

        return x

class DEIntegralModel(nn.Module):
    def __init__(self):
        super(DEIntegralModel, self).__init__()

        self.lin1 = nn.Linear(141, 1, bias=False)

    def forward(self, x):
        x = self.lin1(x)

        return x