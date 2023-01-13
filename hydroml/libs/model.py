from torch import nn

class BaryonModel(nn.Module):
    def __init__(self):
        super(BaryonModel, self).__init__()

        # Input Layer
        self.linear_in = nn.Linear(141, 141, bias=True)

        self.linear_1 = nn.Linear(141, 4, bias=True)

        self.linear_2 = nn.Linear(4, 141, bias=True)

        self.leaky = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, t):
        t = self.leaky(self.linear_in(t))

        t = self.leaky(self.linear_1(t))

        t = self.leaky(self.linear_2(t))

        return t

class BaryonLModel(nn.Module):
    def __init__(self):
        super(BaryonLModel, self).__init__()

        # Input Layer
        self.linear_in = nn.Linear(141, 141, bias=True)

    def forward(self, t):
        t = self.linear_in(t)

        return t

####

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

        self.lin1 = nn.Linear(462, 307, bias=False)
        self.lin2 = nn.Linear(307, 139, bias=False)
        self.lin3 = nn.Linear(139, 139, bias=False)

    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)

        return x