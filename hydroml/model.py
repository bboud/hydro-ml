from torch import nn

class BaryonModel(nn.Module):
    def __init__(self, sizeInit, sizeFinal):
        super(BaryonModel, self).__init__()

        # Input Layer
        self.linear_in = nn.Linear(sizeInit, sizeInit)

        self.linear_1 = nn.Linear(sizeInit, sizeFinal)

        self.linear_2 = nn.Linear(sizeFinal, sizeFinal)

        self.leaky = nn.LeakyReLU()

    def forward(self, t):
        t = self.leaky(self.linear_in(t))

        t = self.leaky(self.linear_1(t))

        t = self.leaky(self.linear_2(t))

        return t
    
class EccentricitiesModel(nn.Module):
    def __init__(self, sizeInit, sizeFinal):
        super(EccentricitiesModel, self).__init__()

        self.layer_in = nn.Linear(sizeInit, 64)

        self.layer_1 = nn.Linear(64, 256)

        self.layer_out = nn.Linear(256, sizeFinal)

        self.leaky = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, t):

        t = self.leaky(self.layer_in(t))
        t = self.leaky(self.layer_1(t))
        t = self.tanh(self.layer_out(t))

        return t