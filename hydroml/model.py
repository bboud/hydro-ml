from torch import nn

class BaryonModel(nn.Module):
    def __init__(self):
        super(BaryonModel, self).__init__()

        # Input Layer
        self.linear_in = nn.Linear(141, 256)

        self.linear_1 = nn.Linear(256, 256)

        self.linear_2 = nn.Linear(256, 141)

        self.leaky = nn.LeakyReLU()

    def forward(self, t):
        t = self.leaky(self.linear_in(t))

        t = self.leaky(self.linear_1(t))

        t = self.leaky(self.linear_2(t))

        return t
    
class EccentricitiesModel(nn.Module):
    def __init__(self):
        super(EccentricitiesModel, self).__init__()

        self.layer_in = nn.Linear(64, 64)

        self.layer_1 = nn.Linear(64, 141)

        self.layer_out = nn.Linear(141, 141)

        self.leaky = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, t):

        t = self.leaky(self.layer_in(t))
        t = self.leaky(self.layer_1(t))
        t = self.tanh(self.layer_out(t))

        return t