from torch import nn

class BaryonModel(nn.Module):
    def __init__(self):
        super(BaryonModel, self).__init__()

        # Input Layer
        self.linear_in = nn.Linear(141, 256, bias=False)

        self.linear_1 = nn.Linear(256, 256, bias=False)

        self.linear_2 = nn.Linear(256, 141, bias=False)

        self.leaky = nn.LeakyReLU()

    def forward(self, t):
        t = self.leaky(self.linear_in(t))

        t = self.leaky(self.linear_1(t))

        t = self.leaky(self.linear_2(t))

        return t