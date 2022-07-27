from torch import nn

class BaryonModel(nn.Module):
    def __init__(self):
        super(BaryonModel, self).__init__()
        self.linear = nn.Linear(141, 141, bias=False)

    def forward(self, t):
        t = self.linear(t)
        return t

class DEModel(nn.Module):
    def __init__(self):
        super(DEModel, self).__init__()
        self.linear = nn.Linear(64, 141, bias=False)

    def forward(self, t):
        t = self.linear(t)
        return t