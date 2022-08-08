from torch import nn

class BaryonModel(nn.Module):
    def __init__(self):
        super(BaryonModel, self).__init__()
        self.linear = nn.Linear(141, 141, bias=True)

    def forward(self, t):
        t = self.linear(t)
        return t

class DEModel(nn.Module):
    def __init__(self, size_1, size_2):
        super(DEModel, self).__init__()
        self.linear = nn.Linear(size_1, size_2)

    def forward(self, t):
        t = self.linear(t)
        return t