import torch
from torch.autograd import Variable
from torch import nn

def evaluate(params):
    model, x = params
    return model(x)

class ModelSum(nn.Module):

    def __init__(self, components=[]):
        super(ModelSum, self).__init__()
        self.components = nn.ModuleList(components)

    def component_list(self):
        return list(self.components)

    def next_block(self, model):
        l = self.component_list() + [model]
        return ModelSum(l)

    def training_component(self):
        return self.component_list()[-1]

    def parameters(self):
        return self.training_component().parameters()

    def freezed_components(self):
        return self.component_list()[:-1]

    def preout(self, x):
        values = list(map(evaluate, [(m, x) for m in self.freezed_components()]))
        if len(values) == 0:
            return None
        return torch.stack(values).sum(0)

    def forward(self, x):
        current = self.training_component()(x)
        pre = self.preout(x)
        if pre is not None:
            current += Variable(pre.data, requires_grad=False)
        return current

    def has_collapsed(self):
        return self.training_component().has_collapsed()

    def loss(self):
        return self.training_component().loss()

    def get_capacity(self):
        return torch.cat([c.get_capacity() for c in self.components]).sum()
