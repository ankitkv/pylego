from torch import nn
from torch.nn import functional as F

from ..baseclf import BaseClassification


class Classifier(nn.Module):

    def __init__(self, input_size, hidden_size, n_classes):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        return self.linear2(F.elu(self.linear1(x.view(x.size(0), -1))))


class ClassificationModel(BaseClassification):

    def __init__(self, flags, *args, **kwargs):
        model = Classifier(28 * 28, flags.hidden_size, 10)
        super().__init__(model, flags, *args, **kwargs)

    def loss_function(self, forward_ret, labels=None):
        return F.cross_entropy(forward_ret, labels)
