import torch.nn as nn
import torch.nn.functional as F

class Dense(nn.Module):
    training : bool
    def __init__(self, in_dim, out_dim, num_classes):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_classes = num_classes

        super(Dense, self).__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(in_dim, out_dim)
        self.output = nn.Linear(out_dim, num_classes)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden(x)
        x = F.relu(x)
        x = self.output(x)
        x = F.log_softmax(x, dim=0)
        return x