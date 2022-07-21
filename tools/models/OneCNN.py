import torch.nn as nn
import torch.nn.functional as F

class OneCNN(nn.Module):

    def __init__(self, in_channel, in_dim, num_classes):
        super(OneCNN, self).__init__()
        out_channel = 9
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(5,5))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(out_channel*in_dim, num_classes)
    
    def forward(self, x):
        x = nn.functional.relu(self.conv(x))
        x = self.flatten(x)
        x = nn.functional.log_softmax(self.fc(x), dim=1)
        return x