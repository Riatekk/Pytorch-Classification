import torch.nn as nn
import torch.nn.functional as F

class MultiLayerCNN(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(MultiLayerCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 10, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc = nn.Linear(20*4*4, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 20*4*4)
        x = nn.functional.log_softmax(self.fc(x), dim=1)
        return x