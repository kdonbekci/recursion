import torch.nn as nn
import torch.nn.functional as F


# vanilla CNN model
class VCNN(nn.Module):
    def __init__(self):
        super(VCNN, self).__init__()
        self.conv1 = nn.Conv2d(6, 8, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 10, 3)
        self.fc1 = nn.Linear(10 * 512 * 512, 5000)
        self.fc2 = nn.Linear(5000, 3000)
        self.fc3 = nn.Linear(3000, 1108)

    def forward(self, x):
        # two convolution layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # flatten the data
        x = x.view(-1, 10 * 512 * 512)

        # feed forward networks
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # final softmax layer
        x = nn.Softmax(self.fc3(x))
        return x
