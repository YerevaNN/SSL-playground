import torch.nn as nn

class LeNet():
    def __init__(self):
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 4 * 4, 500)

        self.fc2 = nn.Linear(500, 10)

        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(nn.functional.elu(self.conv1(x)))
        x = self.pool(nn.functional.elu(self.conv2(x)))
        x = self.pool(nn.functional.elu(self.conv3(x)))

        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(x)
        x = nn.functional.elu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
