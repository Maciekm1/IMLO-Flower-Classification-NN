import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 102

class FavConvolutionalNetwork(nn.Module):
    def __init__(self):
        super(FavConvolutionalNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        self.batch_norm1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3)
        self.batch_norm2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
        self.batch_norm3 = nn.BatchNorm2d(512)

        self.conv4 = nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=3)
        self.batch_norm4 = nn.BatchNorm2d(2048)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(2048 * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.drop = nn.Dropout(p=0.7)

        self.out = nn.Linear(1024, NUM_CLASSES)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batch_norm1(x)
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.batch_norm2(x)
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.batch_norm3(x)
        x = self.pool(x)

        x = F.relu(self.conv4(x))
        x = self.batch_norm4(x)
        x = self.pool(x)

        x = x.view(-1, 2048 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)

        x = self.out(x)

        return x
