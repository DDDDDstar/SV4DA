import torch
from torch import nn
import torch.nn.functional as F


class RegressionModel(nn.Module):
    def __init__(self, num_feature, num_classes):
        super(RegressionModel, self).__init__()
        self.linear1 = nn.Linear(num_feature, 8)
        self.drop = nn.Dropout(0.1)
        self.linear2 = nn.Linear(8, num_classes)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.drop(x1)
        outputs = self.linear2(x2)
        return outputs


class LinearAttackModel(nn.Module):
    def __init__(self, num_feature):
        super(LinearAttackModel, self).__init__()
        self.linear1 = nn.Linear(num_feature, num_feature)
        self.hidden = nn.Linear(num_feature, 4*num_feature)
        self.linear2 = nn.Linear(4*num_feature, num_feature)

    def forward(self, x):
        x1 = torch.sigmoid(self.linear1(x))
        x2 = torch.sigmoid(self.hidden(x1))
        outputs = self.linear2(x2)
        return outputs


class NN(nn.Module):
    def __init__(self, num_feature, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(num_feature, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(CNN, self).__init__()
        # block1
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=5, padding=0,
                               stride=1,
                               bias=True)
        self.pool1 = nn.MaxPool2d(2, 2)

        # block2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0,
                               stride=1,
                               bias=True)
        self.pool2 = nn.MaxPool2d(2, 2)

        # block3
        self.columns_fc1 = nn.Linear(1024, 512)

        # block4
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, inputs):

        x = self.pool1(F.relu(self.conv1(inputs)))

        x = self.pool2(F.relu(self.conv2(x)))

        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.columns_fc1(x))

        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return x


class CNNCifar(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(CNNCifar, self).__init__()
        self.all_layers = 5

        # block 1
        self.conv1 = nn.Conv2d(num_channels, 32, 5)
        self.pool1 = nn.MaxPool2d(2, 2)

        # block 2
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(2, 2)

        # block 3
        self.fc3 = nn.Linear(64 * 5 * 5, 512)

        # block 4
        self.fc4 = nn.Linear(512, 128)

        # block 5
        self.fc5 = nn.Linear(128, num_classes)

    def forward(self, inputs):

        x = self.pool1(F.relu(self.conv1(inputs)))
        x = self.pool2(F.relu(self.conv2(x)))

        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc3(x))

        x = F.relu(self.fc4(x))

        x = F.dropout(x, training=self.training)
        x = self.fc5(x)

        return x
