import torch.nn as nn

class Net_cnn_range(nn.Module):
    def __init__(self):
        super(Net_cnn_range, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, 3, 1, 1),  #输入为3层，输出6层，卷积核为3*3，步长为1，pad为1
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  #池化
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(14 * 14 * 64, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(nn.Linear(1024, 256), nn.BatchNorm1d(256),
                                 nn.ReLU(), nn.Linear(256, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class Net_cnn_angle(nn.Module):
    def __init__(self):
        super(Net_cnn_angle, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, 3, 1, 1),  #输入为1层，输出6层，卷积核为3*3，步长为1，pad为1
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  #池化
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(14 * 14 * 64, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(nn.Linear(1024, 256), nn.BatchNorm1d(256),
                                 nn.ReLU(), nn.Linear(256, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class fc_net(nn.Module):
    def __init__(self):
        super(fc_net,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(20, 10)
        )
    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = self.fc(x)

        return x       