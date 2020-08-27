import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time

a = time.time()

data_tf = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize([0.5],[0.5])])

#准备训练数据
train_set = torchvision.datasets.MNIST('./data', train=True,  transform=data_tf, download=True)
#准备测试数据
test_set = torchvision.datasets.MNIST('./data', train=False,  transform=data_tf, download=True)

train_data = DataLoader(train_set, batch_size=64, shuffle=True) # 64
test_data = DataLoader(test_set, batch_size=128, shuffle=False) # 128

#示例网络1
class fc_net_2layer(nn.Module):
    def __init__(self):
        super(fc_net_2layer, self).__init__()
        self.fc = nn.Sequential(
                    nn.Linear(28 * 28, 10),
                    nn.ReLU(),
                    nn.Linear(10, 10) #最后输出10个分类
                )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

#示例网络2
class fc_net_4layer(nn.Module):
    def __init__(self):
        super(fc_net_4layer, self).__init__()
        self.fc = nn.Sequential(
                    nn.Linear(28 * 28, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10) #最后输出10个分类
                )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

#示例网络3
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        
        self.layer1 = nn.Sequential(
                nn.Conv2d(1,16,kernel_size=3), # 16, 26 ,26
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True))
        
        self.layer2 = nn.Sequential(
                nn.Conv2d(16,32,kernel_size=3),# 32, 24, 24
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2,stride=2)) # 32, 12,12     (24-2) /2 +1
        
        self.layer3 = nn.Sequential(
                nn.Conv2d(32,64,kernel_size=3), # 64,10,10
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True))
        
        self.layer4 = nn.Sequential(
                nn.Conv2d(64,128,kernel_size=3),  # 128,8,8
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2,stride=2))  # 128, 4,4
        
        self.fc = nn.Sequential(
                nn.Linear(128 * 4 * 4,64),
                nn.ReLU(inplace=True),
                nn.Linear(64,32),
                nn.ReLU(inplace=True),
                nn.Linear(32,10))
        
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        
        return x

#选择示例网络
net = fc_net_4layer()
#net = fc_net_2layer()
#net = CNN()
#print(net)

#设置损失函数
criterion = nn.CrossEntropyLoss()
#设置网络优化方式
optimizer = torch.optim.SGD(net.parameters(), 1e-2) #学习率0.1 0.01

losses = []
acces = []
eval_losses = []
eval_acces = []

#开始训练 
for e in range(10):
    train_loss = 0
    train_acc = 0
    net.train()
    for im, label in train_data:
        im = Variable(im)
        label = Variable(label)
        # 前向传播
        out = net(im)
        loss = criterion(out, label)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.item()
        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        train_acc += acc
        #print('epoch: {}, Batch Train Loss: {:.6f}, Bacth Train Acc: {:.6f}'.format(e, loss.item(), acc))
        
    losses.append(train_loss / len(train_data))
    acces.append(train_acc / len(train_data))
    # 在测试集上检验效果
    eval_loss = 0
    eval_acc = 0
    net.eval() # 将模型改为预测模式
    for im, label in test_data:
        im = Variable(im)
        label = Variable(label)
        out = net(im)
        loss = criterion(out, label)
        # 记录误差
        eval_loss += loss.item()
        # 记录准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        eval_acc += acc
        #print('epoch: {}, Batch Evaluate Loss: {:.6f}, Bacth Evaluate Acc: {:.6f}'.format(e, loss.item(), acc))
        
    eval_losses.append(eval_loss / len(test_data))
    eval_acces.append(eval_acc / len(test_data))
    print('***** One epoch has finished ******')
    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
          .format(e, train_loss / len(train_data), train_acc / len(train_data), 
                     eval_loss / len(test_data), eval_acc / len(test_data)))
b = time.time()
print(b - a)