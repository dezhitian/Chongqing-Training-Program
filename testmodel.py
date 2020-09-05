import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision  #用于下载并导入数据集
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import torchvision.transforms as transforms
import net_cnn


class MyDataset(torch.utils.data.Dataset):  # 创类：MyDataset,继承torch.utils.data.Dataset
    def __init__(self, datatxt, transform=None):
        super(MyDataset, self).__init__()
        fh = open(datatxt, 'r')  # 打开txt，读取内容
        imgs = []
        for line in fh:  # 按行循环txt文本中的内容
            line = line.rstrip()  # 删除本行string字符串末尾的指定字符
            words = line.split()  # 通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
            imgs.append((words[0], int(words[1])))  # 把txt里的内容读入imgs列表保存，words[0]是图片信息，words[1]是label

        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):  # 按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]  # fn是图片path
        img = Image.open(fn).convert('RGB')  # from PIL import Image

        if self.transform is not None:  # 是否进行transform
            img = self.transform(img)
        return img, label  # return回哪些内容，在训练时循环读取每个batch，就能获得哪些内容

    def __len__(self):  # 它返回的是数据集的长度，必须有
        return len(self.imgs)

# optimizer_net.state_dict(checkpoint['optimizer'])

# start_epoch = checkpoint['epoch'] + 1
# 导入测试集
label_list = []
'''标准化、图片变换'''
mean=[0.485, 0.456, 0.406]
stdv=[0.229, 0.224, 0.225]
train_transforms = transforms.Compose([
    #transforms.CenterCrop(224),#中心裁剪
    transforms.Resize([224,224]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=stdv)])

#装载训练集
test_range_data = MyDataset(datatxt='val_range1.txt', transform=train_transforms)
test_angle_data = MyDataset(datatxt='val_angle1.txt', transform=train_transforms)


test_range_loader = torch.utils.data.DataLoader(dataset=test_range_data, batch_size=8, shuffle=False)
test_angle_loader = torch.utils.data.DataLoader(dataset=test_angle_data, batch_size=8, shuffle=False)

## 导入训练好的模型
checkpoint = torch.load('model.pkl')

model_net_range = net_cnn.Net_cnn_range() 
model_net_range.load_state_dict(checkpoint['net_range'])

model_net_angle = net_cnn.Net_cnn_angle() 
model_net_angle.load_state_dict(checkpoint['net_angle'])

model_net_fc = net_cnn.fc_net() 
model_net_fc.load_state_dict(checkpoint['net_fc'])

# print(model_net_range)
# args = checkpoint['epoch']

#设置损失函数
criterion = nn.CrossEntropyLoss()  #交叉熵损失
# # 在测试集上检验效果
test_acc_range= 0
test_acc_angle = 0
test_acc_net = 0
model_net_range.eval()
model_net_angle.eval()
model_net_fc.eval()

test_loss_range = 0
test_loss_angle = 0
test_loss_net = 0

test_range_loader_list = list(enumerate(test_range_loader))
test_angle_loader_list = list(enumerate(test_angle_loader))


for j in range(len(test_angle_loader_list)):
    im_range = test_range_loader_list[j][1][0]
    label_range = test_range_loader_list[j][1][1]
    im_range = Variable(im_range)
    label_range = Variable(label_range)
                
    im_angle = test_angle_loader_list[j][1][0]
    label_angle = test_angle_loader_list[j][1][1]
    im_angle = Variable(im_angle)
    label_angle = Variable(label_angle)


    out_range = model_net_range(im_range)#输出神经网络的预测值
    loss_range = criterion(out_range, label_range)
    test_loss_range += loss_range.item()

    out_angle = model_net_angle(im_angle)
    loss_angle = criterion(out_angle, label_range)
    test_loss_angle += loss_angle.item()


    _, pred_range = out_range.max(1)
    num_correct_range = (pred_range == label_range).sum().item()
    acc_range = num_correct_range / len(test_range_data) #im_range.shape[0]
    test_acc_range += acc_range

    _, pred_angle = out_angle.max(1)
    num_correct_angle = (pred_angle == label_range).sum().item()
    acc_angle = num_correct_angle / im_range.shape[0]
    test_acc_angle += acc_angle

    im_net = torch.cat((out_range, out_angle), 1)
    out_net = model_net_fc(im_net)
    loss_net = criterion(out_net, label_range)
    test_loss_net += loss_net.item()

    _, pred_net = out_net.max(1)
    num_correct_net = (pred_net == label_range).sum().item()
    acc_net = num_correct_net / im_range.shape[0]
    test_acc_net += acc_net
    label_list.append(pred_net)
#test_acc_range = test_acc_range / len(test_angle_loader)*100
#test_acc_angle = test_acc_angle / len(test_angle_loader)*100
#test_acc_net = test_acc_net / len(test_angle_loader)*100
        # print(pred_net)
        # # print(pred_net.shape)
        # print('test_acc: ' + str(round(test_acc_DTM, 3)) + ' ' + str((round(test_acc_ATM, 3))) + ' ' + str((round(test_acc_RTM, 3))) + ' ' + str((round(test_acc_net, 3))))
        # print('test loss: ' + str(round(test_loss_DTM, 4)) + ' ' + str((round(test_loss_ATM, 4))) + ' ' + str((round(test_loss_RTM, 4))) + ' ' + str((round(test_loss_net, 4))))
#print('test_acc: ' + str(round(test_acc_net, 3)))
print(label_list)