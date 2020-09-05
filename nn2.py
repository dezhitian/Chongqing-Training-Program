import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
#from torch import nn
import torchvision  #用于下载并导入数据集
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import torchvision.transforms as transforms
import net_cnn
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

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


'''标准化、图片变换'''
mean=[0.485, 0.456, 0.406]
stdv=[0.229, 0.224, 0.225]
train_transforms = transforms.Compose([
    #transforms.CenterCrop(224),#中心裁剪
    transforms.Resize([224,224]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=stdv)])

train_range_data = MyDataset(datatxt='train_range.txt', transform=train_transforms)
test_range_data = MyDataset(datatxt='val_range.txt', transform=train_transforms)

train_angle_data = MyDataset(datatxt='train_angle.txt', transform=train_transforms)
test_angle_data = MyDataset(datatxt='val_angle.txt', transform=train_transforms)

#装载训练集
train_range_loader = torch.utils.data.DataLoader(dataset=train_range_data, batch_size=8, shuffle=True)
test_range_loader = torch.utils.data.DataLoader(dataset=test_range_data, batch_size=8, shuffle=False)

train_angle_loader = torch.utils.data.DataLoader(dataset=train_angle_data, batch_size=8, shuffle=True)
test_angle_loader = torch.utils.data.DataLoader(dataset=test_angle_data, batch_size=8, shuffle=False)
#examples = enumerate(test_loader)
#batch_idx, (example_data, example_targets) = next(examples)
#print(example_targets)
#print(example_data.shape)


#选择示例网络
#net = fc_net_4layer()
#net = fc_net_2layer()
#net = CNN()
#net = LetNet()
#print(net)
net_range = net_cnn.Net_cnn_range()       
net_angle = net_cnn.Net_cnn_angle()
net_fc = net_cnn.fc_net()

#net_range.cuda()
#net_angle.cuda()
#net_fc.cude()
LR = 0.001  #学习率设置为0.001

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# LR = 0.001  #学习率设置为0.001
# net = LetNet().to(device)
#设置损失函数
criterion = nn.CrossEntropyLoss()  #交叉熵损失

#设置网络优化方式
optimizer_range = torch.optim.Adam(net_range.parameters(), LR )
optimizer_angle = torch.optim.Adam(net_angle.parameters(), LR )
optimizer_net = torch.optim.Adam(net_fc.parameters(), LR )


#开始训练
losses = []
acces = []
eval_losses = []
eval_acces = []
epoch = 20

test_range_loader_list = list(enumerate(test_range_loader))
test_angle_loader_list = list(enumerate(test_angle_loader))

#建立字典
state = {'net_range':net_range.state_dict(),'net_angle':net_angle.state_dict(),'net_fc':net_fc.state_dict(), 'optimizer_net':optimizer_net.state_dict(), 'epoch':epoch}

for epoch in range(epoch):
        train_range_loader_list = list(enumerate(train_range_loader))
        train_angle_loader_list = list(enumerate(train_angle_loader))
    

        train_acc_range = 0
        train_acc_angle = 0
        train_acc_net = 0
        net_range.train()
        net_angle.train()
        net_fc.train()

        train_loss_range= 0
        train_loss_angle = 0
        train_loss_net = 0

        for i in range(len(train_angle_loader_list)):

                img_range = train_range_loader_list[i][1][0]
                label_range = train_range_loader_list[i][1][1]
                img_range = Variable(img_range)
                label_range = Variable(label_range)
                #label_range = label_range.cuda()
                #img_range = img_range.cuda()

                img_angle = train_angle_loader_list[i][1][0]
                label_angle = train_angle_loader_list[i][1][1]
                img_angle = Variable(img_angle)
                #img_angle = img_angle.cuda()
                label_angle = Variable(label_angle)
               # label_angle = label_angle.cuda()


                out_range = net_range(img_range)
                loss_range = criterion(out_range, label_range)
                optimizer_range.zero_grad()
                loss_range.backward(retain_graph=True)
                optimizer_range.step()
                train_loss_range += loss_range.item()

                out_angle = net_angle(img_angle)
                loss_angle = criterion(out_angle, label_angle)
                optimizer_angle.zero_grad()
                loss_angle.backward(retain_graph=True)
                optimizer_angle.step()
                train_loss_angle += loss_angle.item()

                _,pred_range = out_range.max(1)
                num_correct_range = (pred_range == label_range).sum().item()
                acc_range = num_correct_range / img_range.shape[0]
                train_acc_range += acc_range

                _, pred_angle = out_angle.max(1)
                num_correct_angle = (pred_angle == label_angle).sum().item()
                acc_angle = num_correct_angle / img_angle.shape[0]
                train_acc_angle += acc_angle


                img_net = torch.cat((out_range, out_angle), 1)
                out_net = net_fc(img_net)
                loss_net = criterion(out_net, label_range)
                optimizer_net.zero_grad()
                loss_net.backward(retain_graph=True)
                optimizer_net.step()
                train_loss_net += loss_net.item()

                _, pred_net = out_net.max(1)
                num_correct_net = (pred_net == label_range).sum().item()
                acc_net = num_correct_net / img_range.shape[0]
                train_acc_net += acc_net

        train_acc_range = train_acc_range / len(train_angle_loader)*100
        train_acc_angle = train_acc_angle / len(train_angle_loader)*100
      
        train_acc_net = train_acc_net / len(train_angle_loader)*100
        print('***************** Epoch: ' + str(epoch) + ' *****************')
        # print('train_acc: ' + str(round(train_acc_DTM,3)) + ' ' + str((round(train_acc_ATM,3))) + ' ' + str((round(train_acc_RTM,3))) + ' ' + str((round(train_acc_net,3))))
        # print('train loss: ' + str(round(train_loss_DTM, 4)) + ' ' + str((round(train_loss_ATM, 4))) + ' ' + str((round(train_loss_RTM, 4))) + ' ' + str((round(train_loss_net, 4))))
        print('train_acc: ' + str(round(train_acc_net, 3)))



    # # 在测试集上检验效果
        test_acc_range= 0
        test_acc_angle = 0
        test_acc_net = 0
        net_range.eval()
        net_angle.eval()
        net_fc.eval()
        test_loss_range = 0
        test_loss_angle = 0
        test_loss_net = 0


        for j in range(len(test_angle_loader_list)):
                im_range = test_range_loader_list[j][1][0]
                label_range = test_range_loader_list[j][1][1]
                im_range = Variable(im_range)
                label_range = Variable(label_range)
                #label_range = label_range.cuda()
                #im_range = im_range.cuda()

                im_angle = test_angle_loader_list[j][1][0]
                label_angle = test_angle_loader_list[j][1][1]
                im_angle = Variable(im_angle)
                #im_angle = im_angle.cuda()
                label_angle = Variable(label_angle)
                #label_angle = label_angle.cuda()

                out_range = net_range(im_range)#输出神经网络的预测值
                loss_range = criterion(out_range, label_range)
                test_loss_range += loss_range.item()

                out_angle = net_angle(im_angle)
                loss_angle = criterion(out_angle, label_range)
                test_loss_angle += loss_angle.item()


                _, pred_range = out_range.max(1)
                num_correct_range = (pred_range == label_range).sum().item()
                acc_range = num_correct_range / im_range.shape[0]
                test_acc_range += acc_range

                _, pred_angle = out_angle.max(1)
                num_correct_angle = (pred_angle == label_range).sum().item()
                acc_angle = num_correct_angle / im_range.shape[0]
                test_acc_angle += acc_angle

                im_net = torch.cat((out_range, out_angle), 1)
                out_net = net_fc(im_net)
                loss_net = criterion(out_net, label_range)
                test_loss_net += loss_net.item()

                _, pred_net = out_net.max(1)
                num_correct_net = (pred_net == label_range).sum().item()
                acc_net = num_correct_net / im_range.shape[0]
                test_acc_net += acc_net

        test_acc_range = test_acc_range / len(test_angle_loader)*100
        test_acc_angle = test_acc_angle / len(test_angle_loader)*100
        test_acc_net = test_acc_net / len(test_angle_loader)*100
        # print(pred_net)
        # # print(pred_net.shape)
        # print('test_acc: ' + str(round(test_acc_DTM, 3)) + ' ' + str((round(test_acc_ATM, 3))) + ' ' + str((round(test_acc_RTM, 3))) + ' ' + str((round(test_acc_net, 3))))
        # print('test loss: ' + str(round(test_loss_DTM, 4)) + ' ' + str((round(test_loss_ATM, 4))) + ' ' + str((round(test_loss_RTM, 4))) + ' ' + str((round(test_loss_net, 4))))
        print('test_acc: ' + str(round(test_acc_net, 3)))
        if(epoch % 4 == 0):
            torch.save(state,'model_new.pkl')
        