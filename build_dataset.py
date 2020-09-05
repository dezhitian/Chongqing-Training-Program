from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import net
import os

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
mean = [0.5071, 0.4867, 0.4408]
stdv = [0.2675, 0.2565, 0.2761]
train_transforms = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=stdv)])

train_data = MyDataset(datatxt='train_range.txt', transform=train_transforms)

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=20, shuffle=True)

""" 训练时:"""
for data, label in train_loader:
    pass