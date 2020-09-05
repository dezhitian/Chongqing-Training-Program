# hello-world
代码说明：
原始数据来源：基于77GHzFMCW毫米波雷达采集手势回波数据（.bin文件）

信号处理部分（基于MATLAB）
bin2mat_p_mimo.m:批处理代码，将文件夹里的.bin数据转化为.mat以供matlab后期处理
bin2mat_p_mimo.m:单个bin to mat处理代码
image_pro_RD.m：对回波数据进行距离维和多普勒维的2D-fft变换，提取距离和速度信息，为使特征信息明显，使用了MTI对消，非相参积累，CFAR检测以及点迹凝聚等处理手段。
savepicture.m:采用MUSIC算法提取手势的角度信息

神经网络搭建：(anaconda+torch(1.1.0)+torchvision(0.3.0))
label.py：给数据集（图片格式）加标签并按照一定比例划分测试集和训练集
net_cnn.py：采用cnn网络（4个卷积层，4个全连接层）
nn2.py：制定自己的数据集，并导入神经网络训练模型，每5个epoch保存一次模型
testmodel.py：导入训练好的模型，对测试集进行检测，分析训练效果
