%% written by No.12
clc;
clear all;
close all;
%% 参数设置
%基本参数
frames = 32;  %帧数
fs = 10e6; %采样率
n = 256;  %快时间点数
num = 255; %每帧的周期数
k = 105e12;%调频斜率(Hz/s)
f0 = 77e9; %载频
c = 3e8;   
prt = 138e-6%脉冲重复间隔
prf = 1 / prt;%脉冲重复频率
pfa = 1e-20;%虚警率
Npro = 8;%保护单元数目 
Nref = 20;%参考单元数目
derta_R = c*fs/2/k/n;%距离分辨率
derta_V = c/2/255/(138e-6)/f0;%最大不模糊速度/255
%% 二维FFT
%导入数据
load ball_256_32_1result

data1 = adcData(1,:);%取回波数据的第1行，即第1个雷达接收器接收到的雷达回波数据
data2 = adcData(2,:);%取回波数据的第2行，即第2个雷达接收器接收到的雷达回波数据
data3 = adcData(3,:);%取回波数据的第3行，即第3个雷达接收器接收到的雷达回波数据
data4 = adcData(4,:);%取回波数据的第4行，即第4个雷达接收器接收到的雷达回波数据


s_data1 = reshape(data1,[256,255,32]);%256*255*32
s_data1 = permute(s_data1,[2 1 3]);%255*256*32
%时域加窗，滤除噪声的影响
for q = 1:32
    for e = 1:255
        s_data1(e,:,q) = s_data1(e,:,q).*[hanning(75)',zeros(1,256-75)];
    end
end
    
s_data2 = reshape(data2,[256,255,32]);%256*255*32
s_data2 = permute(s_data2,[2 1 3]);%255*256*32
for q = 1:32
    for e = 1:255
        s_data2(e,:,q) = s_data2(e,:,q).*[hanning(75)',zeros(1,256-75)];
    end
end
    
s_data3 = reshape(data3,[256,255,32]);%256*255*32
s_data3 = permute(s_data3,[2 1 3]);%255*256*32
for q = 1:32
    for e = 1:255
        s_data3(e,:,q) = s_data3(e,:,q).*[hanning(75)',zeros(1,256-75)];
    end
end
    
s_data4 = reshape(data4,[256,255,32]);%256*255*32
s_data4 = permute(s_data4,[2 1 3]);%255*256*32
for q = 1:32
    for e = 1:255
        s_data4(e,:,q) = s_data4(e,:,q).*[hanning(75)',zeros(1,256-75)];
    end
end
%% 通道1
%快时间维FFT
s_fft_data1 = zeros(255,256,32);%预先分配内存，提高运算速度
s_fft2_data1 = zeros(254,256,32);
for i = 1:32
    for j = 1:255
    s_fft_data1(j,:,i)=fft(s_data1(j,:,i));
    end
end
%MTI对消
for i = 1:32
    for j = 1:254
        s_data1(j,:,i) = s_fft_data1(j+1,:,i)-s_fft_data1(j,:,i);
    end
end 
%慢时间维FFT
for i = 1:32
    for j = 1:256
    s_fft2_data1(:,j,i)=fftshift(fft(s_data1(1:254,j,i)));
    end
end
%% 通道2
%快时间维FFT
s_fft_data2 = zeros(255,256,32);%预先分配内存，提高运算速度
s_fft2_data2 = zeros(254,256,32);
for i = 1:32
    for j = 1:255
    s_fft_data2(j,:,i)=fft(s_data2(j,:,i));
    end
end
%MTI对消
for i = 1:32
    for j = 1:254
        s_data2(j,:,i) = s_fft_data2(j+1,:,i)-s_fft_data2(j,:,i);
    end
end 
%慢时间维FFT
for i = 1:32
    for j = 1:256
    s_fft2_data2(:,j,i)=fftshift(fft(s_data2(1:254,j,i)));
    end
end
%% 通道3
%快时间维FFT
s_fft_data3 = zeros(255,256,32);%预先分配内存，提高运算速度
s_fft2_data3 = zeros(254,256,32);
for i = 1:32
    for j = 1:255
    s_fft_data3(j,:,i)=fft(s_data3(j,:,i));
    end
end
%MTI对消
for i = 1:32
    for j = 1:254
        s_data3(j,:,i) = s_fft_data3(j+1,:,i)-s_fft_data3(j,:,i);
    end
end 
%慢时间维FFT
for i = 1:32
    for j = 1:256
    s_fft2_data3(:,j,i)=fftshift(fft(s_data3(1:254,j,i)));
    end
end
%% 通道4
%快时间维FFT
s_fft_data4 = zeros(255,256,32);%预先分配内存，提高运算速度
s_fft2_data4 = zeros(254,256,32);
for i = 1:32
    for j = 1:255
    s_fft_data4(j,:,i)=fft(s_data4(j,:,i));
    end
end
%MTI对消
for i = 1:32
    for j = 1:254
        s_data4(j,:,i) = s_fft_data4(j+1,:,i)-s_fft_data4(j,:,i);
    end
end 
%慢时间维FFT
for i = 1:32
    for j = 1:256
    s_fft2_data4(:,j,i)=fftshift(fft(s_data4(1:254,j,i)));
    end
end

%% 非相参积累（四通道回波数据幅值叠加，功率积累，增大信噪比）
dd = abs(s_fft2_data1)+abs(s_fft2_data2)+abs(s_fft2_data3)+abs(s_fft2_data4);
%keyboard
s_fft_2 = dd;
%% CA-CFAR检测
s_cfar = abs(s_fft_2).^2; 
Nz = (Nref+Npro)/2;
s_R = zeros(254,284,32);%预先分配内存，提高运算速度
s_RV = zeros(282,284,32);
% 补零（是为了保证每个采样点均可以被检测单元检测到）
a = zeros(num-1,Nz,32);
b = zeros(Nz,n+2*Nz,32);
for i = 1:32
    s_R(:,:,i) = [a(:,:,i),s_cfar(:,:,i),a(:,:,i)];%快时间维补零
    s_RV(:,:,i) = [b(:,:,i);s_R(:,:,i);b(:,:,i)]; %慢时间维补零
end

% 确定cfar检测门限
for q = 1:32
    for i = (Nz+1):(Nz+254) 
        for j = (Nz+1):(Nz+256) 
            omega1=2*(sum(s_RV(i,j-Nz:j+Nz,q))-sum(s_RV(i,j-Npro/2:j+Npro/2,q)))/Nref; 
            omega2=2*(sum(s_RV(i-Nz:i+Nz,j,q))-sum(s_RV(i-Npro/2:i+Npro/2,j,q)))/Nref; 
            K1(i-Nz,j-Nz,q) = (pfa^(-1/Nref)-1)*omega1;
            K2(i-Nz,j-Nz,q) = (pfa^(-1/Nref)-1)*omega2;
            K(i-Nz,j-Nz,q)=round((K1(i-Nz,j-Nz,q)+K2(i-Nz,j-Nz,q))/2);%取均值后取整
        end
    end
end

%% 检测
s_target = zeros(254,256,32);
for q = 1:32
    for i = 1:254  %行循环
        for j = 1:256  %列循环
           if s_cfar(i,j,q)>=K(i,j,q) 
               s_cfar(i,j,q)=s_cfar(i,j,q);
           else
               s_cfar(i,j,q)=0;
           end
        end
    end
%% 点迹凝聚
    [x,y] = find(s_cfar(:,:,q)==max(max(s_cfar(:,:,q))));
    Y(q) = y;%取出32帧中的距离信息
    s_target(x,y,q)=s_cfar(x,y,q);
%% 小球轨迹检测PD图
    figure(1);
    imagesc((1:256)*derta_R,(-127:127)*derta_V,s_target(:,:,q));
    set(gca,'box','on','xlim',[0,2],'YDir','normal');; title('小球轨迹检测RD图');xlabel('距离（m）'); ylabel('速度（m/s）'); pause(0.2);
end
%% 帧数-距离作图
figure(2)
plot(1:32,(Y-1)*derta_R,'o');title('帧数—距离作图');axis([0 32 0 1.5]);xlabel('帧数');ylabel('距离（m）');