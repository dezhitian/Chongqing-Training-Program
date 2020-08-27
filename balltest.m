%% written by No.12
clc;
clear all;
close all;
%% ��������
%��������
frames = 32;  %֡��
fs = 10e6; %������
n = 256;  %��ʱ�����
num = 255; %ÿ֡��������
k = 105e12;%��Ƶб��(Hz/s)
f0 = 77e9; %��Ƶ
c = 3e8;   
prt = 138e-6%�����ظ����
prf = 1 / prt;%�����ظ�Ƶ��
pfa = 1e-20;%�龯��
Npro = 8;%������Ԫ��Ŀ 
Nref = 20;%�ο���Ԫ��Ŀ
derta_R = c*fs/2/k/n;%����ֱ���
derta_V = c/2/255/(138e-6)/f0;%���ģ���ٶ�/255
%% ��άFFT
%��������
load ball_256_32_1result

data1 = adcData(1,:);%ȡ�ز����ݵĵ�1�У�����1���״���������յ����״�ز�����
data2 = adcData(2,:);%ȡ�ز����ݵĵ�2�У�����2���״���������յ����״�ز�����
data3 = adcData(3,:);%ȡ�ز����ݵĵ�3�У�����3���״���������յ����״�ز�����
data4 = adcData(4,:);%ȡ�ز����ݵĵ�4�У�����4���״���������յ����״�ز�����


s_data1 = reshape(data1,[256,255,32]);%256*255*32
s_data1 = permute(s_data1,[2 1 3]);%255*256*32
%ʱ��Ӵ����˳�������Ӱ��
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
%% ͨ��1
%��ʱ��άFFT
s_fft_data1 = zeros(255,256,32);%Ԥ�ȷ����ڴ棬��������ٶ�
s_fft2_data1 = zeros(254,256,32);
for i = 1:32
    for j = 1:255
    s_fft_data1(j,:,i)=fft(s_data1(j,:,i));
    end
end
%MTI����
for i = 1:32
    for j = 1:254
        s_data1(j,:,i) = s_fft_data1(j+1,:,i)-s_fft_data1(j,:,i);
    end
end 
%��ʱ��άFFT
for i = 1:32
    for j = 1:256
    s_fft2_data1(:,j,i)=fftshift(fft(s_data1(1:254,j,i)));
    end
end
%% ͨ��2
%��ʱ��άFFT
s_fft_data2 = zeros(255,256,32);%Ԥ�ȷ����ڴ棬��������ٶ�
s_fft2_data2 = zeros(254,256,32);
for i = 1:32
    for j = 1:255
    s_fft_data2(j,:,i)=fft(s_data2(j,:,i));
    end
end
%MTI����
for i = 1:32
    for j = 1:254
        s_data2(j,:,i) = s_fft_data2(j+1,:,i)-s_fft_data2(j,:,i);
    end
end 
%��ʱ��άFFT
for i = 1:32
    for j = 1:256
    s_fft2_data2(:,j,i)=fftshift(fft(s_data2(1:254,j,i)));
    end
end
%% ͨ��3
%��ʱ��άFFT
s_fft_data3 = zeros(255,256,32);%Ԥ�ȷ����ڴ棬��������ٶ�
s_fft2_data3 = zeros(254,256,32);
for i = 1:32
    for j = 1:255
    s_fft_data3(j,:,i)=fft(s_data3(j,:,i));
    end
end
%MTI����
for i = 1:32
    for j = 1:254
        s_data3(j,:,i) = s_fft_data3(j+1,:,i)-s_fft_data3(j,:,i);
    end
end 
%��ʱ��άFFT
for i = 1:32
    for j = 1:256
    s_fft2_data3(:,j,i)=fftshift(fft(s_data3(1:254,j,i)));
    end
end
%% ͨ��4
%��ʱ��άFFT
s_fft_data4 = zeros(255,256,32);%Ԥ�ȷ����ڴ棬��������ٶ�
s_fft2_data4 = zeros(254,256,32);
for i = 1:32
    for j = 1:255
    s_fft_data4(j,:,i)=fft(s_data4(j,:,i));
    end
end
%MTI����
for i = 1:32
    for j = 1:254
        s_data4(j,:,i) = s_fft_data4(j+1,:,i)-s_fft_data4(j,:,i);
    end
end 
%��ʱ��άFFT
for i = 1:32
    for j = 1:256
    s_fft2_data4(:,j,i)=fftshift(fft(s_data4(1:254,j,i)));
    end
end

%% ����λ��ۣ���ͨ���ز����ݷ�ֵ���ӣ����ʻ��ۣ���������ȣ�
dd = abs(s_fft2_data1)+abs(s_fft2_data2)+abs(s_fft2_data3)+abs(s_fft2_data4);
%keyboard
s_fft_2 = dd;
%% CA-CFAR���
s_cfar = abs(s_fft_2).^2; 
Nz = (Nref+Npro)/2;
s_R = zeros(254,284,32);%Ԥ�ȷ����ڴ棬��������ٶ�
s_RV = zeros(282,284,32);
% ���㣨��Ϊ�˱�֤ÿ������������Ա���ⵥԪ��⵽��
a = zeros(num-1,Nz,32);
b = zeros(Nz,n+2*Nz,32);
for i = 1:32
    s_R(:,:,i) = [a(:,:,i),s_cfar(:,:,i),a(:,:,i)];%��ʱ��ά����
    s_RV(:,:,i) = [b(:,:,i);s_R(:,:,i);b(:,:,i)]; %��ʱ��ά����
end

% ȷ��cfar�������
for q = 1:32
    for i = (Nz+1):(Nz+254) 
        for j = (Nz+1):(Nz+256) 
            omega1=2*(sum(s_RV(i,j-Nz:j+Nz,q))-sum(s_RV(i,j-Npro/2:j+Npro/2,q)))/Nref; 
            omega2=2*(sum(s_RV(i-Nz:i+Nz,j,q))-sum(s_RV(i-Npro/2:i+Npro/2,j,q)))/Nref; 
            K1(i-Nz,j-Nz,q) = (pfa^(-1/Nref)-1)*omega1;
            K2(i-Nz,j-Nz,q) = (pfa^(-1/Nref)-1)*omega2;
            K(i-Nz,j-Nz,q)=round((K1(i-Nz,j-Nz,q)+K2(i-Nz,j-Nz,q))/2);%ȡ��ֵ��ȡ��
        end
    end
end

%% ���
s_target = zeros(254,256,32);
for q = 1:32
    for i = 1:254  %��ѭ��
        for j = 1:256  %��ѭ��
           if s_cfar(i,j,q)>=K(i,j,q) 
               s_cfar(i,j,q)=s_cfar(i,j,q);
           else
               s_cfar(i,j,q)=0;
           end
        end
    end
%% �㼣����
    [x,y] = find(s_cfar(:,:,q)==max(max(s_cfar(:,:,q))));
    Y(q) = y;%ȡ��32֡�еľ�����Ϣ
    s_target(x,y,q)=s_cfar(x,y,q);
%% С��켣���PDͼ
    figure(1);
    imagesc((1:256)*derta_R,(-127:127)*derta_V,s_target(:,:,q));
    set(gca,'box','on','xlim',[0,2],'YDir','normal');; title('С��켣���RDͼ');xlabel('���루m��'); ylabel('�ٶȣ�m/s��'); pause(0.2);
end
%% ֡��-������ͼ
figure(2)
plot(1:32,(Y-1)*derta_R,'o');title('֡����������ͼ');axis([0 32 0 1.5]);xlabel('֡��');ylabel('���루m��');