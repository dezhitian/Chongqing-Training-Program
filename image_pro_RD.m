%% written by No.12
clc;close all;clear all;
Path = 'C:\Users\18845\Desktop\测试\result\qiantui\';                   % 设置数据存放的文件夹路径
File = dir(fullfile(Path,'*.mat'));  % 显示文件夹下所有符合后缀名为.mat文件的完整信息
FileNames = {File.name}';            % 提取符合后缀名为.mat的所有文件的文件名，转换为n行1列
Length_Names = size(FileNames,1);    % 获取所提取数据文件的个数
%keyboard

%% 参数设置
%基本参数
frames = 32;  %帧数
fs = 10e6; %采样率
n = 256;  %快时间点数
num = 64; %每帧的周期数
k = 105e12;%调频斜率(Hz/s)
f0 = 77e9; %载频
c = 3e8;
prt = 138e-6;%脉冲重复间隔
prf = 1 / prt;%脉冲重复频率
pfa = 1e-7;%虚警率
Npro = 6;%保护单元数目
Nref = 16;%参考单元数目
derta_R = c*fs/2/k/n;%距离分辨率c/2/B
derta_V = c/2/64/(138e-6)/f0;%最大不模糊速度/64  lamda/2/T
%% 二维FFT
%导入数据
for k = 1 : Length_Names
    close all;clc;
    % 连接路径和文件名得到完整的文件路径
    K_Trace = strcat(Path, FileNames(k));
    str1 =  FileNames(k);
    str1 = char(str1);
    % 读取数据（因为这里是.txt格式数据，所以直接用load()函数)
    %eval(['Data',num2str(k),'=','load(K_Trace{1,1})',';']);
    % 注意1：eval()函数是括号内的内容按照命令行执行，
    %       即eval(['a','=''2','+','3',';'])实质为a = 2 + 3;
    % 注意2：由于K_Trace是元胞数组格式，需要加{1,1}才能得到字符串
    % ========================
    load(K_Trace{1,1});
    %keyboard
    Data_FFT2 = cell(1,8);%存放每个通道的数据
    for r= 1:8
        data1 = adcData(r,:);%取回波数据的第r行，即第r个雷达接收器接收到的雷达回波数据
        s_data1 = reshape(data1,[256,64,32]);%256*64*32
        s_data1 = permute(s_data1,[2 1 3]);%255*256*32
        %时域加窗，滤除噪声的影响
        % for q = 1:32
        %     for e = 1:255
        %         s_data1(e,:,q) = s_data1(e,:,q).*[hanning(75)',zeros(1,256-75)];
        %     end
        % end
        
        %% 速度、距离FFT
        %快时间维FFT
        s_fft_data1 = zeros(64,256,32);%预先分配内存，提高运算速度
        s_fft2_data1 = zeros(63,256,32);
        for i = 1:frames
            for j = 1:64
                s_fft_data1(j,:,i)=fft(s_data1(j,:,i));
            end
        end
        %MTI对消
        for i = 1:frames
            for j = 1:63
                s_data1(j,:,i) = s_fft_data1(j+1,:,i)-s_fft_data1(j,:,i);
            end
        end
        %慢时间维FFT
        for i = 1:frames
            for j = 1:256
                s_fft2_data1(:,j,i)=fftshift(fft(s_data1(1:63,j,i)));
            end
        end
        
        Data_FFT2{r} = s_fft2_data1;
    end     
%% 非相参积累（八通道回波数据幅值叠加，功率积累，增大信噪比）
    for r =1:8
        s_fft_2 = abs(Data_FFT2{r});
    end
    
    %背景滤波
    for kk = 1:n
        if kk>27
            s_fft_2(:,kk) = 0;
        end
    end
%     nam1 = '后拉';
%     nam2 = int2str(k);
%     nam = [nam1,nam2];
%     %keyboard
%     save(['C:\Users\18845\Desktop\FFt2数据\RD\5\',nam],'s_fft_2');
    %keyboard
    %% CA-CFAR检测
    s_cfar = abs(s_fft_2).^2;
    Nz = (Nref+Npro)/2;
    % s_R = zeros(254,284,32);%预先分配内存，提高运算速度
    % s_RV = zeros(282,284,32);
    % 补零（是为了保证每个采样点均可以被检测单元检测到）
    a = zeros(num-1,Nz,32);
    b = zeros(Nz,n+2*Nz,32);
    for i = 1:32
        s_R(:,:,i) = [a(:,:,i),s_cfar(:,:,i),a(:,:,i)];%快时间维补零
        s_RV(:,:,i) = [b(:,:,i);s_R(:,:,i);b(:,:,i)]; %慢时间维补零
    end
    
    % 确定cfar检测门限
    for q = 1:32
        for i = (Nz+1):(Nz+63)
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
    s_target = zeros(63,256,32);
    for q = 1:32
        for i = 1:63  %行循环
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
        X(q) = x;%取出32帧中的速度信息
        %keyboard
        s_target(x,y,q)=s_cfar(x,y,q);
        % 小球轨迹检测PD图
%             figure(2);
%             imagesc((1:256)*derta_R,(-32:32)*derta_V,s_target(:,:,q));
%             set(gca,'box','on','xlim',[0,2],'YDir','normal'); title('动态手势RD图');xlabel('距离（m）'); ylabel('速度（m/s）'); pause(0.5);
%keyboard
    end
    %% 帧数-距离作图
%     path2 = ['C:\Users\18845\Desktop\图图图\速度\',path ];
%     path3 = [path2,'\',str0];
    figure(3)
    plot(1:32,(Y-1)*derta_R,'o');title('帧数—距离作图');axis([0 32 0 1.5]);xlabel('帧数');ylabel('距离（m）');   
    f=getframe(gcf);
    imwrite(f.cdata,['C:\Users\18845\Desktop\图图图\距离\',str1,'.jpg'])
    figure(4)
    plot(1:32,(32-X)*derta_V,'o');title('帧数—速度作图');axis([0 32 -7 7]);xlabel('帧数');ylabel('速度（m/s）');
    f=getframe(gcf);
    imwrite(f.cdata,['C:\Users\18845\Desktop\图图图\速度\',str1,'.jpg'])
    %keyboard
end