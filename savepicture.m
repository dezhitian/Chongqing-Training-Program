clc;close all;clear all;
Path = 'C:\Users\18845\Desktop\测试\result\qiantui\';                   % 设置数据存放的文件夹路径
File = dir(fullfile(Path,'*.mat'));  % 显示文件夹下所有符合后缀名为.mat文件的完整信息
FileNames = {File.name}';            % 提取符合后缀名为.mat的所有文件的文件名，转换为n行1列
Length_Names = size(FileNames,1);    % 获取所提取数据文件的个数
%% 参数设置
prt=138e-6;%prt为138us
f0=77e9;
c=3e8;
k=105e12;%调频斜率
fs=10e6;
N_z=8;%阵元数目，8个天线
M=1;%信元数目
Ns=256;
lamda=c/f0;
dd=0.6;%阵元间距
derad=pi/180;%角度到弧度
d=0:dd:(N_z-1)*dd;

N_p=10;%保护单元
N_c=20;%参考单元
a=15;
index=1+N_p/2+N_c/2:Ns-N_p/2-N_c/2;
sgn_cfa=zeros(1,Ns);

N=64;%%脉冲积累数
dtr=c*fs/(2*Ns*k);
dtv=c/(2*N*prt*f0);%最大不模糊速度/64
r=(1:Ns)*dtr;
v=(-31:32)*dtv;

for k = 1 : Length_Names
    close all;
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
 
    %load('0050_adc_data_Raw_0result.mat');
    detn = N*Ns;
    det=64*Ns;
    ss=zeros(32,detn);
    angle_r=zeros(361,32);
    [m1,n1]=size(adcData);
    adcData_2=zeros(m1,n1);
    for n=1:32
        adcData_2(:,(n-1)*det+1:(n-1)*det+Ns)=adcData(:,(n-1)*det+Ns+1:(n-1)*det+2*Ns)-adcData(:,(n-1)*det+1:(n-1)*det+Ns);
    end
    for n=1:32
        sgn_1=adcData_2(:,(n-1)*det+1:(n-1)*det+Ns);%取每帧的第一个脉冲
        
        %% CA-CFAR
        sgn_cfa=sgn_1;
        for i=index
            cell_left=sgn_cfa(i-N_c/2-N_p/2:i-N_p/2-1);
            cell_right=sgn_cfa(i+N_p/2+1:i+N_p/2+N_c/2);
            z=0.5*(mean(abs(cell_left))+mean(abs(cell_right)));
            xt=abs(z*a);
            if(abs(sgn_cfa(1,i))>xt)
                sgn_1(1,i)=sgn_1(1,i);
            else
                sgn_1(1,i)=0;
            end
        end
        for j=1:N_p/2+N_c/2
            cell_r=sgn_cfa(j+N_p/2+1:j+N_p/2+N_c/2);
            z_1=mean(abs(cell_r));
            xt_1=z_1*a;
            if(abs(sgn_cfa(1,j))>abs(xt_1))
                sgn_1(1,j)=sgn_1(1,j);
            else
                sgn_1(1,j)=0;
            end
        end
        for j=Ns-N_p/2-N_c/2+1:Ns
            cell_r=sgn_cfa(j-N_p/2-N_c/2:j-N_p/2-1);
            z_1=mean(abs(cell_r));
            xt_1=z_1*a;
            if(abs(sgn_cfa(1,j))>=abs(xt_1))
                sgn_1(1,j)=sgn_1(1,j);
            else
                sgn_1(1,j)=0;
            end
        end
        
        
        Rxx=sgn_1*sgn_1'/Ns;%%计算协方差
        [EV,D]=eig(Rxx);%特征值分解
        EVA=diag(D)';
        [EVA,I]=sort(EVA);%对特征值排序
        EV=fliplr(EV(:,I));%对应特征矢量排序
        angle=zeros(1,361);
        Pmusic=zeros(1,361);
        for iang=1:361
            angle(iang)=(iang-181)/2;
            phim=angle(iang)*derad;%角度转化为弧度
            a=exp(-1j*2*pi*d*sin(phim)).';
            En=EV(:,M+1:N_z);
            Pmusic(iang)=1/((a')*En*(En')*a);
        end
        Pmusic=abs(Pmusic);
        Pmmax=max(Pmusic);
        %   Pmusic(Pmusic<Pmmax)=0;
        Pmusic=10*log(Pmusic/Pmmax);
        angle_r(:,n)=Pmusic';
    end
    angle_rr=angle_r(120:240,1:1:32);
    angle_2=angle(120:240);
    [r_a,l_a]=size(angle_rr);
    zh=1:l_a;
    % zh=1:32;
    imagesc(zh,angle_2,angle_rr);
    xlabel('帧数');
    ylabel('角度');
    axis xy;

    f=getframe(gcf);
    
    imwrite(f.cdata,['C:\Users\18845\Desktop\图图图\角度\',str1,'.jpg'])
end