clc;close all;clear all;
Path = 'C:\Users\18845\Desktop\����\result\qiantui\';                   % �������ݴ�ŵ��ļ���·��
File = dir(fullfile(Path,'*.mat'));  % ��ʾ�ļ��������з��Ϻ�׺��Ϊ.mat�ļ���������Ϣ
FileNames = {File.name}';            % ��ȡ���Ϻ�׺��Ϊ.mat�������ļ����ļ�����ת��Ϊn��1��
Length_Names = size(FileNames,1);    % ��ȡ����ȡ�����ļ��ĸ���
%% ��������
prt=138e-6;%prtΪ138us
f0=77e9;
c=3e8;
k=105e12;%��Ƶб��
fs=10e6;
N_z=8;%��Ԫ��Ŀ��8������
M=1;%��Ԫ��Ŀ
Ns=256;
lamda=c/f0;
dd=0.6;%��Ԫ���
derad=pi/180;%�Ƕȵ�����
d=0:dd:(N_z-1)*dd;

N_p=10;%������Ԫ
N_c=20;%�ο���Ԫ
a=15;
index=1+N_p/2+N_c/2:Ns-N_p/2-N_c/2;
sgn_cfa=zeros(1,Ns);

N=64;%%���������
dtr=c*fs/(2*Ns*k);
dtv=c/(2*N*prt*f0);%���ģ���ٶ�/64
r=(1:Ns)*dtr;
v=(-31:32)*dtv;

for k = 1 : Length_Names
    close all;
    % ����·�����ļ����õ��������ļ�·��

    K_Trace = strcat(Path, FileNames(k));
    str1 =  FileNames(k);
    str1 = char(str1);
    % ��ȡ���ݣ���Ϊ������.txt��ʽ���ݣ�����ֱ����load()����)
    %eval(['Data',num2str(k),'=','load(K_Trace{1,1})',';']);
    % ע��1��eval()�����������ڵ����ݰ���������ִ�У�
    %       ��eval(['a','=''2','+','3',';'])ʵ��Ϊa = 2 + 3;
    % ע��2������K_Trace��Ԫ�������ʽ����Ҫ��{1,1}���ܵõ��ַ���
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
        sgn_1=adcData_2(:,(n-1)*det+1:(n-1)*det+Ns);%ȡÿ֡�ĵ�һ������
        
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
        
        
        Rxx=sgn_1*sgn_1'/Ns;%%����Э����
        [EV,D]=eig(Rxx);%����ֵ�ֽ�
        EVA=diag(D)';
        [EVA,I]=sort(EVA);%������ֵ����
        EV=fliplr(EV(:,I));%��Ӧ����ʸ������
        angle=zeros(1,361);
        Pmusic=zeros(1,361);
        for iang=1:361
            angle(iang)=(iang-181)/2;
            phim=angle(iang)*derad;%�Ƕ�ת��Ϊ����
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
    xlabel('֡��');
    ylabel('�Ƕ�');
    axis xy;

    f=getframe(gcf);
    
    imwrite(f.cdata,['C:\Users\18845\Desktop\ͼͼͼ\�Ƕ�\',str1,'.jpg'])
end