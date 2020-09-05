%% written by No.12
clc;close all;clear all;
Path = 'C:\Users\18845\Desktop\����\result\qiantui\';                   % �������ݴ�ŵ��ļ���·��
File = dir(fullfile(Path,'*.mat'));  % ��ʾ�ļ��������з��Ϻ�׺��Ϊ.mat�ļ���������Ϣ
FileNames = {File.name}';            % ��ȡ���Ϻ�׺��Ϊ.mat�������ļ����ļ�����ת��Ϊn��1��
Length_Names = size(FileNames,1);    % ��ȡ����ȡ�����ļ��ĸ���
%keyboard

%% ��������
%��������
frames = 32;  %֡��
fs = 10e6; %������
n = 256;  %��ʱ�����
num = 64; %ÿ֡��������
k = 105e12;%��Ƶб��(Hz/s)
f0 = 77e9; %��Ƶ
c = 3e8;
prt = 138e-6;%�����ظ����
prf = 1 / prt;%�����ظ�Ƶ��
pfa = 1e-7;%�龯��
Npro = 6;%������Ԫ��Ŀ
Nref = 16;%�ο���Ԫ��Ŀ
derta_R = c*fs/2/k/n;%����ֱ���c/2/B
derta_V = c/2/64/(138e-6)/f0;%���ģ���ٶ�/64  lamda/2/T
%% ��άFFT
%��������
for k = 1 : Length_Names
    close all;clc;
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
    %keyboard
    Data_FFT2 = cell(1,8);%���ÿ��ͨ��������
    for r= 1:8
        data1 = adcData(r,:);%ȡ�ز����ݵĵ�r�У�����r���״���������յ����״�ز�����
        s_data1 = reshape(data1,[256,64,32]);%256*64*32
        s_data1 = permute(s_data1,[2 1 3]);%255*256*32
        %ʱ��Ӵ����˳�������Ӱ��
        % for q = 1:32
        %     for e = 1:255
        %         s_data1(e,:,q) = s_data1(e,:,q).*[hanning(75)',zeros(1,256-75)];
        %     end
        % end
        
        %% �ٶȡ�����FFT
        %��ʱ��άFFT
        s_fft_data1 = zeros(64,256,32);%Ԥ�ȷ����ڴ棬��������ٶ�
        s_fft2_data1 = zeros(63,256,32);
        for i = 1:frames
            for j = 1:64
                s_fft_data1(j,:,i)=fft(s_data1(j,:,i));
            end
        end
        %MTI����
        for i = 1:frames
            for j = 1:63
                s_data1(j,:,i) = s_fft_data1(j+1,:,i)-s_fft_data1(j,:,i);
            end
        end
        %��ʱ��άFFT
        for i = 1:frames
            for j = 1:256
                s_fft2_data1(:,j,i)=fftshift(fft(s_data1(1:63,j,i)));
            end
        end
        
        Data_FFT2{r} = s_fft2_data1;
    end     
%% ����λ��ۣ���ͨ���ز����ݷ�ֵ���ӣ����ʻ��ۣ���������ȣ�
    for r =1:8
        s_fft_2 = abs(Data_FFT2{r});
    end
    
    %�����˲�
    for kk = 1:n
        if kk>27
            s_fft_2(:,kk) = 0;
        end
    end
%     nam1 = '����';
%     nam2 = int2str(k);
%     nam = [nam1,nam2];
%     %keyboard
%     save(['C:\Users\18845\Desktop\FFt2����\RD\5\',nam],'s_fft_2');
    %keyboard
    %% CA-CFAR���
    s_cfar = abs(s_fft_2).^2;
    Nz = (Nref+Npro)/2;
    % s_R = zeros(254,284,32);%Ԥ�ȷ����ڴ棬��������ٶ�
    % s_RV = zeros(282,284,32);
    % ���㣨��Ϊ�˱�֤ÿ������������Ա���ⵥԪ��⵽��
    a = zeros(num-1,Nz,32);
    b = zeros(Nz,n+2*Nz,32);
    for i = 1:32
        s_R(:,:,i) = [a(:,:,i),s_cfar(:,:,i),a(:,:,i)];%��ʱ��ά����
        s_RV(:,:,i) = [b(:,:,i);s_R(:,:,i);b(:,:,i)]; %��ʱ��ά����
    end
    
    % ȷ��cfar�������
    for q = 1:32
        for i = (Nz+1):(Nz+63)
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
    s_target = zeros(63,256,32);
    for q = 1:32
        for i = 1:63  %��ѭ��
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
        X(q) = x;%ȡ��32֡�е��ٶ���Ϣ
        %keyboard
        s_target(x,y,q)=s_cfar(x,y,q);
        % С��켣���PDͼ
%             figure(2);
%             imagesc((1:256)*derta_R,(-32:32)*derta_V,s_target(:,:,q));
%             set(gca,'box','on','xlim',[0,2],'YDir','normal'); title('��̬����RDͼ');xlabel('���루m��'); ylabel('�ٶȣ�m/s��'); pause(0.5);
%keyboard
    end
    %% ֡��-������ͼ
%     path2 = ['C:\Users\18845\Desktop\ͼͼͼ\�ٶ�\',path ];
%     path3 = [path2,'\',str0];
    figure(3)
    plot(1:32,(Y-1)*derta_R,'o');title('֡����������ͼ');axis([0 32 0 1.5]);xlabel('֡��');ylabel('���루m��');   
    f=getframe(gcf);
    imwrite(f.cdata,['C:\Users\18845\Desktop\ͼͼͼ\����\',str1,'.jpg'])
    figure(4)
    plot(1:32,(32-X)*derta_V,'o');title('֡�����ٶ���ͼ');axis([0 32 -7 7]);xlabel('֡��');ylabel('�ٶȣ�m/s��');
    f=getframe(gcf);
    imwrite(f.cdata,['C:\Users\18845\Desktop\ͼͼͼ\�ٶ�\',str1,'.jpg'])
    %keyboard
end