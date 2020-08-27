%%% This script is used to read the binary file produced by the DCA1000
%%% and Mmwave Studio
%%% Command to run in Matlab GUI -
%readDCA1000('<ADC capture bin file>') function [retVal] = readDCA1000(fileName)
%% global variables
% change based on sensor config
numADCSamples = 256; % number of ADC samples per chirp
numADCBits = 16; % number of ADC bits per sample
numRX = 4; % number of receivers
numLanes = 2; % do not change. number of lanes is always 2
isReal = 0; % set to 1 if real only data, 0 if complex data0
%% read file
% read .bin file
fid = fopen('D:\ti\data\����ʶ��\data_result_temp\youhua1.bin','r');
adcData = fread(fid, 'int16');
%% Raw�ļ�ת��
num735=int32(735);%��735ת��Ϊ����
adcDataLength=int32(length(adcData));%��ȡ��������
numl=idivide(adcDataLength,num735,'floor');%��ȡ����֡�ĸ���
numMod=mod(adcDataLength,num735);
num0=num735-numMod;%���㲹����
adcData=[adcData;zeros((num0),1)];%���ݲ���
matADCdata=reshape(adcData,735,numl+1);%����ת����
matADCdata(1:7,:)=[];%ȥ������֡�е�ͷ��
adcData=reshape(matADCdata,728*(numl+1),1);%����ת����
adcData((728*(numl+1)-num0+1):728*(numl+1)) = [];%ȥ��
%% ���ݽ��
% if 12 or 14 bits ADC per sample compensate for sign extension
if numADCBits ~= 16
l_max = 2^(numADCBits-1)-1;
adcData(adcData > l_max) = adcData(adcData > l_max) - 2^numADCBits;
end
fclose(fid);
fileSize = size(adcData, 1);
% real data reshape, filesize = numADCSamples*numChirps
if isReal
    numChirps = fileSize/numADCSamples/numRX;
    LVDS = zeros(1, fileSize);
    %create column for each chirp
    LVDS = reshape(adcData, numADCSamples*numRX, numChirps);
    %each row is data from one chirp
    LVDS = LVDS.';
else
% for complex data
% filesize = 2 * numADCSamples*numChirps
numChirps = fileSize/2/numADCSamples/numRX;
LVDS = zeros(1, fileSize/2);
%combine real and imaginary part into complex data
%read in file: 2I is followed by 2Q
counter = 1;
for i=1:4:fileSize-1
LVDS(1,counter) = adcData(i) + sqrt(-1)*adcData(i+2);
LVDS(1,counter+1) = adcData(i+1)+sqrt(-1)*adcData(i+3); 
counter = counter + 2;
end
% create column for each chirp
LVDS = reshape(LVDS, numADCSamples*numRX, numChirps);
%each row is data from one chirp
LVDS = LVDS.';
end
%organize data per RX
adcData = zeros(numRX,numChirps*numADCSamples);
for row = 1:numRX
    for i = 1: numChirps
        adcData(row, (i-1)*numADCSamples+1:i*numADCSamples) = LVDS(i, (row-1)*numADCSamples+1:row*numADCSamples);
    end
end
% return receiver data
retVal = adcData;
x1 = adcData.';
x2 = [];
for i = 1:4
temp1 = reshape(x1(:,i),numADCSamples,[]);
temp2 = temp1(:,1:2:end);
temp3 = temp1(:,2:2:end);
temp2 = reshape(temp2,1,[]);
temp3 = reshape(temp3,1,[]);
temp4 = [temp2;temp3];
x2 = [x2;temp4];
end
adcData = x2; 
save('D:\ti\data\����ʶ��\data_result_temp\mimo_youhua1.mat','adcData');
