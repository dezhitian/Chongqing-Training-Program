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
fid = fopen('D:\ti\data\手势识别\data_result_temp\youhua1.bin','r');
adcData = fread(fid, 'int16');
%% Raw文件转换
num735=int32(735);%把735转换为整型
adcDataLength=int32(length(adcData));%获取向量长度
numl=idivide(adcDataLength,num735,'floor');%获取数据帧的个数
numMod=mod(adcDataLength,num735);
num0=num735-numMod;%计算补零数
adcData=[adcData;zeros((num0),1)];%数据补零
matADCdata=reshape(adcData,735,numl+1);%向量转矩阵
matADCdata(1:7,:)=[];%去掉数据帧中的头部
adcData=reshape(matADCdata,728*(numl+1),1);%矩阵转向量
adcData((728*(numl+1)-num0+1):728*(numl+1)) = [];%去零
%% 数据解包
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
save('D:\ti\data\手势识别\data_result_temp\mimo_youhua1.mat','adcData');
