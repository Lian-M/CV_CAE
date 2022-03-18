%%%%%%%将SAR图像数据标准化%%%%%%%
clear 
clc
close all
%% 将图像（复数）的绝对值归一化到标准正态分布~（0，1）%%
% load('data.mat');
% load('label.mat');
% [m,n,l]=size(data);
% data_nor=zeros(m,n,l);
% label_nor=zeros(m,n,l);
% for i=1:l
%     tic
%     data_mean=mean2(abs(data(:,:,i)));
%     data_std=std2(abs(data(:,:,i)));
%     data_nor(:,:,i)=(abs(data(:,:,i))-data_mean)/data_std;
%     label_mean=mean2(abs(label(:,:,i)));
%     label_std=std2(abs(label(:,:,i)));
%     label_nor(:,:,i)=(abs(label(:,:,i))-label_mean)/label_std;
%     i
%     toc
%     disp(['运行时间：',num2str(toc)]);
% end
%% 将图像绝对值dB表示后归一化到[0,1]%%
% load('data.mat');
% load('label.mat');
% [m,n,l]=size(data);
% data_nor=zeros(m,n,l);
% label_nor=zeros(m,n,l);
% data=20*log10(abs(data));
% label=20*log10(abs(label));
% tic
% for i=1:l
%     
%     data_nor(:,:,i)=(data(:,:,i)-min(min(data(:,:,i))))/(max(max(data(:,:,i)))-min(min(data(:,:,i))));
% 
%     label_nor(:,:,i)=(label(:,:,i)-min(min(label(:,:,i))))/(max(max(label(:,:,i)))-min(min(label(:,:,i))));
%     i
% end
% toc
% disp(['运行时间：',num2str(toc)]);
%% 将图像（复数）的绝对值归一化到[0，1]%%
% load('F:\\Doctor\\LianMeng\\data\\data.mat');
% load('F:\\Doctor\\LianMeng\\data\\label.mat');
% [m,n,l]=size(data);
% data_nor=zeros(m,n,l);
% label_nor=zeros(m,n,l);
% tic
% parfor i=1:l
%     
%     data_nor(:,:,i)=(abs(data(:,:,i))-min(min(abs(data(:,:,i)))))/(max(max(abs(data(:,:,i))))-min(min(abs(data(:,:,i)))));
% 
%     label_nor(:,:,i)=(abs(label(:,:,i))-min(min(abs(label(:,:,i)))))/(max(max(abs(label(:,:,i))))-min(min(abs(label(:,:,i)))));
%     i
% end
% toc
% disp(['运行时间：',num2str(toc)]);
%% 将图像幅度的归一化到[0，1]，实部和虚部分两个通道保存%%
load('F:\\Doctor\\LianMeng\\data\\Dual256\\data1.mat');
load('F:\\Doctor\\LianMeng\\data\\Dual256\\data2.mat');
load('F:\\Doctor\\LianMeng\\data\\Dual256\\label.mat');

[m,n,l]=size(data1);
data1_cvnor=zeros(2,m,n,l);                                                   % 归一化后实部与虚部分两个通道存储
data2_cvnor=zeros(2,m,n,l);  
label_nor=zeros(m,n,l);
tic
for i=1:l
    
    data1_cvnor(1,:,:,i)=(real(data1(:,:,i))-min(min(abs(data1(:,:,i)))))/(max(max(abs(data1(:,:,i))))-min(min(abs(data1(:,:,i)))));                  % 归一化后的实部   
    data1_cvnor(2,:,:,i)=(imag(data1(:,:,i))-min(min(abs(data1(:,:,i)))))/(max(max(abs(data1(:,:,i))))-min(min(abs(data1(:,:,i)))));                  % 归一化后的虚部
    
    data2_cvnor(1,:,:,i)=(real(data2(:,:,i))-min(min(abs(data2(:,:,i)))))/(max(max(abs(data2(:,:,i))))-min(min(abs(data2(:,:,i)))));                  % 归一化后的实部   
    data2_cvnor(2,:,:,i)=(imag(data2(:,:,i))-min(min(abs(data2(:,:,i)))))/(max(max(abs(data2(:,:,i))))-min(min(abs(data2(:,:,i)))));                  % 归一化后的虚部

%     label_cvnor(1,:,:,i)=(real(label(:,:,i))-min(min(abs(label(:,:,i)))))/(max(max(abs(label(:,:,i))))-min(min(abs(label(:,:,i)))));    
%     label_cvnor(2,:,:,i)=(imag(label(:,:,i))-min(min(abs(label(:,:,i)))))/(max(max(abs(label(:,:,i))))-min(min(abs(label(:,:,i)))));
    label_nor(:,:,i)=(abs(label(:,:,i))-min(min(abs(label(:,:,i)))))/(max(max(abs(label(:,:,i))))-min(min(abs(label(:,:,i)))));
    i
end
toc
disp(['运行时间：',num2str(toc)]);
%% 复数矩阵变换为其绝对值的实矩阵，注意这里并没有采用dB表示%%
save('F:\\Doctor\\LianMeng\\data\\Dual256\\nor\\data1_cvnor.mat','data1_cvnor','-v7.3')
save('F:\\Doctor\\LianMeng\\data\\Dual256\\nor\\data2_cvnor.mat','data2_cvnor','-v7.3')
save('F:\\Doctor\\LianMeng\\data\\Dual256\\nor\\label_nor.mat','label_nor','-v7.3')