%%%%%%%��SARͼ�����ݱ�׼��%%%%%%%
clear 
clc
close all
%% ��ͼ�񣨸������ľ���ֵ��һ������׼��̬�ֲ�~��0��1��%%
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
%     disp(['����ʱ�䣺',num2str(toc)]);
% end
%% ��ͼ�����ֵdB��ʾ���һ����[0,1]%%
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
% disp(['����ʱ�䣺',num2str(toc)]);
%% ��ͼ�񣨸������ľ���ֵ��һ����[0��1]%%
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
% disp(['����ʱ�䣺',num2str(toc)]);
%% ��ͼ����ȵĹ�һ����[0��1]��ʵ�����鲿������ͨ������%%
load('F:\\Doctor\\LianMeng\\data\\Dual256\\data1.mat');
load('F:\\Doctor\\LianMeng\\data\\Dual256\\data2.mat');
load('F:\\Doctor\\LianMeng\\data\\Dual256\\label.mat');

[m,n,l]=size(data1);
data1_cvnor=zeros(2,m,n,l);                                                   % ��һ����ʵ�����鲿������ͨ���洢
data2_cvnor=zeros(2,m,n,l);  
label_nor=zeros(m,n,l);
tic
for i=1:l
    
    data1_cvnor(1,:,:,i)=(real(data1(:,:,i))-min(min(abs(data1(:,:,i)))))/(max(max(abs(data1(:,:,i))))-min(min(abs(data1(:,:,i)))));                  % ��һ�����ʵ��   
    data1_cvnor(2,:,:,i)=(imag(data1(:,:,i))-min(min(abs(data1(:,:,i)))))/(max(max(abs(data1(:,:,i))))-min(min(abs(data1(:,:,i)))));                  % ��һ������鲿
    
    data2_cvnor(1,:,:,i)=(real(data2(:,:,i))-min(min(abs(data2(:,:,i)))))/(max(max(abs(data2(:,:,i))))-min(min(abs(data2(:,:,i)))));                  % ��һ�����ʵ��   
    data2_cvnor(2,:,:,i)=(imag(data2(:,:,i))-min(min(abs(data2(:,:,i)))))/(max(max(abs(data2(:,:,i))))-min(min(abs(data2(:,:,i)))));                  % ��һ������鲿

%     label_cvnor(1,:,:,i)=(real(label(:,:,i))-min(min(abs(label(:,:,i)))))/(max(max(abs(label(:,:,i))))-min(min(abs(label(:,:,i)))));    
%     label_cvnor(2,:,:,i)=(imag(label(:,:,i))-min(min(abs(label(:,:,i)))))/(max(max(abs(label(:,:,i))))-min(min(abs(label(:,:,i)))));
    label_nor(:,:,i)=(abs(label(:,:,i))-min(min(abs(label(:,:,i)))))/(max(max(abs(label(:,:,i))))-min(min(abs(label(:,:,i)))));
    i
end
toc
disp(['����ʱ�䣺',num2str(toc)]);
%% ��������任Ϊ�����ֵ��ʵ����ע�����ﲢû�в���dB��ʾ%%
save('F:\\Doctor\\LianMeng\\data\\Dual256\\nor\\data1_cvnor.mat','data1_cvnor','-v7.3')
save('F:\\Doctor\\LianMeng\\data\\Dual256\\nor\\data2_cvnor.mat','data2_cvnor','-v7.3')
save('F:\\Doctor\\LianMeng\\data\\Dual256\\nor\\label_nor.mat','label_nor','-v7.3')