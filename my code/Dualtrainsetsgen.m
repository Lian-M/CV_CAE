%%%%%%%%%%%%%%%GEO星机双基地SAR点目标成像（加噪散焦/未加噪聚焦）单通道%%%%%%%%%%%%%%
clear
clc
close all
%% 参数设置
c=3e8;                                                                     % 光速
fc=1e10;                                                                    % 载频10GHz
lamda=c/fc;                                                                % 雷达波长
D=1;                                                                       % 接收飞机的方位向孔径宽（m）
d=0.5;                                                                       % 接收机两通道天线间距（m）
% PRF=256;                                                                 % 脉冲重复频率
Re=6.371e6;                                                                % 地球半径
Ht=3.6e7;                                                                  % GEO卫星飞行高度
Hr=8000;                                                                   % 飞机飞行高度
Vs=2300;                                                                   % GEO卫星运行速度（地固坐标系下）
Vp=200;                                                                    % 飞机飞行速度(假设收发平台航迹平行方向相同，且目标速度与平台速度方向的夹角为30deg)
Num=60000;                                                                     % 生成训练图像的数量
data1=zeros(256,256,Num);
data2=zeros(256,256,Num);
label=zeros(256,256,Num);
load('F:\\Doctor\\LianMeng\\data\\s_ut1.mat');                                                         % 载入weibull分布杂波
s_ut1t=s_ut1;
tic
parfor i=1:Num

Vt=1+(20-1)*rand(1);                                                       % 目标速度均匀分布（1，20），在海平面上与与平台速度方向的夹角为30deg
% Vt=20;
At=0+(2-0)*rand(1);                                                        % 目标加速度均匀分布（0，2）
% At=2;
% beta=(0+(3-0)*rand(1))*pi/180;                                           % 目标速度与平台速度方向在海平面上的夹角服从均匀分布（0，180）
beta=0*pi/180;                                                            

fai_T=0*pi/180;                                                            % GEO相对目标的斜视角
theta_T=0.6*pi/180;                                                        % GEO相对目标的下视角
eca=pi-theta_T-(pi-asin((Re+Ht)*sin(theta_T)/Re));                         % GEO相对目标的地心角
Vs_gr=Vs*Re*cos(eca)/(Re+Ht);                                              % GEO波束地面速度
fai_R=0*pi/180;                                                            % 飞机相对目标的斜视角服从均匀分布（0，10）
theta_R=(10+(30-10)*rand(1))*pi/180;                                       % 飞机相对目标的下视角服从均匀分布（10，30）
Rt0=Re*sin(eca)./sin(theta_T);                                             % GEO卫星距离目标的初始距离
Rr0=Hr./cos(theta_R)./cos(fai_R);                                          % 飞机距离目标的初始距离
Vrad_T=Vt.*sin(beta).*sin(theta_T);                                        % 目标沿发射机径向速度
Veff_s=sqrt(Vs*Vs_gr);                                                     % GEO的等效速度
Valong_T=Vt.*cos(beta);                                                    % 目标沿发射机航迹速度
Arad_T=At.*sin(beta).*sin(theta_T);                                        % 目标沿发射机径向加速度
Vrad_R=Vt.*sin(beta).*sin(theta_R).*cos(fai_R)+Vt.*cos(beta).*sin(fai_R);  % 目标沿接收机径向速度
Vp_gr=Vp;                                                                  % 飞机的地面投影速度
Valong_R=Vt.*cos(beta);                                                    % 目标沿接收机航迹速度
Arad_R=At.*sin(beta).*sin(theta_R).*cos(fai_R)+At.*cos(beta).*sin(fai_R);  % 目标沿接收机径向加速度
sigma=0.8+(1.2-0.8)*rand(1);                                               % 目标的反射系数均匀分布（0.8-1.2）
%%%%%%%%距离向%%%%%%%%%%%%%%
Tr=20e-6;                                                                  % 脉冲宽度
Br=150e6;                                                                  % 发射信号带宽
kr=Br/Tr;                                                                  % 距离向调频斜率
Nr=1024;                                                                   % 距离向采样点数
r=(Rt0(1)+Rr0(1))/2+linspace(-400,400,Nr);                                 % 半距离和域
dt=400*4/c/Nr;                                                             % 采样周期
fr=1/dt;                                                                   % 距离向采样频率
Np=floor(Tr*fr);                                                           % 脉冲内采样点数255
t=(Rt0(1)+Rr0(1))/c+(-Np/2-Nr/2+1:Np/2+Nr/2)/fr;                           % 距离向时间域
Nfast=Np+Nr;
Nfft=Nfast;
f=linspace(-1/2/dt,1/2/dt,Nfast);                                          % f域序列
%%%%%%%%%方位向%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
L=0.886*lamda/D*Rr0/cos(fai_R);                                            % 接收机方位向孔径宽度
Na=2048;                                                                   % 方位向采样点数
du=2*L/Vp/Na;                                                              % 方位向采样周期
PRF=1/du;                                                                  % 脉冲重复频率
u=(-Na/2:Na/2-1)*du;                                                               
fu=(-Na/2:Na/2-1)/du/Na;                                                   % 方位向fu域序列
%% 回波模拟
s_utt1=zeros(Nfast,Na);                                                     % 第一路时域回波矩阵
s_utt2=zeros(Nfast,Na);                                                     % 第二路时域回波矩阵
U=ones(Nfast,1)*u;                                                         % 扩充为矩阵
T=t'*ones(1,Na);
SCNR=0+(20-0)*rand(1);                                                     % SCNR服从均匀分布：0-20dB
% SCNR=0;

    Rt=Rt0+(Vs*fai_T+Vrad_T)*U+0.5*(Veff_s^2/Rt0-2*Vs*Valong_T/Rt0+Arad_T)*U.^2;                               % 发射机与目标间的距离矩阵（沿方位向变化） 
    Rr1=Rr0+(Vp*fai_R+Vrad_R)*U+0.5*(Vp^2/Rr0-2*Vp*Valong_R/Rr0+Arad_R)*U.^2;                                  % 接收机第一路天线与目标间的距离矩阵（沿方位向变化）
    Rr2=Rr0-d*fai_R+(Vp*fai_R+Vrad_R-d/Rr0*(Vp_gr-Valong_R))*U+0.5*(Vp^2/Rr0-2*Vp*Valong_R/Rr0+Arad_R)*U.^2;   % 接收机第二路天线与目标间的距离矩阵（沿方位向变化）
    R1=Rt+Rr1;
    R2=Rt+Rr2;
    DT1=T-R1/c;
    DT2=T-R2/c;
    phase1=pi*kr*DT1.^2-2*pi/lamda*R1;
    phase2=pi*kr*DT2.^2-2*pi/lamda*R2;
    s_utt1=s_utt1+sigma*exp(sqrt(-1)*phase1).*(abs(DT1)<Tr/2).*(abs(Vp*U)<L/2);                             % 此时的数据是不含噪声的
    s_utt2=s_utt2+sigma*exp(sqrt(-1)*phase2).*(abs(DT2)<Tr/2).*(abs(Vp*U)<L/2);

P1=sum(sum(abs(s_utt1).^2))/4864/2048;                                     % 通道一目标信号回波平均功率
P2=sum(sum(abs(s_utt2).^2))/4864/2048;                                     % 通道二目标信号回波平均功率
% load('D:\\matlab\\s_ut1.mat');                                                         % 载入weibull分布杂波
Pn=sum(sum(abs(s_ut1t).^2))/4864/2048;                                      % 杂波平均功率
rate1=sqrt(P1/(10^(SCNR/10))/Pn);                                          % 通道一杂波幅度缩放因子
rate2=sqrt(P2/(10^(SCNR/10))/Pn);                                          % 通道一杂波幅度缩放因子
s_ut1n=s_ut1t*rate1;                                                         % 通道一中对杂波幅度进行缩放，以满足设定的SCNR
s_ut2n=s_ut1t*rate2;                                                         % 通道二中对杂波幅度进行缩放，以满足设定的SCNR
s_utt11=s_utt1+s_ut1n;                                                      % 通道一信号加噪后信号（威布尔）
s_utt22=s_utt2+s_ut2n;                                                      % 通道二信号加噪后信号（威布尔）
s_utt=s_utt1;

% s_ut11=awgn(s_ut,SCNR);                                                    % 通道1信号加噪后信号（白噪声）
% s_ut22=s_ut;
%% 距离压缩处理与RCMC
tr=t-(Rt0+Rr0)/c;  
CPr=exp(sqrt(-1)*pi*kr*tr.^2).*(abs(tr)<Tr/2);
FCPr=fft(CPr,Nfft);
Sr_ut1=zeros(Nfft,Na);   
Sr_ut2=zeros(Nfft,Na);
Sr_ut=zeros(Nfft,Na);
for k=1:Na
    temp1=fft(s_utt11(:,k).',Nfft);  
    temp2=fft(s_utt22(:,k).',Nfft); 
    temp=fft(s_utt(:,k).',Nfft); 
    FSr1=temp1.*conj(FCPr);
    FSr2=temp2.*conj(FCPr);
    FSr=temp.*conj(FCPr);
    G1=fftshift(ifft(FSr1));
    G2=fftshift(ifft(FSr2));
    G=fftshift(ifft(FSr));
    Sr_ut1(:,k)=G1.';
    Sr_ut2(:,k)=G2.';
    Sr_ut(:,k)=G.';
end
%% 方位向压缩
CPa1=exp(-sqrt(-1)*2*pi/lamda*((Vs*fai_T+Vp*fai_R)*u+0.5*(Veff_s^2/Rt0+Vp^2/Rr0)*u.^2));  % 对静止目标聚焦（假设未知运动目标参数）
CPa2=exp(-sqrt(-1)*2*pi/lamda*((Vs*fai_T+Vrad_T+Vp*fai_R+Vrad_R)*u+0.5*(Veff_s^2/Rt0-2*Vs*Valong_T/Rt0+Arad_T+Vp^2/Rr0-2*Vp*Valong_R/Rr0+Arad_R)*u.^2));  % 对运动目标聚焦（假设已知运动目标参数）

FCPa1=fft(CPa1,Na);
FCPa2=fft(CPa2,Na);
Sa_ut1=zeros(Nfft,Na);   
Sa_ut2=zeros(Nfft,Na);
Sa_ut=zeros(Nfft,Na);
for m=1:Nfft
    temp1=fft(Sr_ut1(m,:));  
    temp2=fft(Sr_ut2(m,:));
    temp=fft(Sr_ut(m,:));
    FSr1=temp1.*conj(FCPa1);
    FSr2=temp2.*conj(FCPa1);
    FSr=temp.*conj(FCPa2);
    Sa_ut1(m,:)=fftshift(ifft(FSr1));
    Sa_ut2(m,:)=fftshift(ifft(FSr2));
    Sa_ut(m,:)=fftshift(ifft(FSr));
end
Sa_ut1t=zeros(256,256);
Sa_ut2t=zeros(256,256);
Sa_utt=zeros(256,256);
for rr=1:256
    for aa=1:256
%         Sa_ut1t(rr,aa)=Sa_ut1(2177+rr,769+aa);                                             % 将图像裁剪成W*H（取原始图像中心区域）
%         Sa_ut2t(rr,aa)=Sa_ut2(2177+rr,769+aa);                                             % 起始点位：(4864/2+1-512/2),(2048/2+1-512/2)
%         Sa_utt(rr,aa)=Sa_ut(2177+rr,769+aa);
        Sa_ut1t(rr,aa)=Sa_ut1(2305+rr,897+aa);                                             % 将图像裁剪成100*100（取原始图像中心区域）
        Sa_ut2t(rr,aa)=Sa_ut2(2305+rr,897+aa);                                             % 起始点位：(4864/2+1-256/2),(2048/2+1-256/2)
        Sa_utt(rr,aa)=Sa_ut(2305+rr,897+aa);
    end
end

    data1(:,:,i)=Sa_ut1t;
    data2(:,:,i)=Sa_ut2t;
    label(:,:,i)=Sa_utt;
%     data(1,:,:,i)=(real(Sa_ut1t)-min(min(abs(Sa_ut1t))))/(max(max(abs(Sa_ut1t)))-min(min(abs(Sa_ut1t))));                  % 归一化后的实部  
%     data(2,:,:,i)=(imag(Sa_ut1t)-min(min(abs(Sa_ut1t))))/(max(max(abs(Sa_ut1t)))-min(min(abs(Sa_ut1t))));                  % 归一化后的虚部
%     label(1,:,:,i)=(abs(Sa_ut2t))-min(min(abs(Sa_ut2t)))/(max(max(abs(Sa_ut2t)))-min(min(abs(Sa_ut2t))));

i
end
save('F:\\Doctor\\LianMeng\\data\\Dual256\\data1.mat','data1','-v7.3')
save('F:\\Doctor\\LianMeng\\data\\Dual256\\data2.mat','data2','-v7.3')
save('F:\\Doctor\\LianMeng\\data\\Dual256\\label.mat','label','-v7.3')
toc
disp(['运行时间：',num2str(toc)]);

% % G=20*log10(abs(data(:,:,2)));
% G1=20*log10(abs(Sa_ut1t));
% gm=max(max(G1));
% G1=(G1-gm);
% subplot(2,3,1)
% % mesh(u*Vp,(t-(Rt0+Rr0)/c)*c/2,G);
% imagesc(G1);
% xlabel('方位向 (m)');ylabel('距离向 (m)');zlabel('幅度 (dB)');
% title('通道1加噪散焦的成像结果');
% 
% % G1=20*log10(abs(label(:,:,2)));
% G2=20*log10(abs(Sa_ut2t));
% gm=max(max(G2));
% G2=(G2-gm);
% subplot(2,3,2)
% % mesh(u*Vp,(t-(Rt0+Rr0)/c)*c/2,G1);
% imagesc(G2);
% xlabel('方位向 (m)');ylabel('距离向 (m)');zlabel('幅度 (dB)');
% title('通道2加噪散焦的成像结果');
% 
% % G1=20*log10(abs(label(:,:,2)));
% G=20*log10(abs(Sa_utt));
% gm=max(max(G));
% G=(G-gm);
% subplot(2,3,3)
% % mesh(u*Vp,(t-(Rt0+Rr0)/c)*c/2,G1);
% imagesc(G);
% xlabel('方位向 (m)');ylabel('距离向 (m)');zlabel('幅度 (dB)');
% title('通道1无噪聚焦的成像结果');
% 
% subplot(2,3,4)
% mesh(G1);
% subplot(2,3,5)
% mesh(G2);
% subplot(2,3,6)
% mesh(G);







