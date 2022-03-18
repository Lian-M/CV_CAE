%%%%%%%%%%%%%%%GEO�ǻ�˫����SAR��Ŀ����񣨼���ɢ��/δ����۽�����ͨ��%%%%%%%%%%%%%%
clear
clc
close all
%% ��������
c=3e8;                                                                     % ����
fc=1e10;                                                                    % ��Ƶ10GHz
lamda=c/fc;                                                                % �״ﲨ��
D=1;                                                                       % ���շɻ��ķ�λ��׾���m��
d=0.5;                                                                       % ���ջ���ͨ�����߼�ࣨm��
% PRF=256;                                                                 % �����ظ�Ƶ��
Re=6.371e6;                                                                % ����뾶
Ht=3.6e7;                                                                  % GEO���Ƿ��и߶�
Hr=8000;                                                                   % �ɻ����и߶�
Vs=2300;                                                                   % GEO���������ٶȣ��ع�����ϵ�£�
Vp=200;                                                                    % �ɻ������ٶ�(�����շ�ƽ̨����ƽ�з�����ͬ����Ŀ���ٶ���ƽ̨�ٶȷ���ļн�Ϊ30deg)
Num=60000;                                                                     % ����ѵ��ͼ�������
data1=zeros(256,256,Num);
data2=zeros(256,256,Num);
label=zeros(256,256,Num);
load('F:\\Doctor\\LianMeng\\data\\s_ut1.mat');                                                         % ����weibull�ֲ��Ӳ�
s_ut1t=s_ut1;
tic
parfor i=1:Num

Vt=1+(20-1)*rand(1);                                                       % Ŀ���ٶȾ��ȷֲ���1��20�����ں�ƽ��������ƽ̨�ٶȷ���ļн�Ϊ30deg
% Vt=20;
At=0+(2-0)*rand(1);                                                        % Ŀ����ٶȾ��ȷֲ���0��2��
% At=2;
% beta=(0+(3-0)*rand(1))*pi/180;                                           % Ŀ���ٶ���ƽ̨�ٶȷ����ں�ƽ���ϵļнǷ��Ӿ��ȷֲ���0��180��
beta=0*pi/180;                                                            

fai_T=0*pi/180;                                                            % GEO���Ŀ���б�ӽ�
theta_T=0.6*pi/180;                                                        % GEO���Ŀ������ӽ�
eca=pi-theta_T-(pi-asin((Re+Ht)*sin(theta_T)/Re));                         % GEO���Ŀ��ĵ��Ľ�
Vs_gr=Vs*Re*cos(eca)/(Re+Ht);                                              % GEO���������ٶ�
fai_R=0*pi/180;                                                            % �ɻ����Ŀ���б�ӽǷ��Ӿ��ȷֲ���0��10��
theta_R=(10+(30-10)*rand(1))*pi/180;                                       % �ɻ����Ŀ������ӽǷ��Ӿ��ȷֲ���10��30��
Rt0=Re*sin(eca)./sin(theta_T);                                             % GEO���Ǿ���Ŀ��ĳ�ʼ����
Rr0=Hr./cos(theta_R)./cos(fai_R);                                          % �ɻ�����Ŀ��ĳ�ʼ����
Vrad_T=Vt.*sin(beta).*sin(theta_T);                                        % Ŀ���ط���������ٶ�
Veff_s=sqrt(Vs*Vs_gr);                                                     % GEO�ĵ�Ч�ٶ�
Valong_T=Vt.*cos(beta);                                                    % Ŀ���ط���������ٶ�
Arad_T=At.*sin(beta).*sin(theta_T);                                        % Ŀ���ط����������ٶ�
Vrad_R=Vt.*sin(beta).*sin(theta_R).*cos(fai_R)+Vt.*cos(beta).*sin(fai_R);  % Ŀ���ؽ��ջ������ٶ�
Vp_gr=Vp;                                                                  % �ɻ��ĵ���ͶӰ�ٶ�
Valong_R=Vt.*cos(beta);                                                    % Ŀ���ؽ��ջ������ٶ�
Arad_R=At.*sin(beta).*sin(theta_R).*cos(fai_R)+At.*cos(beta).*sin(fai_R);  % Ŀ���ؽ��ջ�������ٶ�
sigma=0.8+(1.2-0.8)*rand(1);                                               % Ŀ��ķ���ϵ�����ȷֲ���0.8-1.2��
%%%%%%%%������%%%%%%%%%%%%%%
Tr=20e-6;                                                                  % ������
Br=150e6;                                                                  % �����źŴ���
kr=Br/Tr;                                                                  % �������Ƶб��
Nr=1024;                                                                   % �������������
r=(Rt0(1)+Rr0(1))/2+linspace(-400,400,Nr);                                 % ��������
dt=400*4/c/Nr;                                                             % ��������
fr=1/dt;                                                                   % ���������Ƶ��
Np=floor(Tr*fr);                                                           % �����ڲ�������255
t=(Rt0(1)+Rr0(1))/c+(-Np/2-Nr/2+1:Np/2+Nr/2)/fr;                           % ������ʱ����
Nfast=Np+Nr;
Nfft=Nfast;
f=linspace(-1/2/dt,1/2/dt,Nfast);                                          % f������
%%%%%%%%%��λ��%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
L=0.886*lamda/D*Rr0/cos(fai_R);                                            % ���ջ���λ��׾����
Na=2048;                                                                   % ��λ���������
du=2*L/Vp/Na;                                                              % ��λ���������
PRF=1/du;                                                                  % �����ظ�Ƶ��
u=(-Na/2:Na/2-1)*du;                                                               
fu=(-Na/2:Na/2-1)/du/Na;                                                   % ��λ��fu������
%% �ز�ģ��
s_utt1=zeros(Nfast,Na);                                                     % ��һ·ʱ��ز�����
s_utt2=zeros(Nfast,Na);                                                     % �ڶ�·ʱ��ز�����
U=ones(Nfast,1)*u;                                                         % ����Ϊ����
T=t'*ones(1,Na);
SCNR=0+(20-0)*rand(1);                                                     % SCNR���Ӿ��ȷֲ���0-20dB
% SCNR=0;

    Rt=Rt0+(Vs*fai_T+Vrad_T)*U+0.5*(Veff_s^2/Rt0-2*Vs*Valong_T/Rt0+Arad_T)*U.^2;                               % �������Ŀ���ľ�������ط�λ��仯�� 
    Rr1=Rr0+(Vp*fai_R+Vrad_R)*U+0.5*(Vp^2/Rr0-2*Vp*Valong_R/Rr0+Arad_R)*U.^2;                                  % ���ջ���һ·������Ŀ���ľ�������ط�λ��仯��
    Rr2=Rr0-d*fai_R+(Vp*fai_R+Vrad_R-d/Rr0*(Vp_gr-Valong_R))*U+0.5*(Vp^2/Rr0-2*Vp*Valong_R/Rr0+Arad_R)*U.^2;   % ���ջ��ڶ�·������Ŀ���ľ�������ط�λ��仯��
    R1=Rt+Rr1;
    R2=Rt+Rr2;
    DT1=T-R1/c;
    DT2=T-R2/c;
    phase1=pi*kr*DT1.^2-2*pi/lamda*R1;
    phase2=pi*kr*DT2.^2-2*pi/lamda*R2;
    s_utt1=s_utt1+sigma*exp(sqrt(-1)*phase1).*(abs(DT1)<Tr/2).*(abs(Vp*U)<L/2);                             % ��ʱ�������ǲ���������
    s_utt2=s_utt2+sigma*exp(sqrt(-1)*phase2).*(abs(DT2)<Tr/2).*(abs(Vp*U)<L/2);

P1=sum(sum(abs(s_utt1).^2))/4864/2048;                                     % ͨ��һĿ���źŻز�ƽ������
P2=sum(sum(abs(s_utt2).^2))/4864/2048;                                     % ͨ����Ŀ���źŻز�ƽ������
% load('D:\\matlab\\s_ut1.mat');                                                         % ����weibull�ֲ��Ӳ�
Pn=sum(sum(abs(s_ut1t).^2))/4864/2048;                                      % �Ӳ�ƽ������
rate1=sqrt(P1/(10^(SCNR/10))/Pn);                                          % ͨ��һ�Ӳ�������������
rate2=sqrt(P2/(10^(SCNR/10))/Pn);                                          % ͨ��һ�Ӳ�������������
s_ut1n=s_ut1t*rate1;                                                         % ͨ��һ�ж��Ӳ����Ƚ������ţ��������趨��SCNR
s_ut2n=s_ut1t*rate2;                                                         % ͨ�����ж��Ӳ����Ƚ������ţ��������趨��SCNR
s_utt11=s_utt1+s_ut1n;                                                      % ͨ��һ�źż�����źţ���������
s_utt22=s_utt2+s_ut2n;                                                      % ͨ�����źż�����źţ���������
s_utt=s_utt1;

% s_ut11=awgn(s_ut,SCNR);                                                    % ͨ��1�źż�����źţ���������
% s_ut22=s_ut;
%% ����ѹ��������RCMC
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
%% ��λ��ѹ��
CPa1=exp(-sqrt(-1)*2*pi/lamda*((Vs*fai_T+Vp*fai_R)*u+0.5*(Veff_s^2/Rt0+Vp^2/Rr0)*u.^2));  % �Ծ�ֹĿ��۽�������δ֪�˶�Ŀ�������
CPa2=exp(-sqrt(-1)*2*pi/lamda*((Vs*fai_T+Vrad_T+Vp*fai_R+Vrad_R)*u+0.5*(Veff_s^2/Rt0-2*Vs*Valong_T/Rt0+Arad_T+Vp^2/Rr0-2*Vp*Valong_R/Rr0+Arad_R)*u.^2));  % ���˶�Ŀ��۽���������֪�˶�Ŀ�������

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
%         Sa_ut1t(rr,aa)=Sa_ut1(2177+rr,769+aa);                                             % ��ͼ��ü���W*H��ȡԭʼͼ����������
%         Sa_ut2t(rr,aa)=Sa_ut2(2177+rr,769+aa);                                             % ��ʼ��λ��(4864/2+1-512/2),(2048/2+1-512/2)
%         Sa_utt(rr,aa)=Sa_ut(2177+rr,769+aa);
        Sa_ut1t(rr,aa)=Sa_ut1(2305+rr,897+aa);                                             % ��ͼ��ü���100*100��ȡԭʼͼ����������
        Sa_ut2t(rr,aa)=Sa_ut2(2305+rr,897+aa);                                             % ��ʼ��λ��(4864/2+1-256/2),(2048/2+1-256/2)
        Sa_utt(rr,aa)=Sa_ut(2305+rr,897+aa);
    end
end

    data1(:,:,i)=Sa_ut1t;
    data2(:,:,i)=Sa_ut2t;
    label(:,:,i)=Sa_utt;
%     data(1,:,:,i)=(real(Sa_ut1t)-min(min(abs(Sa_ut1t))))/(max(max(abs(Sa_ut1t)))-min(min(abs(Sa_ut1t))));                  % ��һ�����ʵ��  
%     data(2,:,:,i)=(imag(Sa_ut1t)-min(min(abs(Sa_ut1t))))/(max(max(abs(Sa_ut1t)))-min(min(abs(Sa_ut1t))));                  % ��һ������鲿
%     label(1,:,:,i)=(abs(Sa_ut2t))-min(min(abs(Sa_ut2t)))/(max(max(abs(Sa_ut2t)))-min(min(abs(Sa_ut2t))));

i
end
save('F:\\Doctor\\LianMeng\\data\\Dual256\\data1.mat','data1','-v7.3')
save('F:\\Doctor\\LianMeng\\data\\Dual256\\data2.mat','data2','-v7.3')
save('F:\\Doctor\\LianMeng\\data\\Dual256\\label.mat','label','-v7.3')
toc
disp(['����ʱ�䣺',num2str(toc)]);

% % G=20*log10(abs(data(:,:,2)));
% G1=20*log10(abs(Sa_ut1t));
% gm=max(max(G1));
% G1=(G1-gm);
% subplot(2,3,1)
% % mesh(u*Vp,(t-(Rt0+Rr0)/c)*c/2,G);
% imagesc(G1);
% xlabel('��λ�� (m)');ylabel('������ (m)');zlabel('���� (dB)');
% title('ͨ��1����ɢ���ĳ�����');
% 
% % G1=20*log10(abs(label(:,:,2)));
% G2=20*log10(abs(Sa_ut2t));
% gm=max(max(G2));
% G2=(G2-gm);
% subplot(2,3,2)
% % mesh(u*Vp,(t-(Rt0+Rr0)/c)*c/2,G1);
% imagesc(G2);
% xlabel('��λ�� (m)');ylabel('������ (m)');zlabel('���� (dB)');
% title('ͨ��2����ɢ���ĳ�����');
% 
% % G1=20*log10(abs(label(:,:,2)));
% G=20*log10(abs(Sa_utt));
% gm=max(max(G));
% G=(G-gm);
% subplot(2,3,3)
% % mesh(u*Vp,(t-(Rt0+Rr0)/c)*c/2,G1);
% imagesc(G);
% xlabel('��λ�� (m)');ylabel('������ (m)');zlabel('���� (dB)');
% title('ͨ��1����۽��ĳ�����');
% 
% subplot(2,3,4)
% mesh(G1);
% subplot(2,3,5)
% mesh(G2);
% subplot(2,3,6)
% mesh(G);







