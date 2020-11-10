clc
clear

%% =====================�ܳ�ʵ��4ͨ��ͼ===================
%   I = imread('D:\PycharmProjects\frames_4\frame1247.jpg');
%   I = I(:,:,1);
% [H,W] = size(I);  
% %
% I0 = I(H/2+1:H , W/2+1:W);
% I45 = I(1:H/2, W/2+1:W);
% I90 = I(1:H/2, 1:W/2); 
%%

%% =========================3ͨ��ͼ=========================
I0 = imread('0.png');   % 0��ͼ��
I45 = imread('45.png');  %  45��ͼ��
I90 = imread('90.png');   %  90��ͼ��

%I0 = imresize(I0,[480,640]); I45 = imresize(I45,[480,640]);  I90 = imresize(I90,[480,640]);

tic;
I0 = rgb2gray(I0);  I45 = rgb2gray(I45);   I90 = rgb2gray(I90);   % lucid����ɼ���Ϊ��ͨ���Ҷ�ͼ��ת��Ϊ��ͨ��ͼ
%I0 =  double(I0)/255  ;  I45 =  double(I45)/255;  I90 =  double(I90)/255;

%I = I0 + I90;   % stokes����
I = imread('fog20.jpg');   %Ҳ���Է���ƫ������ĵ�����ͼ
%I_max = max(max(I));  % ��������ȣ��������ع��ͼ��
Q = I0 - I90;
U = 2*I45-I;
%%

%% =============================ƫ����================================
DoP = (Q.^2 + U.^2 ).^0.5./I;   %ƫ���
AoP = atan(U./Q)/2;  %ƫ���

Imax = (1+DoP).*I/2;     % ����ǿͼ
Imin = (1-DoP).*I/2;     % ��С��ǿͼ
%Imin = (Imin-min(Imin(:)))./(max(Imin(:))-min(Imin(:)));  %������ķ�����ֱ��ͼ��׼����Ч��һ��
%Imax = ((1+DoP)./(1-DoP)).*Imin;
I_delta = Imax - Imin;   % ���ͼ
%%

%% ====================�������ƫ���PA����1��������򷨣�������һ�����=======================
binary = im2bw(I, graythresh(I));         % ostu��ֵ��map
%binary(800:1024,:)= 0;         % ����Ũ����������������֣���ֱ���ֶ�ȷ���������
se = strel('disk',9);           % ��ʴ�뾶9
binary = imerode(binary,se);       % ��ʴ����
sky_area = immultiply(I,binary);   %��ȡ�������

PA = immultiply(DoP,binary);   %�������ƫ���
%PA_area(isnan(PA_area)) = 0;
PA = mean(PA(PA ~= 0))*2.5;  %����ϵ��
%PA = 0.3;   % ����PA
%%

%% ======================�������ƫ���PA����2��Liang���������ڼ���Ũ��========================
[AoP_A,F] = mode(AoP(:));   % AoP�г���������Ϊ����ƫ���,FΪ����
PA = max(DoP(AoP == AoP_A));   % ��DoP��ӦԪ����ѡ���ֵ��Ϊ������ƫ���PA
%%

%% ======================== ������Զ������ǿA�ޣ���1����ͨ����===================
sky = sky_area(sky_area~=0);     % �Ƚ�����Ԫ��ȥ��
sky = (sort(sky,'descend'));
num = (fix(length(sky)*0.001));
A_inf = mean(sky(1:num));      % A��Ϊ���������0.01%�������ص�ƽ��ֵ

sky_max_value = max(sky);  % �鿴����������Ƿ�����ع�����
if(sky_max_value)>1
    A_inf = 0.95;   % �������ع⣬A���ֶ��趨Ϊ0.95
end
%%

%% =================== ������Զ������ǿA�ޣ���2��Liang�������������淨2��Ч������======================
A_prz = (I90 - (1-DoP).*I/2)/((sin(AoP_A))^2);  % ������ǿ��ƫ�񲿷�
A = A_prz./PA;      % �ο�Liang����
A_inf = 2*I0./(1+PA*cos(2*AoP_A));
mat = A_inf ./ I ;
A_inf = mean(mat((mat-1)<0.05));
ones = ones(size(I));
result = (I90 - A)./(ones-A/A_inf);
%%

%% =============== ������A���������������뵼���˲� =============
win_size = 15;    % �����˲�����
r = win_size*4;  % ���ڰ뾶
eps = 10^-6;     % ���򻯲���

A = I_delta./PA;   % �������A
A = guidedfilter(A,I90, r,eps);   %��I90Ϊ����ͼ
%%

%% ================ ���մ�������ģ�ͻָ� ======================
t = 1 - 0.95*A/A_inf; 
%result = (I - A)./t; 
result = dehaze(I);
imshow([I result]);

%%
toc;