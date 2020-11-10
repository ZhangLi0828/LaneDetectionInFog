clc
clear

%% =====================跑车实验4通道图===================
%   I = imread('D:\PycharmProjects\frames_4\frame1247.jpg');
%   I = I(:,:,1);
% [H,W] = size(I);  
% %
% I0 = I(H/2+1:H , W/2+1:W);
% I45 = I(1:H/2, W/2+1:W);
% I90 = I(1:H/2, 1:W/2); 
%%

%% =========================3通道图=========================
I0 = imread('0.png');   % 0度图像
I45 = imread('45.png');  %  45度图像
I90 = imread('90.png');   %  90度图像

%I0 = imresize(I0,[480,640]); I45 = imresize(I45,[480,640]);  I90 = imresize(I90,[480,640]);

tic;
I0 = rgb2gray(I0);  I45 = rgb2gray(I45);   I90 = rgb2gray(I90);   % lucid相机采集的为三通道灰度图，转化为单通道图
%I0 =  double(I0)/255  ;  I45 =  double(I45)/255;  I90 =  double(I90)/255;

%I = I0 + I90;   % stokes向量
I = imread('fog20.jpg');   %也可以放入偏振相机拍到的总图
%I_max = max(max(I));  % 看最大亮度，避免有曝光的图像
Q = I0 - I90;
U = 2*I45-I;
%%

%% =============================偏振差分================================
DoP = (Q.^2 + U.^2 ).^0.5./I;   %偏振度
AoP = atan(U./Q)/2;  %偏振角

Imax = (1+DoP).*I/2;     % 最大光强图
Imin = (1-DoP).*I/2;     % 最小光强图
%Imin = (Imin-min(Imin(:)))./(max(Imin(:))-min(Imin(:)));  %天大论文方法：直方图标准化，效果一般
%Imax = ((1+DoP)./(1-DoP)).*Imin;
I_delta = Imax - Imin;   % 差分图
%%

%% ====================求大气光偏振度PA，法1：天空区域法，适用于一般雾况=======================
binary = im2bw(I, graythresh(I));         % ostu二值化map
%binary(800:1024,:)= 0;         % 若雾极浓，天空区域难以区分，可直接手动确定天空区域
se = strel('disk',9);           % 腐蚀半径9
binary = imerode(binary,se);       % 腐蚀运算
sky_area = immultiply(I,binary);   %获取天空区域

PA = immultiply(DoP,binary);   %求大气光偏振度
%PA_area(isnan(PA_area)) = 0;
PA = mean(PA(PA ~= 0))*2.5;  %修正系数
%PA = 0.3;   % 试验PA
%%

%% ======================求大气光偏振度PA，法2：Liang法，适用于极端浓雾========================
[AoP_A,F] = mode(AoP(:));   % AoP中出现最多的数为大气偏振角,F为次数
PA = max(DoP(AoP == AoP_A));   % 从DoP对应元素中选最大值作为大气光偏振度PA
%%

%% ======================== 求无穷远大气光强A∞，法1：亮通道法===================
sky = sky_area(sky_area~=0);     % 先将非零元素去掉
sky = (sort(sky,'descend'));
num = (fix(length(sky)*0.001));
A_inf = mean(sky(1:num));      % A∞为最天空区域0.01%最亮像素的平均值

sky_max_value = max(sky);  % 查看天空区域中是否存在曝光区域
if(sky_max_value)>1
    A_inf = 0.95;   % 若存在曝光，A∞手动设定为0.95
end
%%

%% =================== 求无穷远大气光强A∞，法2：Liang法，适用于上面法2，效果不佳======================
A_prz = (I90 - (1-DoP).*I/2)/((sin(AoP_A))^2);  % 大气光强的偏振部分
A = A_prz./PA;      % 参考Liang论文
A_inf = 2*I0./(1+PA*cos(2*AoP_A));
mat = A_inf ./ I ;
A_inf = mean(mat((mat-1)<0.05));
ones = ones(size(I));
result = (I90 - A)./(ones-A/A_inf);
%%

%% =============== 大气光A含大量噪声，引入导向滤波 =============
win_size = 15;    % 导向滤波参数
r = win_size*4;  % 窗口半径
eps = 10^-6;     % 正则化参数

A = I_delta./PA;   % 求大气光A
A = guidedfilter(A,I90, r,eps);   %以I90为引导图
%%

%% ================ 按照大气物理模型恢复 ======================
t = 1 - 0.95*A/A_inf; 
%result = (I - A)./t; 
result = dehaze(I);
imshow([I result]);

%%
toc;