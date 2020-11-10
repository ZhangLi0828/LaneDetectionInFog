function [agl_l,agl_r,r1,r2] =  LaneDetect(frame,i,area)
%frame = imread('D:\PycharmProjects\original_frames\frame13.jpg');
[Height,Width] = size(frame);
lim = 300;   %画线区域

%%
%====================绘制霍夫空间=========================
% imshow(imadjust(mat2gray(H)),[],'XData',theta,'YData',rho,...
%         'InitialMagnification','fit');
% xlabel('\theta (degrees)'), ylabel('\rho');
% axis on, axis normal, hold on;
% colormap(hot); 
% title('霍夫空间')
%峰值
%%
area_l = area;
area_r = area;
area_l(:,Width/2:Width)=0;  % 感兴趣区域左半部分，识别左车道线
area_r(:,1:Width/2)=0;   % 感兴趣区域右半部分，识别右车道线

%%
%%=================================左车道线===================================
[H, theta , rho] = hough (area_l,'Theta',30:0.01:70);

P = houghpeaks(H,4,'threshold',0.5*max(H(:)));
[agl_l,index] = min(theta(P(:,2)));   % 求最小theta,和其在P数组中的坐标
rou = rho(P(:,1));   % 所有的极径
r1 = rou(index); % 最小的极径

x_l1 = (r1-lim*sind(agl_l))/cosd(agl_l);%  为延长识别出的直线,与纵轴的交点
x_l2 = (r1-Height*sind(agl_l))/cosd(agl_l);% 与感兴趣区域横轴的交点

%lines = houghlines(area_l,theta,rho,P,'FillGap',15,'MinLength',60);  %FillGap: 两线段距离小于一定值时，合并为一条
figure,imshow(frame);

%%

%%================================右车道线======================================
[H_r, theta_r , rho_r] = hough (area_r,'Theta',-70:0.01:-20);
P_r = houghpeaks(H_r,8,'threshold',0.5*max(H_r(:)));
[agl_r,index] = max(theta_r(P_r(:,2)));   % 求最大theta,和其在P数组中的坐标
% x_r = abs(x_t);
rou_r = rho_r(P_r(:,1));   % 所有的极径
r2 = rou_r(index); % 对应的极径

x_r1 = (r2-lim*sind(agl_r))/cosd(agl_r);%  为延长识别出的直线,与纵轴的交点
x_r2 = (r2-Height*sind(agl_r))/cosd(agl_r);% 与感兴趣区域横轴的交点
  
%lines = houghlines(area_r,theta_r,rho_r,P_r,'FillGap',10,'MinLength',10);
%imshow(area) ,hold on
line([x_l1,x_l2],[lim,Height],'LineWidth',5,'Color','green','Marker','o','Markersize',1.4);
line([x_r1,x_r2],[lim,Height],'LineWidth',5,'Color','green','Marker','o','Markersize',1.4);
set(gcf,'Position',[0,0,640,480]);
set(gca,'position',[0 0 1 1])
saveas(gca,['../lane_frames1/frame',num2str(i),'.bmp'],'bmp');
close;

imshow(imadjust(mat2gray(H)),[],'XData',theta,'YData',rho,...
        'InitialMagnification','fit');
xlabel('\theta (degrees)'), ylabel('\rho');
axis on, axis normal, hold on;
colormap(hot); 
title('霍夫空间')
end



%%
%count = 1;
% points = zeros(2,2);
% for k = 1:length(lines)
%    points(count,1) = lines(k).point1(1);
%    points(count,2) = lines(k).point1(2);
%    count =count +1;
%    points(count,1) = lines(k).point2(1);
%    points(count,2) = lines(k).point2(2);
%    count =count +1;
%    xy = [lines(k).point1; lines(k).point2];
%    plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');
%    % Plot beginnings and ends of lines
%    plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
%    plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');
% end
%imwrite(img,'area.jpg');
%%