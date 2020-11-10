function [agl_l,agl_r,r1,r2] =  LaneDetect(frame,i,area)
%frame = imread('D:\PycharmProjects\original_frames\frame13.jpg');
[Height,Width] = size(frame);
lim = 300;   %��������

%%
%====================���ƻ���ռ�=========================
% imshow(imadjust(mat2gray(H)),[],'XData',theta,'YData',rho,...
%         'InitialMagnification','fit');
% xlabel('\theta (degrees)'), ylabel('\rho');
% axis on, axis normal, hold on;
% colormap(hot); 
% title('����ռ�')
%��ֵ
%%
area_l = area;
area_r = area;
area_l(:,Width/2:Width)=0;  % ����Ȥ������벿�֣�ʶ���󳵵���
area_r(:,1:Width/2)=0;   % ����Ȥ�����Ұ벿�֣�ʶ���ҳ�����

%%
%%=================================�󳵵���===================================
[H, theta , rho] = hough (area_l,'Theta',30:0.01:70);

P = houghpeaks(H,4,'threshold',0.5*max(H(:)));
[agl_l,index] = min(theta(P(:,2)));   % ����Сtheta,������P�����е�����
rou = rho(P(:,1));   % ���еļ���
r1 = rou(index); % ��С�ļ���

x_l1 = (r1-lim*sind(agl_l))/cosd(agl_l);%  Ϊ�ӳ�ʶ�����ֱ��,������Ľ���
x_l2 = (r1-Height*sind(agl_l))/cosd(agl_l);% �����Ȥ�������Ľ���

%lines = houghlines(area_l,theta,rho,P,'FillGap',15,'MinLength',60);  %FillGap: ���߶ξ���С��һ��ֵʱ���ϲ�Ϊһ��
figure,imshow(frame);

%%

%%================================�ҳ�����======================================
[H_r, theta_r , rho_r] = hough (area_r,'Theta',-70:0.01:-20);
P_r = houghpeaks(H_r,8,'threshold',0.5*max(H_r(:)));
[agl_r,index] = max(theta_r(P_r(:,2)));   % �����theta,������P�����е�����
% x_r = abs(x_t);
rou_r = rho_r(P_r(:,1));   % ���еļ���
r2 = rou_r(index); % ��Ӧ�ļ���

x_r1 = (r2-lim*sind(agl_r))/cosd(agl_r);%  Ϊ�ӳ�ʶ�����ֱ��,������Ľ���
x_r2 = (r2-Height*sind(agl_r))/cosd(agl_r);% �����Ȥ�������Ľ���
  
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
title('����ռ�')
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