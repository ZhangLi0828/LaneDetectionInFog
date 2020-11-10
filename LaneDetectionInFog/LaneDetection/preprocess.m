function [frame,area] = preprocess(frame)
frame = rgb2gray(frame);
frame = imresize(frame,[480,640]);
img = medfilt2(frame);
% se = strel('disk',3);        
% img = imerode(img,se);
[Height,Width] = size(frame);
area = edge(frame,'canny',0.12,0.6);
area(Height-30:Height,:)=0;
roi = 250;

area(1:roi,:)=0;  % 选取感兴趣面积            
area(roi:roi+60,1:150)=0; 
area(roi:roi+60,Width-150:Width)=0;
area(:,1:10)=0;
area(:,Width-10:Width)=0;
area = bwareaopen(area,50,8);  % 去小面积小点
