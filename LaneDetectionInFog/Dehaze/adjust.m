function adj = adjust(img,percen)
%对比度优化图像，将对比度拉伸到[0,1]，保证一定的视觉效果

if ~exist('percen','var') || isempty(percen), percen=[0.01 0.99]; end;

minn=min(img(:));
img=img-minn;
img=img./max(img(:));
contrast_limit = stretchlim(img,percen);  %matlab专门提高图片的对比度的函数,而不关心参数高低的处理
val = 0.2;
contrast_limit(2,:) = max(contrast_limit(2,:), 0.2);
contrast_limit(2,:) = val*contrast_limit(2,:) + (1-val)*max(contrast_limit(2,:), mean(contrast_limit(2,:)));
contrast_limit(1,:) = val*contrast_limit(1,:) + (1-val)*min(contrast_limit(1,:), mean(contrast_limit(1,:)));
adj=imadjust(img,contrast_limit,[],1); % imajust：matlab专门调节灰度图像的亮度的函数
