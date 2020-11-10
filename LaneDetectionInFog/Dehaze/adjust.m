function adj = adjust(img,percen)
%�Աȶ��Ż�ͼ�񣬽��Աȶ����쵽[0,1]����֤һ�����Ӿ�Ч��

if ~exist('percen','var') || isempty(percen), percen=[0.01 0.99]; end;

minn=min(img(:));
img=img-minn;
img=img./max(img(:));
contrast_limit = stretchlim(img,percen);  %matlabר�����ͼƬ�ĶԱȶȵĺ���,�������Ĳ����ߵ͵Ĵ���
val = 0.2;
contrast_limit(2,:) = max(contrast_limit(2,:), 0.2);
contrast_limit(2,:) = val*contrast_limit(2,:) + (1-val)*max(contrast_limit(2,:), mean(contrast_limit(2,:)));
contrast_limit(1,:) = val*contrast_limit(1,:) + (1-val)*min(contrast_limit(1,:), mean(contrast_limit(1,:)));
adj=imadjust(img,contrast_limit,[],1); % imajust��matlabר�ŵ��ڻҶ�ͼ������ȵĺ���
