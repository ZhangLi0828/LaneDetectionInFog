function [img_dehazed, transmission] = dehazing(img_hazy, air_light, gamma)

%==================������ͼ����У��=================
[h,w,n_colors] = size(img_hazy);
if (n_colors ~= 3)  error(['���벻����ͨ��']); end  % ��ɫͼҲ��������ͨ���ĵ�ɫͼ
if ~exist('air_light','var') || isempty(air_light) || (numel(air_light)~=3)  error('������Ҳ����Ϊ��ͨ��');  end
if ~exist('gamma','var') || isempty(gamma), gamma = 1; end
img_hazy = im2double(img_hazy);
img_hazy_corrected = img_hazy.^gamma; % gamma = 1ʱû��

dist_from_airlight = double(zeros(h,w,n_colors));
for color_idx=1:n_colors
    dist_from_airlight(:,:,color_idx) = img_hazy_corrected(:,:,color_idx) - air_light(:,:,color_idx);
end

radius = sqrt( dist_from_airlight(:,:,1).^2 + dist_from_airlight(:,:,2).^2 +dist_from_airlight(:,:,3).^2 );

dist_unit_radius = reshape(dist_from_airlight,[h*w,n_colors]);
dist_norm = sqrt(sum(dist_unit_radius.^2,2));
dist_unit_radius = bsxfun(@rdivide, dist_unit_radius, dist_norm);
n_points = 1000;
fid = fopen(['TR',num2str(n_points),'.txt']);
points = cell2mat(textscan(fid,'%f %f %f')) ;
fclose(fid);
mdl = KDTreeSearcher(points);
ind = knnsearch(mdl, dist_unit_radius);


%===============����ͼ===============
K = accumarray(ind,radius(:),[n_points,1],@max);
radius_new = reshape( K(ind), h, w);
transmission_estimation = radius./radius_new;
%========�ο�He����͸�����޶���0.1=========
trans_min = 0.1; 
transmission_estimation = min(max(transmission_estimation, trans_min),1);

%=========�ο�non-local�������ڰ�ƽ�����Ż��˴���ͼ========
trans_lower_bound = 1 - min(bsxfun(@rdivide,img_hazy_corrected,reshape(air_light,1,1,3)) ,[],3);
transmission_estimation = max(transmission_estimation, trans_lower_bound);
bin_count       = accumarray(ind,1,[n_points,1]);
bin_count_map   = reshape(bin_count(ind),h,w);
bin_eval_fun    = @(x) min(1, x/50);
K_std = accumarray(ind,radius(:),[n_points,1],@std);
radius_std = reshape( K_std(ind), h, w);
radius_eval_fun = @(r) min(1, 3*max(0.001, r-0.1));
radius_reliability = radius_eval_fun(radius_std./max(radius_std(:)));
data_term_weight   = bin_eval_fun(bin_count_map).*radius_reliability;
lambda = 0.1;
transmission = wls_optimization(transmission_estimation, data_term_weight, img_hazy, lambda);


img_dehazed = zeros(h,w,n_colors);
leave_haze = 1.06;  %����һ���ľ���
for color_idx = 1:3  % ��������ģ�ͻָ�
    img_dehazed(:,:,color_idx) = ( img_hazy_corrected(:,:,color_idx) - ...
        (1-leave_haze.*transmission).*air_light(color_idx) )./ max(transmission,trans_min);
end

img_dehazed(img_dehazed>1) = 1;    %���ع�
img_dehazed(img_dehazed<0) = 0;    
img_dehazed = img_dehazed.^(1/gamma); %gamma = 1ʱû��

adj_percent = [0.005, 0.995];
img_dehazed = adjust(img_dehazed,adj_percent);  %�Ż���ͼ��ĶԱȶ�
img_dehazed = im2uint8(img_dehazed);

end 
