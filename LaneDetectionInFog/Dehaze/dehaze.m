function out = dehaze(img_hazy)
img_hazy = imread('fog20.jpg');

img_hazy = imresize(img_hazy,[480, 640]);
I = rgb2gray(img_hazy);
I = double(I)/255.0;

map(:,:,1)=I;
map(:,:,2)=I;
map(:,:,3)=I;
hazy = map;
gamma = 1; % 修正系数

%=======估算大气光，偏振对雾况敏感，若其估算的大气光图不准确，可以辅助计算=======
Atom = estimate_airlight(im2double(hazy).^(gamma));  
A = reshape(Atom,1,1,3); 

[img_dehazed, trans_refined] = dehazing(hazy, A, gamma);
img_dehazed = rgb2gray(img_dehazed);

%img_dehazed = double(img_dehazed)/255;
img_hazy = double(hazy)/255.0;

%subplot(1,2,1); imshow(img_hazy);    title('Hazy input')
figure; imshow(img_dehazed); title('De-hazed output')
%subplot(1,2,2); imshow(trans_refined); colormap('jet'); title('Transmission')
