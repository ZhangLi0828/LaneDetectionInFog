function out = dehaze(img_hazy)
img_hazy = imread('fog20.jpg');

img_hazy = imresize(img_hazy,[480, 640]);
I = rgb2gray(img_hazy);
I = double(I)/255.0;

map(:,:,1)=I;
map(:,:,2)=I;
map(:,:,3)=I;
hazy = map;
gamma = 1; % ����ϵ��

%=======��������⣬ƫ���������У��������Ĵ�����ͼ��׼ȷ�����Ը�������=======
Atom = estimate_airlight(im2double(hazy).^(gamma));  
A = reshape(Atom,1,1,3); 

[img_dehazed, trans_refined] = dehazing(hazy, A, gamma);
img_dehazed = rgb2gray(img_dehazed);

%img_dehazed = double(img_dehazed)/255;
img_hazy = double(hazy)/255.0;

%subplot(1,2,1); imshow(img_hazy);    title('Hazy input')
figure; imshow(img_dehazed); title('De-hazed output')
%subplot(1,2,2); imshow(trans_refined); colormap('jet'); title('Transmission')
