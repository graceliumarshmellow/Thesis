%% Adding Gaussian Noise
figure(1)
im = im2double(imread('EL clean.png')); 
im = im(:,:,1);
% no. of photons / value
f = 1e8/sum(sum(im));
% value of each pixel
im_noise = poissrnd(f*im)/f;
im_uint = im2uint8(im_noise);
imshow(im_uint);
imwrite(im_uint, 'EL Poisson noisy.png')
