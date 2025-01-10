clear all
clc
F=[];
for i =1:50
    str=['image/img',num2str(i),'.jpg'];
    I=imread(str);
    mysize=size(I);
    if mysize(end)==3
        I=rgb2gray(I);
    elseif mysize(end)==4
        I=I(:,:,1:3);
        I=rgb2gray(I);
    end
    
    F_tmp=double(I)/255;%
    F_tmp=F_tmp(:);
    F=[F F_tmp];
end

run ./gspbox/gsp_start
G=gsp_2dgrid(size(I,1));
A=full(G.W);
A=A+A';
A(A>0)=1;

mask=zeros(size(I,1),size(I,1));
mask(3:end-2,3:end-2)=1;
mask=mask(:);

save 2Dgrid A F mask

subplot(2,2,1);imagesc(reshape(F(:,6),[100 100]));axis equal
title('given images');
subplot(2,2,2);imagesc(reshape(F(:,20),[100 100]));axis equal
title('given images');
subplot(2,2,3);imagesc(reshape(F(:,30),[100 100]));axis equal
title('band-pass images');
subplot(2,2,4);imagesc(reshape(F(:,40),[100 100]));axis equal
title('low-pass images');

% High-pass filter
F_highpass = F - smooth(F, 0.1); % 使用一个低通滤波器平滑图像，然后从原始图像中减去这个平滑版本

% Band-rejection filter
F_bandreject = F;
% 在频域中选择需要抑制的频率范围，将其值设为0
% 例如，假设需要抑制频率范围为[low, high]的信号
F_bandreject(low:high) = 0;


% Comb filter
% 设计一个周期性的频率响应，根据实际需求调整参数
% 例如，增强特定频率分量
F_comb = F + sin(2*pi*f*t); % 根据实际情况构造一个周期性响应，f为频率，t为时间

% High-pass filter
D = spdiags(sum(A,1)', 0, size(A,1), size(A,1));
L = D - A;
[~, S, V] = svd(F', 'econ');
k_high = 20; % Set the number of high-frequency components to keep
F_high = F - V(:,1:k_high)*(V(:,1:k_high)'*F);

% Band-rejection filter
k_bandreject_low = 5; % Set the number of low-frequency components to keep
k_bandreject_high = 30; % Set the number of high-frequency components to keep
F_bandreject = V(:,k_bandreject_low+1:k_bandreject_high)*(V(:,k_bandreject_low+1:k_bandreject_high)'*F);

% Comb filter
delta = 1e-6; % Small positive value to avoid division by zero
D_inv_sqrt = spdiags(1./sqrt(sum(A,1)' + delta), 0, size(A,1), size(A,1));
L_sym = D_inv_sqrt * L * D_inv_sqrt;
theta = pi/4; % Set the angle parameter for the comb filter
F_comb = cos(theta)*F + sin(theta)*(L_sym*F);

% Display filtered images
figure;
subplot(2,2,1);imagesc(reshape(F(:,6),[100 100]));axis equal
title('given images');
subplot(2,2,2);imagesc(reshape(F(:,20),[100 100]));axis equal
title('given images');
subplot(2,2,3);imagesc(reshape(F_high(:,30),[100 100]));axis equal
title('high-pass images');
subplot(2,2,4);imagesc(reshape(F_bandreject(:,40),[100 100]));axis equal
title('band-rejection images');

% Display comb-filtered image
figure;
imagesc(reshape(F_comb(:,40),[100 100]));axis equal
title('comb-filtered image');
