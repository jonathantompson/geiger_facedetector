close all; clearvars; clc;

% Load the kitten
x = single(rgb2gray(imread('kitten.jpg'))) / 255;
x = imresize(x, 1/4);
imshow(x)
w = size(x, 2);
h = size(x, 1);
patch_size = 26;
patch_stride = patch_size;

% Break image into patches (there is probably a faster way to do this
i = 1;
for v = 1:patch_stride:(h - patch_size)
  for u = 1:patch_stride:(w - patch_size)
    patches(i, :, :) = x(v:v+patch_size-1, u:u+patch_size-1);
    i = i + 1;
  end
end

% Perform whitening (from:
% http://ufldl.stanford.edu/wiki/index.php/Implementing_PCA/Whitening)
patches = reshape(patches, size(patches, 1), patch_size * patch_size);
k = 100;
avg = mean(patches, 1);
patches = patches - repmat(avg, size(patches, 1), 1);
sigma = patches * patches' / size(patches, 2);
[U,S,V] = svd(sigma);
xRot = U' * patches;          % rotated version of the data. 
xTilde = U(:,1:k)' * patches; % reduced dimension representation of the data, 
                        % where k is the number of eigenvectors to keep
epsilon = 1e-4;
xPCAwhite = diag(1./sqrt(diag(S) + epsilon)) * U' * patches;
xZCAwhite = U * diag(1./sqrt(diag(S) + epsilon)) * U' * patches;

% Reconstruct the whitened image
i = 1;
for v = 1:patch_stride:(h - patch_size)
  for u = 1:patch_stride:(w - patch_size)
    patch = reshape(xZCAwhite(i,:), patch_size, patch_size);
    x_white(v:v+patch_size-1, u:u+patch_size-1) = patch;
    i = i + 1;
  end
end
figure; imshow(x_white);