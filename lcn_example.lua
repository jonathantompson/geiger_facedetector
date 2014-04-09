require 'image'
require 'nn'

img = image.load('kitten.jpg')
-- img = img:sum(1)
img = image.scale(img, img:size(3)/4, img:size(2)/4)
image.display{image=img, legend='input'}
nchan = img:size(1)
w = img:size(3)
h = img:size(2)

-- define a normalization kernel
kernel = image.gaussian1D(13)

-- Perform subtractive normalization
sub = nn.SpatialSubtractiveNormalization(nchan, kernel)
out_sub = sub:forward(img)
image.display{image=out_sub, legend='subtractive norm output'}

-- Perform subtractive normalization
div = nn.SpatialDivisiveNormalization(nchan, kernel)
out_div = div:forward(out_sub)
image.display{image=out_div, legend='divisive norm output'}

