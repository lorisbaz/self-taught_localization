function img = resize_image_max_size(img, fix_sz)
% Return a resized version of the image, where the longest edge has 
% length 'fix_sz' pixels. The resizing mantains the proportion.
    
[H,W,ch] = size(img);
great_size = max(H,W);
if (great_size > fix_sz)
    proportion = fix_sz / great_size;
    width = floor(W * proportion);
    height = floor(H * proportion);
    img = imresize(img, [height, width], 'bilinear'); 
    % note: interpolation is the same of skimage resize in python
end