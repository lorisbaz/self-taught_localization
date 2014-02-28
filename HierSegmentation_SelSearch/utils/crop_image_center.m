function img_crop = crop_image_center(img)

    bb = get_center_crop(img);
    img_crop = img(bb.ymin:bb.ymax, bb.xmin:bb.xmax, :);

end
        
function bbox = get_center_crop(img)
    [height, width, ~] = size(img);

    if height < width
        offset = floor((width - height) / 2);
        bbox.xmin = offset + 1;
        bbox.ymin = 1;
        bbox.xmax = offset + height;
        bbox.ymax = height;
    else
        offset = floor((height - width) / 2);
        bbox.xmin = 1;
        bbox.ymin = offset + 1;
        bbox.xmax = width;
        bbox.ymax = offset + width;
    end
end