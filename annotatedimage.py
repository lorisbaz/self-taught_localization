import util


class AnnotatedObject:
    """
    Everything regarding the annotation of an object.
    The public fields are:
    - label: string (i.e. 'n0001000')
    - bboxes: list of BBox objects
    - heatmaps: list of Heatmap objects
    """
    def __init__(self):
        self.label = ''
        self.bboxes = []
        self.heatmaps = []

    def __str__(self):
        out = 'label: ' + str(self.label) + '; '
        for bbox in self.bboxes:
            out += str(bbox) + ' '
        return out

class AnnotatedImage:
    """
    All the information regarding an image.
    The public fields are:
    - image_jpeg: array of bytes.
                  An array of bytes containing the image encoded in JPEG.
                  Use the methods set/get_image to set/get this field.
    - image_width, image_height: int. This MUST match the size of image_jpeg
    - image_name: string
                  The unique file identifier of the image
                  (i.e. 'val/ILSVRC2012_val_00000001.JPEG')
    - gt_label: full-image gt label
    - pred_label: full-image pred label
    - gt_objects: array of AnnotatedObject objects
    - pred_objects: array of AnnotatedObject objects
    - crop_description: string, containing a description regarding how the image 
                        has been generated from its original version
    - segmentation_name: string, denoting the unique name of the segmentation
                         mask used for this image.
    """
    def __init__(self):
        self.image_jpeg = ''
        self.image_width = 0
        self.image_height = 0
        self.image_name = ''
        self.gt_label = ''
        self.pred_label = ''
        self.gt_objects = []
        self.pred_objects = []
        self.crop_description = ''
        self.segmentation_name = ''

    def __str__(self):
        out = '{0}:{1} [{2} x {3}]\n'.format(self.image_name, self.gt_label, \
                                             self.image_height, \
                                             self.image_width)
        out += 'gt_objects:\n'
        for obj in self.gt_objects:
            out += '  ' + str(obj)
        return out
        
    def set_image(self, img):
        """
        Set the image, given a ndarray-image
        """
        self.image_jpeg = util.convert_image_to_jpeg_string(img)
        self.image_width = img.shape[1]
        self.image_height = img.shape[0]

    def get_image(self):
        """
        Return a ndarray-image
        """
        img = util.convert_jpeg_string_to_image(self.image_jpeg)
        assert self.image_width == img.shape[1]
        assert self.image_height == img.shape[0]
        return img




