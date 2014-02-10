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

class AnnotatedImage:
    """
    All the information regarding an image.
    The public fields are:
    - image_jpeg: array of bytes.
                  An array of bytes containing the image encoded in JPEG.
                  Use the methods set/get_image to set/get this field.
    - image_name: string
                  The unique file identifier of the image
                  (i.e. 'val/ILSVRC2012_val_00000001.JPEG')
    - gt_objects: array of AnnotatedObject objects
    - pred_objects: array of AnnotatedObject objects
    - crop_description: string, containing a description regarding how the image 
                        has been generated from its original version
    - segmentation_name: string, denoting the unique name of the segmentation
                         mask used for this image.
    """
    def __init__(self):
        self.image_jpeg = ''
        self.image_name = ''
        self.gt_objects = []
        self.pred_objects = []
        self.crop_description = ''
        self.segmentation_name = ''
        
    def set_image(self, img):
        """
        Set the image, given a ndarray-image
        """
        self.image_jpeg = util.convert_image_to_jpeg_string(img)

    def get_image(self):
        """
        Return a ndarray-image
        """
        return util.convert_jpeg_string_to_image(self.image_jpeg)



