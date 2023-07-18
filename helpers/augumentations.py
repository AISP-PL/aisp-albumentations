import os
import albumentations as A
import cv2

from helpers.annotations import Annotations, SaveAnnotations
from helpers.files import ChangeExtension
from helpers.hashing import GetRandomSha1


# Shape : Albumentations transform
transform_shape = A.Compose([
    A.RandomCrop(width=640, height=352, p=0.3),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                       rotate_limit=15, p=0.7),
    A.ElasticTransform(alpha_affine=9, p=0.2),
    A.OpticalDistortion(p=0.2),
    A.ZoomBlur(max_factor=1.1, p=0.2),
], bbox_params=A.BboxParams(format='yolo', min_area=100, min_visibility=0.2))

# Color : Albumentations transform
transform_color = A.Compose([
    A.JpegCompression(quality_lower=20, quality_upper=55, p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.RandomContrast(p=0.3),
    A.RandomToneCurve(scale=0.5, p=0.3),
    A.HueSaturationValue(p=0.3),
    A.ColorJitter(p=0.2),
    A.MultiplicativeNoise(p=0.2),
    A.Downscale(scale_min=0.3, scale_max=0.6, p=0.2),
    A.MedianBlur(blur_limit=5, p=0.1),
    A.ISONoise(color_shift=(0.01, 0.08), intensity=(0.2, 0.8), p=0.1),
    A.PixelDropout(dropout_prob=0.1, p=0.1),
    A.RandomRain(drop_length=10, blur_value=4, p=0.1),
    A.Spatter(p=0.1),
    A.RandomSnow(p=0.1),
    A.RandomSunFlare(p=0.1),
    A.RandomFog(p=0.1),
], bbox_params=A.BboxParams(format='yolo', min_area=100, min_visibility=0.2))

# All : Full transform
transform_all = A.Compose([
    A.SomeOf([transform_color], n=3, p=0.5),
    A.SomeOf([transform_shape], n=3, p=0.5),
], bbox_params=A.BboxParams(format='yolo', min_area=100, min_visibility=0.2))


def Augment(imagePath: str,
            outputDirectory: str,
            annotations: Annotations,
            transformations) -> str:
    ''' Read image, augment image and bboxes and save it to new file. '''

    # Read image
    image = cv2.imread(imagePath)

    # Augmentate image
    transformed = transformations(image=image, bboxes=annotations.annotations)

    # Annotations : Create new
    newAnnotations = Annotations(
        imagePath, dataformat=annotations.dataformat, annotations=transformed['bboxes'])

    # Create filename
    outputFilepath = os.path.join(outputDirectory, f'{GetRandomSha1()}.jpeg')

    # Image : Save
    cv2.imwrite(outputFilepath, transformed['image'])
    # Annotations : Save
    SaveAnnotations(ChangeExtension(outputFilepath, '.txt'), newAnnotations)

    return outputFilepath
