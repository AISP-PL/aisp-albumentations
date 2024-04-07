import logging
import os
from typing import Optional
import albumentations as A
import cv2

from helpers.annotations import Annotations, SaveAnnotations
from helpers.files import ChangeExtension
from helpers.hashing import GetRandomSha1


# Shape : Albumentations transform
transform_shape = A.Compose([
    A.SomeOf([
        A.ImageCompression(quality_lower=30, quality_upper=55, p=0.3),
        A.MotionBlur(blur_limit=7, p=0.3),
    ], n=2),
    A.GridDistortion(num_steps=3, distort_limit=0.25, p=0.2),
    A.RandomCrop(width=480, height=320, p=0.3),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15,
                       p=0.7, border_mode=cv2.BORDER_CONSTANT),
    A.ElasticTransform(alpha_affine=9, p=0.2, border_mode=cv2.BORDER_CONSTANT),
    A.OpticalDistortion(distort_limit=0.2, p=0.2,
                        border_mode=cv2.BORDER_CONSTANT),
    A.ZoomBlur(max_factor=1.1, p=0.2),
], bbox_params=A.BboxParams(format='yolo', min_area=100, min_visibility=0.3))

# Color : Albumentations transform
transform_color = A.Compose([
    # Colors : Brightnes, contrast, tone curve, hue, saturation, value
    A.SomeOf([
        A.RandomBrightnessContrast(p=0.3),
        A.RandomToneCurve(scale=0.5, p=0.3),
        A.HueSaturationValue(p=0.3),
        A.ColorJitter(p=0.2),
        A.Equalize(p=0.3),
        A.ToSepia(p=0.1),
    ], n=2),
    # Quality : Jpeg compression, multiplicative noise, downscale
    A.OneOf([
        A.ImageCompression(quality_lower=30, quality_upper=55, p=0.3),
        A.MultiplicativeNoise(p=0.2),
        A.Downscale(scale_min=0.4, scale_max=0.6, p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.ISONoise(color_shift=(0.01, 0.08), intensity=(0.2, 0.8), p=0.1),
        A.PixelDropout(dropout_prob=0.1, p=0.1),
        A.Spatter(p=0.1),
        A.Superpixels(p=0.1),
        A.GlassBlur(sigma=0.2, max_delta=2, iterations=1, p=0.1),
    ]),
    # Weather : Dropouts, rain, snow, sun flare, fog
    A.OneOf([
        A.RandomRain(drop_length=10, blur_value=4, p=0.1),
        A.RandomSnow(p=0.1),
        A.RandomSunFlare(src_radius=260, num_flare_circles_lower=2,
                         num_flare_circles_upper=6, p=0.1),
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.5,
                    alpha_coef=0.5, p=0.1),
    ]),
], bbox_params=A.BboxParams(format='yolo', min_area=100, min_visibility=0.2))

# All : Full transform
transform_all = A.Compose([
    A.SomeOf([transform_color], n=3, p=0.5),
    A.SomeOf([transform_shape], n=3, p=0.5),
], bbox_params=A.BboxParams(format='yolo', min_area=100, min_visibility=0.2))


def Augment(imagePath: str,
            outputDirectory: str,
            annotations: Annotations,
            transformations) -> Optional[str]:
    ''' Read image, augment image and bboxes and save it to new file. '''

    # Read image
    image = cv2.imread(imagePath)
    if image is None : 
        logging.error(f'Image not found: {imagePath}!')
        return None

    # Augmentate image
    transformed = transformations(image=image, bboxes=annotations.annotations)

    # Annotations : Create new
    newAnnotations = Annotations(
        imagePath, dataformat=annotations.dataformat, annotations=transformed['bboxes'])

    # Create filename
    outputFilepath = os.path.join(outputDirectory, f'{GetRandomSha1()}.jpeg')

    # Image : Save
    cv2.imwrite(outputFilepath, transformed['image'])
    # Annotations : Save only if original exists
    if annotations.exists:
        SaveAnnotations(ChangeExtension(outputFilepath, '.txt'), newAnnotations)

    return outputFilepath
