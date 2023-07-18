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
], bbox_params=A.BboxParams(format='yolo', min_area=100, min_visibility=0.2))

# Color : Albumentations transform
transform_color = A.Compose([
    A.RandomBrightnessContrast(p=0.3),
    A.RandomToneCurve(scale=0.5, p=0.3),
    A.HueSaturationValue(p=0.3),
    A.CoarseDropout(p=0.3),
    A.ColorJitter(p=0.2),
], bbox_params=A.BboxParams(format='yolo', min_area=100, min_visibility=0.2))


def Augment(imagePath: str,
            outputDirectory: str,
            annotations: Annotations,
            transformations):
    ''' Read image, augment image and bboxes and save it to new file. '''

    # Read image
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
