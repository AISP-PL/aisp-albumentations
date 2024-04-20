"""
    Albumentations reference scene or image matching
"""

import os

import albumentations as A
import cv2

# python : Get Scirpt directory
path = os.path.dirname(os.path.realpath(__file__))
# Load images
image_night = cv2.imread(f"{path}/../data/night.jpg")
image_day = cv2.imread(f"{path}/../data/day.jpg")


def transform_night_make() -> A.Compose:
    """Create night transformation."""
    return A.Compose(
        [
            A.PixelDistributionAdaptation(
                reference_images=[image_night],
                read_fn=lambda x: x,
                p=1,
                transform_type="pca",
            )
        ],
        bbox_params=A.BboxParams(format="yolo", min_area=100, min_visibility=0.3),
    )


def transform_storm_make() -> A.Compose:
    """Create night transformation."""
    return A.Compose(
        [
            A.FDA(
                reference_images=[image_night],
                read_fn=lambda x: x,
                p=1,
                beta_limit=(0.1, 0.1),
            )
        ],
        bbox_params=A.BboxParams(format="yolo", min_area=100, min_visibility=0.3),
    )


# def transform_night_make() -> A.Compose:
#     """Create night transformation."""
#     return A.Compose(
#         [
#             A.HistogramMatching(
#                 reference_images=[image_night],
#                 read_fn=lambda x: x,
#                 blend_ratio=(0.8, 0.8),
#                 p=1.0,
#             )
#         ],
#         bbox_params=A.BboxParams(format="yolo", min_area=100, min_visibility=0.3),
#     )
