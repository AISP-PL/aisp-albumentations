#!/usr/bin/python3
import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from helpers.annotations import ReadAnnotations
from helpers.augumentations import (
    Augment,
    transform_all,
    transform_blackboxing_make,
    transform_blur_delicate_make,
    transform_blur_make,
    transform_brighten_make,
    transform_clahe_make,
    transform_color,
    transform_colorshift_make,
    transform_compression_make,
    transform_crop_make,
    transform_darken_make,
    transform_degrade_make,
    transform_downsize_padding_make,
    transform_equalize_make,
    transform_flip_horizontal_make,
    transform_flip_make,
    transform_flip_vertical_make,
    transform_fog_make,
    transform_gaussnoise_make,
    transform_isonoise_make,
    transform_median_blur_make,
    transform_multinoise_make,
    transform_rain_make,
    transform_randrotate_make,
    transform_rotate_make,
    transform_shape,
    transform_sharpen_make,
    transform_snow_make,
    transform_spatter_big_make,
    transform_spatter_make,
    transform_spatter_small_make,
    transform_sunflare_make,
)
from helpers.files import IsImageFile
from helpers.scene_matching import transform_night_make


def GetImages(path: str) -> list:
    """Gets all images from directory as random list."""
    # List of excluded files
    excludes = [".", "..", "./", ".directory"]

    # Files : Filter only images
    filenames = [
        os.path.join(path, filename)
        for filename in os.listdir(path)
        if (filename not in excludes) and (IsImageFile(filename))
    ]

    return filenames


def Process(path: str, arguments: argparse.Namespace):
    """Process directory"""
    # Check : Path is None or empty
    if (path is None) or (path == ""):
        logging.error("Path is None or empty!")
        return

    # Generated : Create output directory
    outputPath = os.path.join(path, "generated")
    Path(outputPath).mkdir(parents=True, exist_ok=True)

    # Images : Get all images from directory
    images = GetImages(path)
    if len(images) == 0:
        logging.error("No images found in directory!")
        return

    # Random : Shuffle all images for randomization
    random.shuffle(images)

    # Counter : Of processed images
    processed_counter = 0
    # Preview: ProgressBar : Create
    progress = tqdm(total=args.iterations, desc="Augumentation", unit="images")
    # Step 1 - augment current images and make new
    for imagePath in images:
        # Created path : None
        created_path: Optional[str] = None
        # Annotations : Ready annotations if exists
        annotations = ReadAnnotations(imagePath)

        # Check : Continue if not all images and not annotated.
        if (arguments.all is False) and (annotations.exists is False):
            logging.warning(
                f"Annotations not found for {imagePath}! Please provide annotations first or add --all !"
            )
            continue

        use_sha = not arguments.disable_sha

        # Crop :
        if arguments.crop != 0:
            try:
                created_path = Augment(
                    imagePath,
                    outputPath,
                    annotations,
                    transform_crop_make(arguments.crop),
                    transformation_name="crop",
                    use_sha=use_sha,
                )
            except Exception as e:
                logging.error(f"Cropping image failed: {e}")

        # Rotate :
        if arguments.rotate != 0:
            try:
                created_path = Augment(
                    imagePath,
                    outputPath,
                    annotations,
                    transform_rotate_make(arguments.rotate),
                    transformation_name="rotate",
                    use_sha=use_sha,
                )
            except Exception as e:
                logging.error(f"Rotating image failed: {e}")

        # RandRotate :
        if arguments.randrotate != 0:
            try:
                created_path = Augment(
                    imagePath,
                    outputPath,
                    annotations,
                    transform_randrotate_make(arguments.randrotate),
                    transformation_name="randrotate",
                    use_sha=use_sha,
                )
            except Exception as e:
                logging.error(f"Rotating image failed: {e}")

        # Flip Horizontal : Image
        if arguments.flip_horizontal:
            try:
                created_path = Augment(
                    imagePath,
                    outputPath,
                    annotations,
                    transform_flip_horizontal_make(),
                    transformation_name="flip_horizontal",
                    use_sha=use_sha,
                )
            except Exception as e:
                logging.error(f"Flipping image failed: {e}")

        # Flip Vertical : Image
        if arguments.flip_vertical:
            try:
                created_path = Augment(
                    imagePath,
                    outputPath,
                    annotations,
                    transform_flip_vertical_make(),
                    transformation_name="flip_vertical",
                    use_sha=use_sha,
                )
            except Exception as e:
                logging.error(f"Flipping image failed: {e}")

        # Flip : Image
        if arguments.flip:
            try:
                created_path = Augment(
                    imagePath,
                    outputPath,
                    annotations,
                    transform_flip_make(),
                    transformation_name="flip",
                    use_sha=use_sha,
                )
            except Exception as e:
                logging.error(f"Flipping image failed: {e}")

        # Color shift : Image
        if arguments.colorshift:
            try:
                created_path = Augment(
                    imagePath,
                    outputPath,
                    annotations,
                    transform_colorshift_make(),
                    transformation_name="colorshift",
                    use_sha=use_sha,
                )
            except Exception as e:
                logging.error(f"Color shifting image failed: {e}")

        # CLAHE : Image
        if arguments.clahe:
            try:
                created_path = Augment(
                    imagePath,
                    outputPath,
                    annotations,
                    transform_clahe_make(),
                    transformation_name="clahe",
                    use_sha=use_sha,
                )
            except Exception as e:
                logging.error(f"CLAHE image failed: {e}")

        # Equalize : Image
        if arguments.equalize:
            try:
                created_path = Augment(
                    imagePath,
                    outputPath,
                    annotations,
                    transform_equalize_make(),
                    transformation_name="equalize",
                    use_sha=use_sha,
                )
            except Exception as e:
                logging.error(f"Equalizing image failed: {e}")

        # Sharpen : Image
        if arguments.sharpen:
            try:
                created_path = Augment(
                    imagePath,
                    outputPath,
                    annotations,
                    transform_sharpen_make(),
                    transformation_name="sharpen",
                    use_sha=use_sha,
                )
            except Exception as e:
                logging.error(f"Sharpening image failed: {e}")

        # Darken : Image
        if arguments.darken:
            try:
                created_path = Augment(
                    imagePath,
                    outputPath,
                    annotations,
                    transform_darken_make(),
                    transformation_name="darken",
                    use_sha=use_sha,
                )
            except Exception as e:
                logging.error(f"Darkening image failed: {e}")

        # Brighten : Image
        if arguments.brighten:
            try:
                created_path = Augment(
                    imagePath,
                    outputPath,
                    annotations,
                    transform_brighten_make(),
                    transformation_name="brighten",
                    use_sha=use_sha,
                )
            except Exception as e:
                logging.error(f"Brightening image failed: {e}")

        # IsoNoise : Image
        if arguments.isonoise:
            try:
                created_path = Augment(
                    imagePath,
                    outputPath,
                    annotations,
                    transform_isonoise_make(),
                    transformation_name="isonoise",
                    use_sha=use_sha,
                )
            except Exception as e:
                logging.error(f"Adding iso noise to image failed: {e}")

        # GaussNoise : Image
        if arguments.gaussnoise:
            try:
                created_path = Augment(
                    imagePath,
                    outputPath,
                    annotations,
                    transform_gaussnoise_make(),
                    transformation_name="gaussnoise",
                    use_sha=use_sha,
                )
            except Exception as e:
                logging.error(f"Adding gauss noise to image failed: {e}")

        # MultiNoise : Image
        if arguments.multi_noise:
            try:
                created_path = Augment(
                    imagePath,
                    outputPath,
                    annotations,
                    transform_multinoise_make(),
                    transformation_name="multi_noise",
                    use_sha=use_sha,
                )
            except Exception as e:
                logging.error(f"Adding multi gauss noise to image failed: {e}")

        # compression : Quality
        if arguments.compression:
            try:
                created_path = Augment(
                    imagePath,
                    outputPath,
                    annotations,
                    transform_compression_make(),
                    transformation_name="compression",
                    use_sha=use_sha,
                )
            except Exception as e:
                logging.error(f"Degrading image failed: {e}")

        # Downsize padding : Image
        if arguments.downsize_padding:
            try:
                created_path = Augment(
                    imagePath,
                    outputPath,
                    annotations,
                    transform_downsize_padding_make(),
                    transformation_name="downsize_padding",
                    use_sha=use_sha,
                )
            except Exception as e:
                logging.error(f"Degrading image failed: {e}")

        # Degrade : Image
        if arguments.degrade:
            try:
                created_path = Augment(
                    imagePath,
                    outputPath,
                    annotations,
                    transform_degrade_make(),
                    transformation_name="degrade",
                    use_sha=use_sha,
                )
            except Exception as e:
                logging.error(f"Degrading image failed: {e}")

        # Snow : Image
        if arguments.snow:
            try:
                created_path = Augment(
                    imagePath,
                    outputPath,
                    annotations,
                    transform_snow_make(),
                    transformation_name="snow",
                    use_sha=use_sha,
                )
            except Exception as e:
                logging.error(f"Snowing image failed: {e}")

        # Rain : Image
        if arguments.rain:
            try:
                created_path = Augment(
                    imagePath,
                    outputPath,
                    annotations,
                    transform_rain_make(),
                    transformation_name="rain",
                    use_sha=use_sha,
                )
            except Exception as e:
                logging.error(f"Raining image failed: {e}")

        # Fog : Image
        if arguments.fog:
            try:
                created_path = Augment(
                    imagePath,
                    outputPath,
                    annotations,
                    transform_fog_make(),
                    transformation_name="fog",
                    use_sha=use_sha,
                )
            except Exception as e:
                logging.error(f"Fogging image failed: {e}")

        # Spatter : Image
        if arguments.spatter:
            try:
                created_path = Augment(
                    imagePath,
                    outputPath,
                    annotations,
                    transform_spatter_make(),
                    transformation_name="spatter",
                    use_sha=use_sha,
                )
            except Exception as e:
                logging.error(f"Spattering image failed: {e}")

        # Spatter big : Image
        if arguments.spatter_big:
            try:
                created_path = Augment(
                    imagePath,
                    outputPath,
                    annotations,
                    transform_spatter_big_make(),
                    transformation_name="spatter_big",
                    use_sha=use_sha,
                )
            except Exception as e:
                logging.error(f"Spattering big image failed: {e}")

        # Spatter small : Image
        if arguments.spatter_small:
            try:
                created_path = Augment(
                    imagePath,
                    outputPath,
                    annotations,
                    transform_spatter_small_make(),
                    transformation_name="spatter_small",
                    use_sha=use_sha,
                )
            except Exception as e:
                logging.error(f"Spattering small image failed: {e}")

        # Blackboxing : Image
        if arguments.blackboxing:
            try:
                created_path = Augment(
                    imagePath,
                    outputPath,
                    annotations,
                    transform_blackboxing_make(size=arguments.blackboxing),
                    transformation_name="blackboxing",
                    is_bboxes_transform=False,
                    use_sha=use_sha,
                )
            except Exception as e:
                logging.error(f"Blackboxing image failed: {e}")

        # Sunflare : Image
        if arguments.sunflare:
            try:
                created_path = Augment(
                    imagePath,
                    outputPath,
                    annotations,
                    transform_sunflare_make(),
                    transformation_name="sunflare",
                    use_sha=use_sha,
                )
            except Exception as e:
                logging.error(f"Sunflaring image failed: {e}")

        # Blur : Image
        if arguments.blur:
            try:
                created_path = Augment(
                    imagePath,
                    outputPath,
                    annotations,
                    transform_blur_make(),
                    transformation_name="blur",
                    use_sha=use_sha,
                )
            except Exception as e:
                logging.error(f"Blurring image failed: {e}")

        # Blur delicate : Image
        if arguments.blur_delicate:
            try:
                created_path = Augment(
                    imagePath,
                    outputPath,
                    annotations,
                    transform_blur_delicate_make(),
                    transformation_name="blur_delicate",
                    use_sha=use_sha,
                )
            except Exception as e:
                logging.error(f"Delicate blurring image failed: {e}")

        # Median Blur : Image
        if arguments.medianblur:
            try:
                created_path = Augment(
                    imagePath,
                    outputPath,
                    annotations,
                    transform_median_blur_make(),
                    transformation_name="medianblur",
                    use_sha=use_sha,
                )
            except Exception as e:
                logging.error(f"Median blurring image failed: {e}")

        # Night : Image
        if arguments.night:
            try:
                created_path = Augment(
                    imagePath,
                    outputPath,
                    annotations,
                    transform_night_make(),
                    transformation_name="night",
                    use_sha=use_sha,
                )
            except Exception as e:
                logging.error(f"Night vision image failed: {e}")

        # Augmentate : Color
        if arguments.augumentColor:
            created_path = Augment(
                imagePath,
                outputPath,
                annotations,
                transform_color,
                transformation_name="augment_color",
                use_sha=use_sha,
            )
        # Augmentate : Shape
        elif arguments.augumentShape:
            created_path = Augment(
                imagePath,
                outputPath,
                annotations,
                transform_shape,
                transformation_name="augment_shape",
                use_sha=use_sha,
            )
        # Augmentate : All
        elif arguments.augumentAll:
            created_path = Augment(
                imagePath,
                outputPath,
                annotations,
                transform_all,
                transformation_name="augment_all",
                use_sha=use_sha,
            )

        # Check : Created path is None
        if created_path is None:
            continue

        # Logging : Created image
        logging.info(f"Created {created_path}!")

        # Counter : Increment
        processed_counter += 1
        progress.update(1)
        # Check : Maximum number of created images
        if processed_counter >= arguments.iterations:
            break


if __name__ == "__main__":
    # Logging : Enable
    if __debug__ is True:
        logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    else:
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    logging.debug("Logging enabled!")

    # Arguments and config
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Input path")
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        nargs="?",
        const=300,
        default=300,
        required=False,
        help="Maximum number of created images",
    )
    parser.add_argument(
        "--disable_sha",
        action="store_true",
        default=False,
        required=False,
        help="Disable SHA and append transformation type to filename",
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        required=False,
        help="All images (annotated and not annotated). Defaut is only annotated.",
    )
    parser.add_argument(
        "--flip_horizontal",
        action="store_true",
        required=False,
        help="Flip horizontal image.",
    )
    parser.add_argument(
        "--flip_vertical",
        action="store_true",
        required=False,
        help="Flip vertical image.",
    )
    parser.add_argument(
        "--crop",
        type=int,
        nargs="?",
        const=0,
        default=0,
        required=False,
        help="Augument by random Crop image (for ex 640).",
    )
    parser.add_argument(
        "--rotate",
        type=int,
        nargs="?",
        const=0,
        default=0,
        required=False,
        help="Augument by direct degrees rotation (for ex 90).",
    )
    parser.add_argument(
        "--randrotate",
        type=int,
        nargs="?",
        const=0,
        default=0,
        required=False,
        help="Random rotation from -degrees to degrees.",
    )
    parser.add_argument(
        "--brighten",
        action="store_true",
        required=False,
        help="Random make image brighten and adjust contrast.",
    )
    parser.add_argument(
        "--sharpen",
        action="store_true",
        required=False,
        help="Random make image sharpen.",
    )
    parser.add_argument(
        "--darken",
        action="store_true",
        required=False,
        help="Random make image darkne and adjust contrast.",
    )
    parser.add_argument(
        "--clahe", action="store_true", required=False, help="Apply CLAHE to image."
    )
    parser.add_argument(
        "--equalize",
        action="store_true",
        required=False,
        help="Equalize image.",
    )
    parser.add_argument(
        "--colorshift",
        action="store_true",
        required=False,
        help="Random color shift in image ",
    )
    parser.add_argument(
        "--isonoise",
        action="store_true",
        required=False,
        help="Random add iso noise to image.",
    )
    parser.add_argument(
        "--gaussnoise",
        action="store_true",
        required=False,
        help="Random add gauss noise.",
    )
    parser.add_argument(
        "--multi_noise",
        action="store_true",
        required=False,
        help="Random multi gauss noise.",
    )
    parser.add_argument(
        "--downsize_padding",
        action="store_true",
        required=False,
        help="Downsize with black padding.",
    )
    parser.add_argument(
        "--compression",
        action="store_true",
        required=False,
        help="compression image quality.",
    )
    parser.add_argument(
        "--degrade",
        action="store_true",
        required=False,
        help="Degrade image quality.",
    )
    parser.add_argument(
        "--spatter", action="store_true", required=False, help="Spatter add."
    )
    parser.add_argument(
        "--spatter_big", action="store_true", required=False, help="Spatter add."
    )
    parser.add_argument(
        "--spatter_small", action="store_true", required=False, help="Spatter add."
    )
    parser.add_argument(
        "--blackboxing",
        required=False,
        type=int,
        nargs="?",
        const=0,
        default=0,
        help="Blackboxing HxH parts of image.",
    )
    parser.add_argument("--snow", action="store_true", required=False, help="Snow add.")
    parser.add_argument(
        "--night",
        action="store_true",
        required=False,
        help="Transform to night vision.",
    )
    parser.add_argument("--rain", action="store_true", required=False, help="Rain add.")
    parser.add_argument("--fog", action="store_true", required=False, help="Fog add.")
    parser.add_argument(
        "--sunflare", action="store_true", required=False, help="Sunflare add."
    )
    parser.add_argument(
        "--blur", action="store_true", required=False, help="Blur image."
    )
    parser.add_argument(
        "--blur_delicate",
        action="store_true",
        required=False,
        help="Blur delicate image.",
    )
    parser.add_argument(
        "--flip", action="store_true", required=False, help="Flip randomly image."
    )
    parser.add_argument(
        "-mb",
        "--medianblur",
        action="store_true",
        required=False,
        help="Median blur image.",
    )
    parser.add_argument(
        "-aa",
        "--augumentAll",
        action="store_true",
        required=False,
        help="All image augmentations.",
    )
    parser.add_argument(
        "-as",
        "--augumentShape",
        action="store_true",
        required=False,
        help="Process extra image shape augmentation.",
    )
    parser.add_argument(
        "-ac",
        "--augumentColor",
        action="store_true",
        required=False,
        help="Process extra image color augmentation.",
    )
    args = parser.parse_args()

    # Process
    Process(args.input, args)
