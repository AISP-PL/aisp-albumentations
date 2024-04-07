#!/usr/bin/python3
import argparse
import logging
import os
import random
import sys
from pathlib import Path

from tqdm import tqdm

from helpers.annotations import ReadAnnotations
from helpers.augumentations import (Augment, transform_all, transform_color,
                                    transform_shape)
from helpers.files import IsImageFile


def GetImages(path: str) -> list:
    ''' Gets all images from directory as random list. '''
    # List of excluded files
    excludes = ['.', '..', './', '.directory']

    # Files : Filter only images
    filenames = [os.path.join(path, filename) for filename in os.listdir(path)
                 if (filename not in excludes) and (IsImageFile(filename))]

    # Step 0.1 - random shuffle list (for big datasets it gives randomization)
    random.shuffle(filenames)

    return filenames


def Process(path: str, arguments: argparse.Namespace):
    ''' Process directory'''
    # Check : Path is None or empty
    if (path is None) or (path == ''):
        logging.error('Path is None or empty!')
        return

    # Generated : Create output directory
    outputPath = os.path.join(path, 'generated')
    Path(outputPath).mkdir(parents=True, exist_ok=True)

    # Images : Get all images from directory
    images = GetImages(path)

    # Counter : Of processed images
    processed_counter = 0
    # Preview: ProgressBar : Create
    progress = tqdm(total=args.iterations, desc='Augumentation', unit='images')
    # Step 1 - augment current images and make new
    for imagePath in images:
        # Annotations : Ready annotations if exists
        annotations = ReadAnnotations(imagePath)

        # Check : Continue if not all images and not annotated.
        if (arguments.all is False) and (annotations.exists is False):
            logging.warning(f'Annotations not found for {imagePath}! Please provide annotations first or add --all !')
            continue


        # Augmentate image
        if (arguments.augumentColor):
            createdPath = Augment(imagePath, outputPath,
                                  annotations, transform_color)
        elif (arguments.augumentShape):
            createdPath = Augment(imagePath, outputPath,
                                  annotations, transform_shape)
        else:
            createdPath = Augment(imagePath, outputPath,
                                  annotations, transform_all)

        # Check : Created path is None
        if (createdPath is None):
            continue

        # Logging : Created image
        logging.info(f'Created {createdPath}!')

        # Counter : Increment
        processed_counter += 1
        progress.update(1)
        # Check : Maximum number of created images
        if (processed_counter >= arguments.iterations):
            break


if (__name__ == '__main__'):
    # Logging : Enable
    if (__debug__ is True):
        logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    else:
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    logging.debug('Logging enabled!')

    # Arguments and config
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        required=True, help='Input path')
    parser.add_argument('-n', '--iterations', type=int, nargs='?', const=300, default=300,
                        required=False, help='Maximum number of created images')
    parser.add_argument('-a', '--all', action='store_true',
                        required=False, help='All images (annotated and not annotated). Defaut is only annotated.')
    parser.add_argument('-aa', '--augumentAll', action='store_true',
                        required=False, help='All image augmentations.')
    parser.add_argument('-as', '--augumentShape', action='store_true',
                        required=False, help='Process extra image shape augmentation.')
    parser.add_argument('-ac', '--augumentColor', action='store_true',
                        required=False, help='Process extra image color augmentation.')
    args = parser.parse_args()

    # Process
    Process(args.input, args)
