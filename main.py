#!/usr/bin/python3
import os
from pathlib import Path
import sys
import random
import argparse
import logging
from helpers.annotations import ReadAnnotations
from helpers.augumentations import Augment, transform_color
from helpers.files import IsImageFile


def GetImages(path : str) -> list:
    ''' Gets all images from directory as random list. '''
    # List of excluded files
    excludes = ['.', '..', './', '.directory']

    # Files : Filter only images
    filenames = [ f"{path}{filename}" for filename in os.listdir(path) 
                  if (filename not in excludes) and (IsImageFile(filename))]

    # Step 0.1 - random shuffle list (for big datasets it gives randomization)
    random.shuffle(filenames)

    return filenames


def Process(path : str, arguments : argparse.Namespace):
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


    # Step 1 - augment current images and make new
    for imagePath in images:
        # Annotations : Ready annotations if exists
        annotations = ReadAnnotations(imagePath)

        # Augmentate image
        if (arguments.augumentColor):
            Augment(imagePath, outputPath, annotations, transform_color)





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
    parser.add_argument('-ow', '--maxImageWidth', type=int, nargs='?', const=1280, default=1280,
                        required=False, help='Output image width / network width')
    parser.add_argument('-oh', '--maxImageHeight', type=int, nargs='?', const=1280, default=1280,
                        required=False, help='Output image height / network height')
    parser.add_argument('-as', '--augumentShape', action='store_true',
                        required=False, help='Process extra image shape augmentation.')
    parser.add_argument('-ac', '--augumentColor', action='store_true',
                        required=False, help='Process extra image color augmentation.')
    args = parser.parse_args()

    # Process
    Process(args.input, args)


