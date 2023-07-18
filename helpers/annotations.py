
from dataclasses import dataclass, field
from enum import Enum
import os
from helpers.files import ChangeExtension

class AnnotationsFormat(str, Enum):
    ''' Enum for annotations format '''
    YOLO = 'yolo'
    COCO = 'coco'
    PascalVOC = 'pascal_voc'
    Albumentations = 'albumentations'

@dataclass
class Annotations:
    ''' Dataclass storing image file annotations'''
    # Image file path
    imagePath : str = None
    # Format of annotations
    dataformat : AnnotationsFormat = AnnotationsFormat.YOLO
    # List of annotations
    annotations : list = field(init=False, default_factory=list)


    def Append(self, annotation : tuple):
        ''' Append annotation to list '''
        self.annotations.append(annotation)

    @property
    def count(self) -> int:
        ''' Count of annotations '''
        return len(self.annotations)


def ReadAnnotations(imagePath : str) -> list:
    '''Read annotations from file.'''
    annotations = Annotations(imagePath)

    # YOLO format : <object-class> <x> <y> <width> <height>
    if (os.path.exists(ChangeExtension(imagePath,'.txt'))):
        # Format : Set
        annotations.dataformat = AnnotationsFormat.YOLO

        # File : Open file
        with open(ChangeExtension(imagePath,'.txt'), 'r') as f:
            for line in f:
                txtAnnote = (line.rstrip('\n').split(' '))
                classNumber = int(txtAnnote[0])
                box = (float(txtAnnote[1]), float(txtAnnote[2]),
                       float(txtAnnote[3]), float(txtAnnote[4]))
                
                annotations.Append((box) + (classNumber,))

    return annotations


def SaveAnnotations(filepath:str, annotations : Annotations):
    ''' Save annotations to file '''
    # YOLO format : <object-class> <x> <y> <width> <height>
    if (annotations.dataformat == AnnotationsFormat.YOLO):
        # File : Open file
        with open(filepath, 'w') as f:
            for annotation in annotations.annotations:
                # Format : <object-class> <x> <y> <width> <height>
                f.write(f"{annotation[4]} {annotation[0]} {annotation[1]} {annotation[2]} {annotation[3]}\n")

    return