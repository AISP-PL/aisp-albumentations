'''
Created on 10 wrz 2020

@author: spasz
'''

import os
import re
import sys
from pathlib import Path
import fnmatch
from numpy import absolute
import shutil
import logging

# Name of output directory
outputDirectory = 'output'


def GetScriptname() -> str:
    ''' Returns currently run script name.'''
    return sys.argv[0]


def GetFiles(base: str, pattern: str) -> list:
    '''Return list of files matching pattern in base folder.'''
    return [n for n in fnmatch.filter(os.listdir(base), pattern) if
            os.path.isfile(os.path.join(base, n))]


def GetFileLocation(path: str) -> str:
    ''' Returns file location '''
    return os.path.dirname(path)


def GetFilename(path: str, dropExtension: bool = False) -> str:
    ''' Returns filename'''
    if (dropExtension is True):
        return os.path.splitext(os.path.basename(path))[0]
    return os.path.basename(path)


def GetDirectoryname(path: str) -> str:
    ''' Returns directory name'''
    return os.path.basename(os.path.dirname(path))


def CombineCommonPath(root: str, rel_path: str) -> str:
    ''' Combine both path's with common part.'''
    rootParts = root.split('/')
    relativeParts = rel_path.split('/')
    popped = False
    for part in rootParts:
        if part == relativeParts[0]:
            relativeParts.pop(0)
            popped = True
        elif popped:
            break

    return '/'.join(rootParts+relativeParts).replace('//', '/')


def PathRelative(absolutePath1: str, absolutePath2: str) -> str:
    '''
        Changed absolutePath2 to become relative to absolutePath1.
        !Warning! - Both paths should be absolute!
    '''
    return os.path.relpath(absolutePath2, absolutePath1)


def PathAbsolute(absolutePath: str, relativePath: str) -> str:
    '''
        Create absolute path from relativePath based on absolutePath.
    '''
    # Return relative path if it's absolute
    if (Path(relativePath).is_absolute()):
        return relativePath

    return str(Path(absolutePath).joinpath(relativePath).resolve())


def SeparatePath(path: str) -> list:
    ''' Totally split path to all elements.'''
    return list(Path(path).parts)


def Parent(path: str, level: int = 0) -> str:
    ''' Return parent path'''
    return str(Path(path).parents[level])


def GetLastPathElemets(path: str, n: int = 3) -> str:
    '''
        Get last n path elements,
        joined with / character.
    '''
    return os.path.join(*Path(path).parts[-3:])


def FixPath(path: str) -> str:
    '''
        Fix path / character at the end.
        Only works with directory paths.
    '''
    # Empty path
    if (len(path) == 0):
        return path

    # Path ending with NT or Posix way
    if (path[-1] in ['/', '\\']):
        return path

    # Append directory ending at the end
    return os.path.join(path, '')


def HasExtension(path: str) -> bool:
    ''' True if path has extension'''
    return len(os.path.splitext(path)[1]) != 0


def DropExtension(path: str) -> str:
    ''' Returns filepath without extension'''
    return os.path.splitext(path)[0]


def GetExtension(path: str) -> str:
    ''' Returns extension'''
    return os.path.splitext(path)[1]


def ChangeExtension(path: str, extension: str) -> str:
    ''' Changes extenstion to new '''
    # If extnsion exists
    if ('.' in path):
        path, _ext = os.path.splitext(path)
        return path + extension

    # otherwise add extension
    return path + extension


def Copyfile(src: str, dst: str) -> str:
    ''' Handle standard copyfile method with all
    possible exceptions.'''
    result = None
    try:
        result = shutil.copyfile(src, dst)
    except shutil.SameFileError:
        logging.warning('(Copyfile) Same file %s and %s!',
                        src, dst)
        pass

    return result


def CreateDirectory(path: str) -> None:
    ''' Creates directory.'''
    return Path(path).mkdir(parents=True, exist_ok=True)


def CreateSymlink(source: str,
                  destination: str,
                  force: bool = False) -> None:
    ''' Creates symlink.'''
    # Force : Remove destination if exists
    if (force is True) and (os.path.exists(destination) is True):
        os.remove(destination)

    # Default : Creates only not existing symlinks
    return os.symlink(source, destination)


def CreateOutputDirectory(filepath: str) -> str:
    ''' Creates output directory.'''
    path = os.path.join(outputDirectory, GetFilename(filepath))
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def CreateOutputPath(filepath: str) -> str:
    ''' Creates output path.'''
    return os.path.join(CreateOutputDirectory(filepath), GetFilename(filepath, True))


def GetStoryfileExtensions() -> list:
    ''' Return list of .story extenstions.'''
    return ['.story']


def IsStoryFile(filepath: str) -> bool:
    ''' Check if file is video file.'''
    return GetExtension(filepath).lower() in GetStoryfileExtensions()


def GetVideofileExtensions() -> list:
    ''' Return list of videofile extenstions.'''
    return ['.avi', '.mp4', '.wmv', '.flv', '.dav', '.mkv', '.hevc', '.265', '.asf', '.mov', '.ts']


def IsVideoFile(filepath: str) -> bool:
    ''' Check if file is video file.'''
    return GetExtension(filepath).lower() in GetVideofileExtensions()


def GetImagefileExtensions() -> list:
    ''' Return list of imagefile extenstions.'''
    return ['.jpg', '.jpeg', '.png', '.bmp']


def IsImageFile(filepath: str) -> bool:
    ''' Check if file is video file.'''
    return GetExtension(filepath).lower() in GetImagefileExtensions()


def FindByRegex(location: str, regexpression: str):
    ''' Find all files matching regexpression in location.'''
    matches = []
    regex = re.compile(regexpression)
    for file in os.listdir(location):
        if regex.match(file):
            matches.append(file)

    return matches


def FileLineCompare(path1: str,
                    path2: str,
                    startLine: int = 0,
                    endLine: int = None,
                    ):
    ''' Compare two files line by line.'''
    # Read files
    with open(path1, 'r') as fp1, open(path2, 'r') as fp2:
        # Line number
        line = 0
        while True:
            # Read both line's texts
            line1 = fp1.readline()
            line2 = fp2.readline()

            # If end line exceeded then return
            if (endLine is not None) and (line > endLine):
                return True
            # File error (checking starts from startLine).
            if (line >= startLine) and (line1 != line2):
                return False
            # File finished.
            if (line1 == '') and (line2 == ''):
                return True
            # Increment line number
            line += 1

    return True
