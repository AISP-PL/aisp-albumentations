'''
    Helper functions for bounding boxes.
'''


def RectCheckFit(rect: list) -> list:
    ''' Check all cooridnates between 0..1'''
    rect[0] = max(0, min(1, rect[0]))
    rect[1] = max(0, min(1, rect[1]))
    rect[2] = max(0, min(1, rect[2]))
    rect[3] = max(0, min(1, rect[3]))
    return rect


def RectToXYWH(box) -> list:
    ''' Conversion '''
    x1, y1, x2, y2 = box
    return [(x1+x2)/2, (y1+y2)/2, abs(x2-x1), abs(y2-y1)]


def XYWHToRect(box: list) -> list:
    ''' Conversion '''

    x, y, w, h = box
    w2 = w/2
    h2 = h/2

    x1 = (x - w2)
    x2 = (x + w2)
    y1 = (y - h2)
    y2 = (y + h2)

    return [x1, y1, x2, y2]
