import numpy as np

LABELS = ['apple', 'banana']
IMAGE_H, IMAGE_W = 416, 416
GRID_H, GRID_W = 13, 13
BOX = 5
CLASS = len(LABELS)
CLASS_WEIGHTS = np.ones(CLASS, dtype='float32')
OBJ_THRESHOLD = 0.3  # 0.5
NMS_THRESHOLD = 0.3  # 0.45
ANCHORS = [
    0.57273,
    0.677385,
    1.87446,
    2.06253,
    3.33843,
    5.47434,
    7.88282,
    3.52778,
    9.77052,
    9.16828]
NO_OBJECT_SCALE = 1.0
OBJECT_SCALE = 5.0
COORD_SCALE = 1.0
CLASS_SCALE = 1.0
BATCH_SIZE = 2
WARM_UP_BATCHES = 0
TRUE_BOX_BUFFER = 50


def getParams():
    config = {
        'IMAGE_H': IMAGE_H,
        'IMAGE_W': IMAGE_W,
        'GRID_H': GRID_H,
        'GRID_W': GRID_W,
        'BOX': BOX,
        'LABELS': LABELS,
        'CLASS': len(LABELS),
        'ANCHORS': ANCHORS,
        'BATCH_SIZE': BATCH_SIZE,
        'OBJECT_SCALE': OBJECT_SCALE,
        'NO_OBJECT_SCALE': NO_OBJECT_SCALE,
        'COORD_SCALE': COORD_SCALE,
        'CLASS_SCALE': CLASS_SCALE,
        'TRUE_BOX_BUFFER': TRUE_BOX_BUFFER
    }
    return config
