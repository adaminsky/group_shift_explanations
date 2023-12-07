import numpy as np


PALETTE = {
    tuple([ 0, 0, 0]): 0,
    tuple([ 70, 70, 70]): 1,
    tuple([100, 40, 40]): 2,
    tuple([ 55, 90, 80]): 3,
    tuple([220, 20, 60]): 4,
    tuple([153, 153, 153]): 5,
    tuple([157, 234, 50]): 6,
    tuple([128, 64, 128]): 7,
    tuple([244, 35, 232]): 8,
    tuple([107, 142, 35]): 9,
    tuple([ 0, 0, 142]): 10,
    tuple([102, 102, 156]): 11,
    tuple([220, 220, 0]): 12,
    tuple([ 70, 130, 180]): 13,
    tuple([ 81, 0, 81]): 14,
    tuple([150, 100, 100]): 15,
    tuple([230, 150, 140]): 16,
    tuple([180, 165, 180]): 17,
    tuple([250, 170, 30]): 18,
    tuple([110, 190, 160]): 19,
    tuple([170, 120, 50]): 20,
    tuple([ 45, 60, 150]): 21,
    tuple([145, 170, 100]): 22,
}

SHIFT2CS_LABELS = {
    0: 0,
    1: 11,
    2: 13,
    3: 0,
    4: 24,
    5: 17,
    6: 7,
    7: 7,
    8: 8,
    9: 21,
    10: 26,
    11: 12,
    12: 20,
    13: 23,
    14: 6,
    15: 15,
    16: 10,
    17: 14,
    18: 19,
    19: 4,
    20: 5,
    21: 0,
    22: 22,
}

def shift2cs_labels(semseg):
    return np.vectorize(SHIFT2CS_LABELS.get)(semseg)

def img2results(img):
    """Given a SHIFT colorized semantic segmentation image, outputs the
    semantic segmentation pixel labels."""
    img = np.moveaxis(img[:, :, ::-1], -1, 0)
    results = np.zeros(img.shape[1:])
    for i in range(img.shape[1]):
        for j in range(img.shape[2]):
            if tuple(img[:, i, j].tolist()) not in PALETTE:
                results[i, j] = 0
            else:
                results[i, j] = SHIFT2CS_LABELS[PALETTE[tuple(img[:, i, j].tolist())]]
    return results

def calculateIOU(pred, gt):
    eval_labels = [v for _, v in PALETTE.items()]
    total_intersect = 0
    total_union = 0
    for label in eval_labels:
        total_intersect += np.sum(np.logical_and(pred == label, gt == label))
        total_union += np.sum(np.logical_or(pred == label, gt == label))
        # print(np.sum(np.logical_and(pred == label, gt == label)) / np.sum(np.logical_or(pred == label, gt == label)))
    return total_intersect / total_union