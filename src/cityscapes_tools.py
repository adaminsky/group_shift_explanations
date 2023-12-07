import torch
import os
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage import measure
import cv2

CLASSES = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)

PALETTE = {
    tuple([128, 64, 128]): 7,
    tuple([244, 35, 232]): 8,
    tuple([70, 70, 70]): 11,
    tuple([102, 102, 156]): 12,
    tuple([190, 153, 153]): 13,
    tuple([153, 153, 153]): 17,
    tuple([250, 170, 30]): 19,
    tuple([220, 220, 0]): 20,
    tuple([107, 142, 35]): 21,
    tuple([152, 251, 152]): 22,
    tuple([70, 130, 180]): 23,
    tuple([220, 20, 60]): 24,
    tuple([255, 0, 0]): 25,
    tuple([0, 0, 142]): 26,
    tuple([0, 0, 70]): 27,
    tuple([0, 60, 100]): 28,
    tuple([0, 80, 100]): 31,
    tuple([0, 0, 230]): 32,
    tuple([119, 11, 32]): 33,
}

IND2ID = {
    0: 7,
    1: 8,
    2: 11,
    3: 12,
    4: 13,
    5: 17,
    6: 19,
    7: 20,
    8: 21,
    9: 22,
    10: 23,
    11: 24,
    12: 25,
    13: 26,
    14: 27,
    15: 28,
    16: 31,
    17: 32,
    18: 33,
}


def cs_labelids2labels(semseg: np.ndarray) -> np.ndarray:
    return np.vectorize(IND2ID.get)(semseg)


def img2results(img: np.ndarray):
    """Given a Cityscapes colorized semantic segmentation image, outputs the
    semantic segmentation pixel labels."""
    img = np.moveaxis(img[:, :, :], -1, 0)
    results = np.zeros(img.shape[1:])
    for i in range(img.shape[1]):
        for j in range(img.shape[2]):
            if tuple(img[:, i, j].tolist()[::-1]) not in PALETTE:
                results[i, j] = 0
            else:
                results[i, j] = PALETTE[tuple(img[:, i, j].tolist()[::-1])]
    return results


def scandir(dir_path, suffix=None, recursive=False, case_sensitive=True):
    """Scan a directory to find the interested files.
    Args:
        dir_path (str | :obj:`Path`): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        case_sensitive (bool, optional) : If set to False, ignore the case of
            suffix. Default: True.
    Returns:
        A generator for all the interested files with relative paths.
    """
    if isinstance(dir_path, (str, Path)):
        dir_path = str(dir_path)
    else:
        raise TypeError('"dir_path" must be a string or Path object')

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    if suffix is not None and not case_sensitive:
        suffix = (
            suffix.lower()
            if isinstance(suffix, str)
            else tuple(item.lower() for item in suffix)
        )

    root = dir_path

    def _scandir(dir_path, suffix, recursive, case_sensitive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith(".") and entry.is_file():
                rel_path = os.path.relpath(entry.path, root)
                _rel_path = rel_path if case_sensitive else rel_path.lower()
                if suffix is None or _rel_path.endswith(suffix):
                    yield rel_path
            elif recursive and os.path.isdir(entry.path):
                # scan recursively if entry.path is a directory
                yield from _scandir(entry.path, suffix, recursive, case_sensitive)

    return _scandir(dir_path, suffix, recursive, case_sensitive)


def calculateIOU(pred: np.ndarray, gt: np.ndarray):
    eval_labels = [v for _, v in PALETTE.items()]
    total_intersect = 0
    total_union = 0
    for label in eval_labels:
        total_intersect += np.sum(np.logical_and(pred == label, gt == label))
        total_union += np.sum(np.logical_or(pred == label, gt == label))
        # print(np.sum(np.logical_and(pred == label, gt == label)) / np.sum(np.logical_or(pred == label, gt == label)))
    return total_intersect / total_union


def evaluateImage(pred, ann):
    """Compute IOU for image and annotation path pair
    pred: path to the prediction image
    ann: path to the annotation image
    """
    pred_img = Image.open(pred)
    pred_img = img2results(np.array(pred_img))

    ann_img = Image.open(ann)
    ann_img = np.array(ann_img)
    return calculateIOU(pred_img, ann_img)


def evaluateImageLists(pred_list, ann_list):
    results = {}
    for (pred, ann) in tqdm(zip(pred_list, ann_list), desc="Calculating IOUs"):
        results.update({ann: evaluateImage(pred, ann)})
    return results


def evaluate_cityscapes(
        predfile_prefix,
        ann_dir,
        ann_suffix="gtFine_labelIds.png"):
    """Evaluation in Cityscapes protocol.

    Args:
        predfile_prefix (str): The prefix of output image file
        ann_dir (str): The prefix of annotation files

    Returns:
        dict[str: float]: Cityscapes evaluation results.
    """
    try:
        import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as CSEval
    except ImportError:
        raise ImportError(
            'Please run "pip install cityscapesscripts" to '
            "install cityscapesscripts first."
        )

    result_dir = predfile_prefix

    eval_results = dict()

    CSEval.args.evalInstLevelScore = True  # type: ignore
    CSEval.args.predictionPath = os.path.abspath(result_dir)  # type: ignore
    CSEval.args.evalPixelAccuracy = True  # type: ignore
    CSEval.args.JSONOutput = False  # type: ignore

    seg_map_list = []
    pred_list = []

    # when evaluating with official cityscapesscripts,
    # **_gtFine_labelIds.png is used
    for seg_map in scandir(ann_dir, ann_suffix, recursive=True):
        if "test" not in seg_map and "train" not in seg_map:
            seg_map_list.append(os.path.join(ann_dir, seg_map))
            if ann_suffix != "gtFine_labelIds.png":
                pred_list.append(predfile_prefix + "/" +
                                 seg_map.split("/")[-1])
            else:
                pred_list.append(CSEval.getPrediction(CSEval.args, seg_map))

    seg_map_list = sorted(seg_map_list)[:500]
    pred_list = sorted(pred_list)[:500]
    # print(seg_map_list[:5])
    # print(pred_list[:5])

    eval_results.update(evaluateImageLists(pred_list, seg_map_list))
    # eval_results.update(CSEval.evaluateImgLists(pred_list, seg_map_list, CSEval.args))

    return eval_results


def semseg_area_features(img: np.ndarray, grid: int = 1) -> np.ndarray:
    N = img.shape[0] // grid
    M = img.shape[1] // grid
    tiles = [
        img[x: x + N, y: y + M]
        for x in range(0, img.shape[0], N)
        for y in range(0, img.shape[1], M)
    ]

    def get_features(tile: np.ndarray) -> np.ndarray:
        eval_labels = sorted([v for _, v in PALETTE.items()])
        area_features = np.zeros(len(eval_labels))
        count_features = np.zeros(len(eval_labels))

        # Get the count features for each semantic segmentation type
        for i, label in enumerate(eval_labels):
            area_features[i] = np.sum(tile == label)
            all_labels = measure.label(
                (tile == label).astype(int), connectivity=2)
            unique_labels = np.unique(all_labels)
            count_features[i] = len(unique_labels) - 1
        return count_features
        # return np.concatenate(
        #     [area_features / (tile.shape[0] * tile.shape[1]), count_features])

    return np.concatenate([get_features(tile) for tile in tiles])


def get_count_features(semseg_instance):
    features = torch.zeros(23)
    for i in range(23):
        features[i] = torch.sum(semseg_instance == i)
    return features / torch.sum(features)


def get_img_features(img: np.ndarray, semseg: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    labels = sorted([v for _, v in PALETTE.items()])
    features = []
    for label in labels:
        mask = semseg[:, :, 2] == label
        if np.sum(mask) > 0:
            features.append(
                np.array([
                    # np.mean(img[:, :, 0][mask]),    # avg. hue
                    # np.std(img[:, :, 0][mask]),     # std. hue
                    # np.mean(img[:, :, 1][mask]),    # avg. saturation
                    # np.std(img[:, :, 1][mask]),     # std. saturation
                    np.mean(img[:, :, 2][mask]),    # avg. value
                    np.std(img[:, :, 2][mask])]))   # std. value
        else:
            features.append(np.array([0.0, 0.0,]))

    features = np.concatenate(features)
    return features
