from __future__ import annotations
import numpy as np
import os
import cv2
from typing import Callable, Iterator
from tqdm import tqdm
import glob
from src.cityscapes_tools import semseg_area_features, get_img_features


class ImageDataset:
    """Encapsulate a dataset consisting of images and some label."""

    def __init__(self, imgs: np.ndarray, targets: np.ndarray, filenames=None) -> None:
        """
        Args:
            imgs: An array of each sample, where the first dimension is the index.
            targets: An array of each label, where the first dimension is the index.
        """
        self._imgs = imgs
        self._filenames = filenames
        self._targets = targets
        self._batch_size = 1

    def set_batch(self, n: int) -> None:
        self._batch_size = n

    def samples(self) -> np.ndarray:
        return self._imgs

    def targets(self) -> np.ndarray:
        return self._targets

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for i in range(0, len(self._imgs)):
            yield (
                self._imgs[i, :],
                self._targets[i, :],
            )

    def add(self, other: ImageDataset) -> None:
        self._imgs = np.concatenate(self._imgs, other._imgs)
        self._targets = np.concatenate(self._targets, other._targets)


class SourceTargetDataset(ImageDataset):
    """A combination of the source data and target data into a 50/50 dataset"""

    def __init__(
        self,
        source_data: ImageDataset,
        source_performance: np.ndarray,
        target_data: ImageDataset,
        target_performance: np.ndarray,
        labels: np.ndarray,
        size: int,
    ) -> None:
        """Create a dataset of given size from the source and target data

        Choose the best size/2 samples from source and worst size/2 from target
        datasets.

        Args:
            source_data: An array of the data from the source distribution.
            source_performance: An array of the model performance on each sample.
            target_data: An array of the data from the target distribution.
            target_performance: An array of the model performance on each sample.
            size: resulting size of the full dataset.
        """
        # self._imgs = np.concatenate([source_data.samples(), target_data.samples()])
        best_source = np.nonzero(source_performance > 0.5)[0]
        worst_target = np.nonzero(target_performance < 0.4)[0]
        # self._labels = (np.concatenate([source_performance[best_source], target_performance]) > 0.7).astype(int)
        self._labels = labels
        # self._targets = np.concatenate([source_data.targets(), target_data.targets()])
        # self._imgs = source_data.samples()
        # self._labels = np.zeros(self._imgs.shape[0])
        # self._targets = source_data.targets()
        self.source_data = source_data
        self.target_data = target_data
        features = []
        imgs_s = []
        imgs_t = []
        gt_semseg_s = []
        gt_semseg_t = []
        # print(len(best_source))
        # print(len(worst_target))
        # print(source_data._imgs.shape)
        # print(target_data._imgs.shape)
        for i in tqdm(best_source[:size//2]):
            features.append(np.concatenate([
                semseg_area_features(source_data._targets[i, :]),
                get_img_features(source_data._imgs[i, :], source_data._targets[i, :])],
            ))
            imgs_s.append(source_data._imgs[i, :])
            gt_semseg_s.append(source_data._targets[i, :])
        for i in tqdm(worst_target[:size//2]):
            features.append(np.concatenate([
                semseg_area_features(target_data._targets[i, :]),
                get_img_features(target_data._imgs[i, :], target_data._targets[i, :])],
            ))
            imgs_t.append(target_data._imgs[i, :])
            gt_semseg_t.append(target_data._targets[i, :])
        self._features = np.stack(features)
        self._imgs_s = np.stack(imgs_s)
        self._imgs_t = np.stack(imgs_t)
        self._gt_semseg_s = np.stack(gt_semseg_s)
        self._gt_semseg_t = np.stack(gt_semseg_t)

        # Subset to the samples containing road, sidewalk, and sky
        present_features = np.logical_and(self._features[:, 19 + 0] > 0, np.logical_and(
            self._features[:, 19 + 1] > 0, self._features[:, 19 + 10] > 0))
        select_source = np.argwhere(present_features[:size//2])
        select_target = np.argwhere(present_features[size//2:])
        N = min(len(select_source), len(select_target))
        select = np.concatenate([select_source[:N], select_target[:N] + size//2])

        self._features = np.squeeze(self._features[select, :])
        self._labels = np.squeeze(self._labels[select])
        self._imgs_s = np.squeeze(self._imgs_s[select_source[:N], :])
        self._imgs_t = np.squeeze(self._imgs_t[select_target[:N], :])
        self._gt_semseg_s = np.squeeze(self._gt_semseg_s[select_source[:N], :])
        self._gt_semseg_t = np.squeeze(self._gt_semseg_t[select_target[:N], :])

        # Remove features not present in all samples
        # not_present_features = np.any(self._features[:, 19:] == 0, axis=0)
        # self._features[:, np.argwhere(not_present_features) + 19] = 0

    def samples(self) -> np.ndarray:
        return self._features

    def labels(self) -> np.ndarray:
        return self._labels

    def __iter__(self):
        for i in range(len(self._imgs)):
            yield (self._features[i, :], self._labels[i])


class SourceTargetDataset2(ImageDataset):
    """A combination of the source data and target data into a 50/50 dataset"""

    def __init__(
        self,
        source_data: ImageDataset,
        source_performance: np.ndarray,
        target_data: ImageDataset,
        target_performance: np.ndarray,
        size: int,
        clone: bool = False
    ) -> None:
        """Create a dataset of given size from the source and target data

        Choose the best size/2 samples from source and worst size/2 from target
        datasets.

        Args:
            source_data: An array of the data from the source distribution.
            source_performance: An array of the model performance on each sample.
            target_data: An array of the data from the target distribution.
            target_performance: An array of the model performance on each sample.
            size: resulting size of the full dataset.
        """
        # self._imgs = np.concatenate([source_data.samples(), target_data.samples()])
        if not clone:
            best_source = np.nonzero(source_performance > 0.5)[0]
            worst_target = np.nonzero(target_performance < 0.4)[0]
            # self._labels = (np.concatenate([source_performance[best_source], target_performance]) > 0.7).astype(int)
            # self._labels = np.concatenate([np.zeros(size//2), np.ones(size//2)])
            # self._targets = np.concatenate([source_data.targets(), target_data.targets()])
            # self._imgs = source_data.samples()
            # self._labels = np.zeros(self._imgs.shape[0])
            # self._targets = source_data.targets()
            features = []
            imgs_s = []
            imgs_t = []
            gt_semseg_s = []
            gt_semseg_t = []
            print(len(best_source))
            print(len(worst_target))
            print(source_data._imgs.shape)
            print(target_data._imgs.shape)
            labels = []
            for i in best_source[:size//2]:
                features.append(np.concatenate([
                    semseg_area_features(source_data._targets[i, :]),
                    get_img_features(source_data._imgs[i, :])],
                ))
                imgs_s.append(source_data._imgs[i, :])
                gt_semseg_s.append(source_data._targets[i, :])
                labels.append()
            for i in worst_target[:size//2]:
                features.append(np.concatenate([
                    semseg_area_features(target_data._targets[i, :]),
                    get_img_features(target_data._imgs[i, :])],
                ))
                imgs_t.append(target_data._imgs[i, :])
                gt_semseg_t.append(target_data._targets[i, :])
            self._features = np.stack(features)
            self._imgs_s = np.stack(imgs_s)
            self._imgs_t = np.stack(imgs_t)
            self._gt_semseg_s = np.stack(gt_semseg_s)
            self._gt_semseg_t = np.stack(gt_semseg_t)

    def samples(self) -> np.ndarray:
        return self._features

    def labels(self) -> np.ndarray:
        return self._labels

    def __iter__(self):
        for i in range(len(self._imgs)):
            yield (self._features[i, :], self._labels[i])

    def clone(self):
        cloned_obj = SourceTargetDataset_with_clone(None, None, None, None, None, True)
        cloned_obj._features = np.copy(self._features)
        cloned_obj._imgs_s = np.copy(self._imgs_s)
        cloned_obj._imgs_t = np.copy(self._imgs_t)
        cloned_obj._gt_semseg_s = np.copy(self._gt_semseg_s)
        cloned_obj._gt_semseg_t = np.copy(self._gt_semseg_t)

def load_from_imgdir(
    imgdir_path: str,
    gtdir_path: str,
    img2gt: Callable[[str], str],
    process_gt: Callable[[np.ndarray], np.ndarray],
    size: int
) -> ImageDataset:

    imgs = []
    targets = []
    filenames = []
    paths = np.array(glob.glob(f"{imgdir_path}/*/*"))
    # Take a random subsample
    subset_indexes = np.random.permutation(len(paths))[:size]
    for img_path in tqdm(paths[subset_indexes]):
        filename = "/".join(img_path.split("/")[-2:])
        sample_img = cv2.imread(img_path)
        gt_img = process_gt(cv2.imread(img2gt(img_path)))
        imgs.append(sample_img)
        targets.append(gt_img)
        filenames.append(filename)

    imgs = np.stack(imgs)
    targets = np.stack(targets)
    return ImageDataset(imgs, targets, filenames=filenames)
