from src.source_target_dataset import SourceTargetDataset, ImageDataset
from src.dt import DecisionTree
from src.logistic_regression import Logistic_regression
from src.eval_model import InpaintingModel
import numpy as np
from skimage import measure
from matplotlib import pyplot as plt
from src.cityscapes_tools import IND2ID
import cv2
from sklearn.feature_selection import SelectFromModel
from scipy import stats
from tqdm import tqdm


def dt_apply(features: np.ndarray, labels: np.ndarray, dt: DecisionTree) -> np.ndarray:
    features_t = features.copy()
    partition_inds, splits = dt.argpartition(features_t, labels)
    for i, part_inds in enumerate(partition_inds):
        split_feat, split_val, label, sibling_partition = splits[i]
        sibling_label = stats.mode(sibling_partition[:, 0])[0]
        # Add to feature value to get > the split_val
        if sibling_label == 0 and label == 1 and split_feat < 0:
            sibling_mean = np.mean(features_t[sibling_partition[sibling_partition[:, 0] == 0, 1].astype(int), -split_feat])
            mean = np.mean(features_t[part_inds.astype(int), -split_feat])
            feat_delta = sibling_mean - mean
            features_t[part_inds.astype(int), -split_feat] += feat_delta
        # Subtract from feature value to get <= split_val
        elif sibling_label == 0 and label == 1:
            sibling_mean = np.mean(features_t[sibling_partition[sibling_partition[:, 0] == 0, 1].astype(int), split_feat])
            mean = np.mean(features_t[part_inds.astype(int), split_feat])
            feat_delta = mean - sibling_mean
            features_t[part_inds.astype(int), split_feat] -= feat_delta
    return features_t


def dt_based_intervention(
        data: SourceTargetDataset,
        dt: DecisionTree,
        inpainter: InpaintingModel) -> SourceTargetDataset:
    """Perform leaf interventions described by a DT on the corresponding
    partitions of a dataset.
    """
    N = data._imgs_s.shape[0]
    partition_inds, splits = dt.argpartition(data._features)
    imgs_modified_s = []
    gt_s = []
    imgs_modified_t = []
    gt_t = []
    for i, part_inds in enumerate(partition_inds):
        split_feat, split_val, sibling_partition = splits[i]
        for ind in part_inds:
            ind = ind.astype(int)
            if ind < N:
                img = data._imgs_s[ind, :]
                semseg = data._gt_semseg_s[ind, :]
            else:
                img = data._imgs_t[ind - N, :]
                semseg = data._gt_semseg_t[ind - N, :]
            # We can remove stuff from this image
            if split_feat >= 19 and split_feat < 38:
                mask = (semseg[:, :, 2] == IND2ID[split_feat - 19]).astype(int)
                all_labels = measure.label(mask, connectivity=2)
                unique_labels = np.unique(all_labels)

                feat_count = len(unique_labels) - 1
                final_mask = np.zeros(mask.shape)
                for component_idx in unique_labels:
                    if component_idx == 0:
                        continue
                    if feat_count < split_val - 2:
                        break
                    final_mask = np.logical_or(
                        final_mask, all_labels == component_idx)
                    feat_count -= 1
                res = inpainter.run(ImageDataset(
                    np.stack([img]), np.stack([final_mask.astype(int)])))
                img = res._imgs[0, :]
                semseg[final_mask.astype(bool)] = 0
            # Increase brightness
            elif split_feat == -38:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(img)
                avg_brightness = np.mean(v)
                v = cv2.add(v, (split_val + 20) - avg_brightness)
                v[v > 255] = 255
                v[v < 0] = 0
                img = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
            # Decrease brightness
            elif split_feat == 38:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(img)
                avg_brightness = np.mean(v)
                v = cv2.add(v, (split_val - 20) - avg_brightness)
                v[v > 255] = 255
                v[v < 0] = 0
                img = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

            if ind < N:
                imgs_modified_s.append(img)
                gt_s.append(semseg)
            else:
                imgs_modified_t.append(img)
                gt_t.append(semseg)
    imgs_modified_s = ImageDataset(np.stack(imgs_modified_s), np.stack(gt_s))
    imgs_modified_t = ImageDataset(np.stack(imgs_modified_t), np.stack(gt_t))

    return SourceTargetDataset(
        imgs_modified_s,
        np.ones(N),
        imgs_modified_t,
        np.zeros(N),
        N * 2)

def lr_based_intervention(data: SourceTargetDataset, CLASSES:int, dt: Logistic_regression, inpainter: InpaintingModel, source_avg_features:np.array, target_avg_features:np.array, all_feat_ids:list) -> SourceTargetDataset:
    """Perform leaf interventions described by a DT on the corresponding
    partitions of a dataset.
    """
    N = data._imgs_s.shape[0]
    feature_names = list(CLASSES) + [c + " count" for c in CLASSES] + [c + " brightness" for c in CLASSES]#["Avg. brightness", "Stdev. brightness"]
    feature_id_ls = list(range(len(feature_names)))
    # remaining_feature_names = []
    # remaining_feature_names.extend(feature_names)
    # remaining_feature_id_ls = []
    # remaining_feature_id_ls.extend(feature_id_ls)
    # data_clone = data.clone()
    selected_feature_count = 20
    # for k in range(5):
    fit_model = dt.fit(data, feature_names)
    selector = SelectFromModel(dt._clf, prefit=True, max_features=selected_feature_count)
    
    selected_features_boolean = selector.get_support()
    # remaining_feature_id_ls = [remaining_feature_id_ls[k] for k in range(len(selected_features_boolean)) if selected_features_boolean[k] is False]
    selected_feature_ids = np.array([feature_id_ls[k] for k in range(len(selected_features_boolean)) if selected_features_boolean[k] == True])
    source_avg_feat_with_selected_feat_ids = source_avg_features[selected_feature_ids]
    target_avg_feat_with_selected_feat_ids = target_avg_features[selected_feature_ids]
    
    # split_feat, semseg, inpainter, split_val
    # if selected_feature_id < 38:

    img_s_copy = np.copy(data._imgs_s)
    gt_semseg_s_copy = np.copy(data._gt_semseg_s)
    img_t_copy = np.copy(data._imgs_t)
    gt_semseg_t_copy = np.copy(data._gt_semseg_t)

    for k in range(len(selected_feature_ids)):
        print("feature name::", selected_feature_ids[k], feature_names[selected_feature_ids[k]])

        selected_feature_id = selected_feature_ids[k]
        if source_avg_feat_with_selected_feat_ids[k] > target_avg_feat_with_selected_feat_ids[k]:
            img_s_copy, gt_semseg_s_copy = do_intervention_by_feature_all(selected_feature_id, img_s_copy, gt_semseg_s_copy, inpainter, source_avg_feat_with_selected_feat_ids[k] - target_avg_feat_with_selected_feat_ids[k])



        else:
            img_t_copy, gt_semseg_t_copy = do_intervention_by_feature_all(selected_feature_id, img_t_copy, gt_semseg_t_copy, inpainter, target_avg_feat_with_selected_feat_ids[k] - source_avg_feat_with_selected_feat_ids[k])
            # else:
            #     do_intervention_by_feature(selected_feature_id, data_clone._gt_semseg_s, inpainter, target_avg_feat_with_selected_feat_id - source_avg_feat_with_selected_feat_id)
            

    imgs_modified = ImageDataset(img_s_copy, gt_semseg_s_copy)
    imgt_modified = ImageDataset(img_t_copy, gt_semseg_t_copy)

    # N = data._imgs_s.shape[0]
    # imgs_modified_s = ImageDataset(np.stack(imgs_modified_s), np.stack(gt_s))
    # imgs_modified_t = ImageDataset(np.stack(imgs_modified_t), np.stack(gt_t))
    return SourceTargetDataset(imgs_modified, np.ones(N), imgt_modified, np.zeros(N), N*2)

def do_intervention_by_feature_all(split_feat, imgs, semseg, inpainter, split_val):
    img_ls = []
    gt_ls = []
    for k in range(semseg.shape[0]):
        img, curr_semseg = do_intervention_by_feature(split_feat, imgs[k], semseg[k], inpainter, split_val)
        img_ls.append(img)
        gt_ls.append(curr_semseg)
    
    # imgs_modified = ImageDataset(np.stack(img_ls), np.stack(gt_ls))

    return np.stack(img_ls), np.stack(gt_ls)

def do_intervention_by_feature(split_feat, img, semseg, inpainter, split_val):
    # if split_val is None:
        
    # We can remove stuff from this image
    if split_feat >= 19 and split_feat < 38:
        mask = (semseg[:,:,2] == IND2ID[split_feat - 19]).astype(int)
        all_labels = measure.label(mask, connectivity=2)
        unique_labels = np.unique(all_labels)

        feat_count = len(unique_labels) - 1
        final_mask = np.zeros(mask.shape)
        for component_idx in unique_labels:
            if component_idx == 0:
                continue
            if feat_count < split_val - 2:
                break
            final_mask = np.logical_or(final_mask, all_labels == component_idx)
            feat_count -= 1
        res = inpainter.run(ImageDataset(np.stack([img]), np.stack([final_mask.astype(int)])))
        img = res._imgs[0,:]
        semseg[final_mask.astype(bool)] = 0
    # Increase brightness
    # elif split_feat == -38:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #     h, s, v = cv2.split(img)
    #     avg_brightness = np.mean(v)
    #     v = cv2.add(v, (split_val + 20) - avg_brightness)
    #     v[v > 255] = 255
    #     v[v < 0] = 0
    #     img = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
    # Decrease brightness
    elif split_feat >= 38:
        mask = (semseg[:, :, 2] == IND2ID[(split_feat - 38) // 2]).astype(int)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img)
        delta_new = np.mean(v[mask]) - split_feat[i, feat_ind]
        v[mask] = cv2.add(v[mask], -delta_new)
        v[v > 255] = 255
        v[v < 0] = 0
        img = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

    return img, semseg


def transformed_feat_intervention(
        data: ImageDataset,
        features: np.ndarray,
        features_t: np.ndarray,
        inpainter: InpaintingModel) -> ImageDataset:
    """Perform augmentations directly on images based on the changes the the features.

    Args:
        data: dataset object with raw images to modify
        features: (N, 21) array representing semseg count features from (0-18) and brightness features from (19-20).
        features_t: (N, 21) array of transformed features.
    Returns:
        Transformed image dataset.
    """
    imgs_modified = []
    gt_modified = []
    delta = features - features_t
    for i in tqdm(range(delta.shape[0])):
        img = data.samples()[i, :]
        semseg = data.targets()[i, :]
        final_mask = np.zeros(semseg[:, :, 2].shape)
        inpaint = False

        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img)
        v = v.astype("float64")
        for feat_ind in list(range(19*3))[::-1]:
            if delta[i, feat_ind] == 0:
                continue
            # We cannot add objects
            if i < 19 and delta[i, feat_ind] <= 0:
                continue
            # Remove objects
            if feat_ind < 19:
                inpaint = True
                mask = (semseg[:, :, 2] == IND2ID[feat_ind]).astype(int)
                all_labels = measure.label(mask, connectivity=2)
                unique_labels = np.unique(all_labels)

                feat_count = 0
                for component_idx in unique_labels:
                    if component_idx == 0:
                        continue
                    if feat_count >= delta[i, feat_ind]:
                        break
                    final_mask = np.logical_or(
                        final_mask, all_labels == component_idx)
                    feat_count += 1
            # Modify avg. hue, saturation, or brightness
            elif feat_ind >= 19 and (feat_ind - 19) % 2 == 0:
                mask = (semseg[:, :, 2] == IND2ID[(feat_ind - 19) // 2])
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                # h, s, v = cv2.split(img)
                # channel = cv2.split(img)[((feat_ind - 19) % 6) // 3]
                # c = ((feat_ind - 19) % 6) // 2
                # c = 2
                delta_new = np.mean(v[mask]) - features_t[i, feat_ind]
                v = cv2.add(v, cv2.multiply(mask.astype("float64"), -delta_new))
                v[v > 255] = 255
                v[v < 0] = 0
                # img = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
                # img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
            # Modify stdev. hue, saturation, or brightness
            elif feat_ind >= 19 and (feat_ind - 19) % 2 == 1:
                mask = (semseg[:, :, 2] == IND2ID[(feat_ind - 19) // 2])
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                # h, s, v = cv2.split(img)
                # c = ((feat_ind - 19) % 6) // 2
                # c = 2
                factor = features_t[i, feat_ind] / features[i, feat_ind]
                v = cv2.multiply(v, cv2.add(np.ones_like(mask) - mask.astype("float64"), cv2.multiply(mask.astype("float64"), factor)))
                v[v > 255] = 255
                v[v < 0] = 0
                # img = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
                # img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        
        v = v.astype("uint8")
        img = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
        if inpaint:
            res = inpainter.run(ImageDataset(
                np.stack([img]), np.stack([final_mask.astype(int)])))
            img = res._imgs[0, :]
            semseg[final_mask.astype(bool)] = 0

        imgs_modified.append(img)
        gt_modified.append(semseg)
    return ImageDataset(np.stack(imgs_modified), np.stack(gt_modified))