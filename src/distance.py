import ot
import numpy as np
import math
import torch


def W2_dist(X1: np.ndarray, X2: np.ndarray, squared=True) -> float:
    """Calculate the squared Wasserstein-2 distance between the distributions"""
    a, b = ot.utils.unif(X1.shape[0]), ot.utils.unif(X2.shape[0])
    M = ot.utils.euclidean_distances(X1, X2, squared=True)
    W2_squared = ot.emd2(a, b, M, numItermax=int(1e9))
    return W2_squared


def group_percent_explained(source, source_t, target, source_groups, target_groups, group_names):
    W2_squared = W2_dist(source, target)
    W2_squared_t = W2_dist(source_t, target)
    pe = (W2_squared - W2_squared_t) / W2_squared
    total_pe = pe * 100
    print("Total Percent Explained:", pe * 100)

    features_keep = []
    for f in range(source_groups.shape[1]):
        if np.sum(source_groups[:, f] == 1) < 1 or np.sum(target_groups[:, f] == 1) < 1:
            continue
        features_keep.append(f)

    pes = []
    for f in features_keep:
        W2_squared = W2_dist(source[source_groups[:, f]==1], target[target_groups[:, f]==1])
        W2_squared_t = W2_dist(source_t[source_groups[:, f]==1], target[target_groups[:, f]==1])
        pe = (W2_squared - W2_squared_t) / W2_squared if W2_squared > 0 else 1
        pes.append(pe)
        print(f"{group_names[f]}", "Percent Explained:", pe * 100)
    print("Worst group PE:", np.min(np.array(pes)) * 100)
    return total_pe, np.min(np.array(pes)) * 100

def percent_flipped(model, source, source_t, groups):
    with torch.no_grad():
        source_output = model(torch.from_numpy(source).float()).numpy().round(0)
        source_t_output = model(torch.from_numpy(source_t).float()).numpy().round(0)
    flipped = np.logical_and(
        source_output == 1,
        np.logical_and(source_t_output == 0, source_output != source_t_output))
    total_pf = np.sum(flipped) / flipped.shape[0]
    group_pf = (groups.T @ flipped).flatten() / np.sum(groups, axis=0).flatten()
    print(100 * total_pf, 100 * group_pf)
    return total_pf * 100, np.min(np.array(group_pf)) * 100