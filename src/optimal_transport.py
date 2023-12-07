from __future__ import annotations
from typing import Tuple
from src.distance import W2_dist
from src.cityscapes_tools import CLASSES
import numpy as np
import ot
from geomloss import SamplesLoss
import torch
from torch.autograd import Variable
from sklearn.cluster import KMeans
from typing import Optional
from sklearn.utils import check_random_state
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MaxAbsScaler


class OTModel():
    def __init__(self):
        self._model = ot.da.EMDTransport()

    def fit(self, X: np.ndarray, Xt: np.ndarray) -> OTModel:
        self._target = Xt.copy()
        self._model.fit(X, Xt=Xt)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._model.transform(Xs=X, Xt=self._target)  # type: ignore

    def cluster_forward(self, X: np.ndarray, n_clusters: int, Y: Optional[np.ndarray] = None) -> np.ndarray:
        if Y is None:
            Y = self._target
        unconstrained_Z = self.transform(X)
        labels = self._pair_clustering(X, unconstrained_Z, n_clusters)
        Z_clusters = self._cluster_mean_transport(X, unconstrained_Z, labels)
        return Z_clusters  

    def _cluster_mean_transport(self, X: np.ndarray, Z: np.ndarray, labels: np.ndarray) -> np.ndarray:
        Z_clusters = X.copy()  # the final output of the cluster mean shift transport
        for cluster_idx in np.unique(labels):
            X_cluster = X[labels == cluster_idx]
            Z_cluster = Z[labels == cluster_idx]
            # since we are doing mean shift cluster transport,
            # C_z = C_x + mean_shift  (mean_shift = C_z_mu - C_x_mu)
            X_cluster_pushed = X_cluster - X_cluster.mean(axis=0) + Z_cluster.mean(axis=0)
            Z_clusters[labels == cluster_idx] = X_cluster_pushed
        return Z_clusters

    @staticmethod
    def _pair_clustering(X: np.ndarray, Z: np.ndarray, n_clusters: int, rng=None) -> np.ndarray:
        rng = check_random_state(rng)
        # Pairing X and Z
        XZ = np.concatenate((X, Z), 1)
        XZ_km = KMeans(n_clusters, init='k-means++', random_state=rng).fit(XZ)
        XZ_labels = XZ_km.predict(XZ)
        return XZ_labels


# Adapted from https://github.com/inouye-lab/explaining-distribution-shifts/blob/master/notebooks/wisconsin-cancer-experiment.ipynb
def iterative_unconstrained_feature_transport(source: np.ndarray, target: np.ndarray, T: OTModel, n_features: int, excluded=None, feature_names=None) -> np.ndarray:
    Z_OT = T.fit(source, target).transform(source)
    W2_X_Y = W2_dist(source, target)  # calculating the W2 distance before any transporting
    X_means = source.mean(axis=0)
    Y_means = target.mean(axis=0)
    diff = Z_OT - source
    if excluded:
        diff[:, excluded] = 0
    argsorted_diff = np.linalg.norm(diff, axis=0).argsort()[::-1]  # a feature-wise divergence array in decreasing order
    free_dim_mask = np.zeros(source.shape[1], dtype=bool)  # a mask where True mean that feature can be transported
    Z_constrained = source.copy()
    W2s = np.zeros(source.shape[1]+1)  # W_2^2 distances of each transport, including T=id
    W2s[0] = W2_X_Y
    W2_deltas = np.zeros(source.shape[1]+1)
    transport_costs = np.zeros(source.shape[1]+1)
    # free_dims_over_time = [[]]
    features = []
    for n_free in range(n_features):
        newest_free = argsorted_diff[n_free]  # selecting the next feature to be included in the transport
        free_dim_mask[newest_free] = True
        Z_constrained[:, newest_free] = Z_OT[:, newest_free]
        d_Z_Y = W2_dist(target, Z_constrained)
        # recording
        percent_change = 100*(W2_X_Y - d_Z_Y) / W2_X_Y  # given as a %
        W2_deltas[n_free+1] = percent_change
        W2s[n_free+1] = d_Z_Y
        if not feature_names:
            if newest_free < 19:
                features.append(CLASSES[newest_free] + " count")
            elif newest_free >= 19 and (newest_free - 19) % 2 == 0:
                features.append(CLASSES[(newest_free - 19) // 2] + " avg. brightness")
            elif newest_free >= 19 and (newest_free - 19) % 2 == 1:
                features.append(CLASSES[(newest_free - 19) // 2] + " stdev. brightness")
        else:
            features.append(feature_names[newest_free])
        # transport_costs[n_free+1] = calc_parsimony(X, Z_constrained)
        # free_dims_over_time.append(np.flatnonzero(free_dim_mask))
        print(f'For {n_free+1} free ({features}),\n\tW2: {d_Z_Y:.3f},\tTotal shift explained: {percent_change:.1f}%')
    return Z_constrained

def iterative_mean_shift_transport(source: np.ndarray, target: np.ndarray, n_features: int, feature_names=None) -> np.ndarray:
    W2_X_Y = W2_dist(source, target)  # calculating the W2 distance before any transporting
    source_means = source.mean(axis=0)
    target_means = target.mean(axis=0)
    mean_diff_sort = np.argsort(abs(source_means - target_means))[::-1]  # sorts in decreasing order
    free_dim_mask = np.zeros(source.shape[1], dtype=bool)  # a mask where True mean that feature can be transported
    Z_constrained = source.copy()

    W2s = np.zeros(source.shape[1]+1)  # W_2^2 distances of each transport, including T=id
    W2s[0] = W2_X_Y
    W2_deltas = np.zeros(source.shape[1]+1)
    free_dims_over_time = [[]]
    shift = np.zeros(source.shape[1])
    for n_free in range(n_features):
        newest_free = mean_diff_sort[n_free]  # selecting the next feature to be included in the transport
        free_dim_mask[newest_free] = True
        Z_constrained[:, newest_free] += target_means[newest_free] - source_means[newest_free]
        shift[newest_free] = target_means[newest_free] - source_means[newest_free]
        d_Z_Y = W2_dist(target, Z_constrained)
        # recording
        percent_change = 100*(W2_X_Y - d_Z_Y) / W2_X_Y  # given as a %
        W2_deltas[n_free+1] = percent_change
        W2s[n_free+1] = d_Z_Y
        free_dims_over_time.append(np.flatnonzero(free_dim_mask))
        shifted_by = target_means[free_dims_over_time[-1]] - source_means[free_dims_over_time[-1]]
        indicies_to_reverse_sort = shifted_by.argsort()[::-1].astype(int)   # [::-1] to sort in decreasing order
        feat_names = []
        for ind in free_dims_over_time[-1][indicies_to_reverse_sort]:
            if not feature_names:
                if ind < 19:
                    feat_names.append(CLASSES[ind] + " count")
                elif ind >= 19 and (ind - 19) % 2 == 0:
                    feat_names.append(CLASSES[(ind - 19) // 2] + " avg. brightness")
                elif ind >= 19 and (ind - 19) % 2 == 1:
                    feat_names.append(CLASSES[(ind - 19) // 2] + " stdev. brightness")
            else:
                feat_names.append(feature_names[ind])
        with np.printoptions(precision=2, suppress=True):
            print(f'For {n_free} free features:\n',
                  f'\tFeatures: {feat_names} have been shifted by: {shifted_by[indicies_to_reverse_sort]}\n', 
                  f'\tW2: {d_Z_Y:.3f},\tTotal shift explained: {percent_change:.1f}%')
    return Z_constrained, shift


def group_feature_transport(
    source: np.ndarray,
    target: np.ndarray,
    source_groups: np.ndarray,
    target_groups: np.ndarray,
    n_features: int,
    lr=0.02,
    iters=90,
    init_x_s=None,
    tol=1e-3,
    blur=0.05,
    loss_type="max") -> np.ndarray:

    labels_s = torch.from_numpy(source_groups)
    labels_t = torch.from_numpy(target_groups)

    # Remove groups with less than 50 members
    features_keep = []
    for f in range(labels_s.shape[1]):
        if torch.sum(labels_t[:, f] == 1) < 1 or torch.sum(labels_s[:, f] == 1) < 1:
            continue
        features_keep.append(f)
    print(features_keep)
    labels_s = labels_s[:, features_keep]
    labels_t = labels_t[:, features_keep]

    x_s, x_t = torch.from_numpy(source.copy()).cuda().float(), torch.from_numpy(target.copy()).cuda().float()
    shift = torch.rand(x_s.shape).cuda().float() / 10
    shift.requires_grad = True
    # optimizer = torch.optim.SGD([shift], lr=lr)
    # x_s.requires_grad = True
    shift_prev = shift.clone().detach()

    loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=blur, scaling=0.6)
    # initial_losses = torch.stack(
    #     [loss_fn((x_s + shift)[labels_s[:, f] == 1], x_t[labels_t[:, f] == 1]) for f in range(labels_s.shape[1])]
    #     + [loss_fn(x_s + shift, x_t)]).detach()

    if init_x_s is not None:
        shift.data = torch.from_numpy(init_x_s.copy()).cuda().float()
        shift.requires_grad = True

    adv_probs = torch.ones(labels_t.shape[1] + 1).cuda() / labels_t.shape[1]
    adj = torch.zeros(labels_t.shape[1] + 1).float().cuda()
    for i in range(iters):
        # optimizer.zero_grad()
        # worst group loss on groups plus total distribution
        losses = torch.stack(
            [loss_fn((x_s + shift)[labels_s[:, f] == 1], x_t[labels_t[:, f] == 1]) for f in range(labels_s.shape[1])])
            # + [loss_fn(x_s + shift, x_t)]) #/ initial_losses
        # losses = torch.stack([loss_fn(x_s + shift, x_t)])

        if loss_type == "max":
            loss = torch.max(losses)
        elif loss_type == "sum":
            loss = torch.sum(losses)
        elif loss_type == "dro":
            adjusted_loss = losses
            if torch.all(adj>0):
                adjusted_loss += adj/torch.sqrt(torch.tensor(labels_s.shape[1]))
            adjusted_loss = adjusted_loss/(adjusted_loss.sum())
            adv_probs = adv_probs * torch.exp(0.01 * adjusted_loss.data)
            adv_probs = adv_probs/(adv_probs.sum())
            loss = losses @ adv_probs
        else:
            raise ValueError("Invalid loss type. Can only be 'max', 'sum', or 'dro'")

        print(f"Iter {i} loss: {loss.item()}")
        shift_prev = shift.clone().detach()
        # loss.backward()
        # optimizer.step()
        [g] = torch.autograd.grad(loss, [shift])
        m = labels_s.shape[0]
        shift.data -= (m / (len(losses))) * g * lr

        # stop early if we've converged
        with torch.no_grad():
            print(f"Change in x_s: {torch.norm(shift.detach() - shift_prev)}")
            if torch.allclose(shift.detach() - shift_prev, torch.zeros_like(shift), atol=tol):
                print(f"Converged at iter {i}")
                break

    shift = shift.detach().cpu().numpy()
    diff = shift #x_s - source
    argsorted_diff = np.linalg.norm(diff, axis=0).argsort()[::-1]  # a feature-wise divergence array in decreasing order
    free_dim_mask = np.zeros(source.shape[1], dtype=bool)  # a mask where True mean that feature can be transported
    free_dim_mask[argsorted_diff[:n_features]] = True

    source_t = source.copy()
    source_t[:, free_dim_mask] = x_s[:, free_dim_mask].detach().cpu().numpy() + shift[:, free_dim_mask]

    return source_t


def group_feature_transport2(
    source: np.ndarray,
    target: np.ndarray,
    source_groups: np.ndarray,
    target_groups: np.ndarray,
    n_features: int,
    lr=0.02,
    iters=90,
    init_x_s=None, tol=1e-3) -> np.ndarray:

    labels_s = torch.from_numpy(source_groups)
    labels_t = torch.from_numpy(target_groups)

    # Remove groups with less than 50 members
    features_keep = []
    for f in range(labels_s.shape[1]):
        if torch.sum(labels_t[:, f] == 1) < 1 or torch.sum(labels_s[:, f] == 1) < 1:
            continue
        features_keep.append(f)
    print(features_keep)
    labels_s = labels_s[:, features_keep]
    labels_t = labels_t[:, features_keep]

    x_s, x_t = torch.from_numpy(source.copy()).cuda(), torch.from_numpy(target.copy()).cuda()
    shift = torch.rand(x_s.shape).cuda() / 10
    shift.requires_grad = True
    # x_s.requires_grad = True
    shift_prev = shift.clone().detach()

    loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.1, scaling=0.99)
    initial_losses = torch.stack(
        [loss_fn((x_s + shift)[labels_s[:, f] == 1], x_t[labels_t[:, f] == 1]) for f in range(labels_s.shape[1])]
        + [loss_fn(x_s + shift, x_t)]).detach()

    if init_x_s is not None:
        shift.data = torch.from_numpy(init_x_s.copy()).cuda()
        shift.requires_grad = True

    adv_probs = torch.ones(labels_t.shape[1] + 1).cuda() / labels_t.shape[1]
    adj = torch.zeros(labels_t.shape[1] + 1).float().cuda()
    for i in range(iters):
        # worst group loss on groups plus total distribution
        losses = torch.stack(
            [loss_fn((x_s + shift)[labels_s[:, f] == 1], x_t[labels_t[:, f] == 1]) for f in range(labels_s.shape[1])])
                        # + [loss_fn(x_s + shift, x_t)]) / initial_losses
        # Can also use torch.max, but sum appears to work better
        # loss = torch.sum(losses) #/ (labels_s.shape[0] + labels_t.shape[0])
        ###
        # adjusted_loss = losses
        # if torch.all(adj>0):
        #     adjusted_loss += adj/torch.sqrt(torch.tensor(labels_s.shape[1]))
        # adjusted_loss = adjusted_loss/(adjusted_loss.sum())
        # adv_probs = adv_probs * torch.exp(0.01 * adjusted_loss.data)
        # adv_probs = adv_probs/(adv_probs.sum())
        # loss = losses @ adv_probs
        loss = torch.max(losses)
        ###
        print(f"Iter {i} loss: {loss.item()}")
        [g] = torch.autograd.grad(loss, [shift])
        m = labels_s.shape[0]
        shift_prev = shift.clone().detach()
        shift.data -= lr * m * g

        # stop early if we've converged
        with torch.no_grad():
            print(f"Change in x_s: {torch.norm(shift.detach() - shift_prev)}")
            if torch.allclose(shift.detach() - shift_prev, torch.zeros_like(shift), atol=tol):
                print(f"Converged at iter {i}")
                break

    shift = shift.detach().cpu().numpy()
    diff = shift #x_s - source
    argsorted_diff = np.linalg.norm(diff, axis=0).argsort()[::-1]  # a feature-wise divergence array in decreasing order
    free_dim_mask = np.zeros(source.shape[1], dtype=bool)  # a mask where True mean that feature can be transported
    free_dim_mask[argsorted_diff[:n_features]] = True

    source_t = source.copy()
    source_t[:, free_dim_mask] = x_s[:, free_dim_mask].detach().cpu().numpy() + shift[:, free_dim_mask]

    return source_t


def group_mean_shift_transport(
    source: np.ndarray,
    target: np.ndarray,
    source_groups: np.ndarray,
    target_groups: np.ndarray,
    n_features: int,
    lr=0.02,
    iters=90) -> Tuple[np.ndarray, np.ndarray]:

    labels_s = torch.from_numpy(source_groups)
    labels_t = torch.from_numpy(target_groups)
    features_keep = []
    for f in range(labels_s.shape[1]):
        if torch.sum(labels_t[:, f] == 1) < 1 or torch.sum(labels_s[:, f] == 1) < 1:
            continue
        features_keep.append(f)
    labels_s = labels_s[:, features_keep]
    labels_t = labels_t[:, features_keep]

    m = Variable(torch.zeros(source.shape[1]), requires_grad=True).cuda()
    loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.01, scaling=0.99)

    orig_source = torch.from_numpy(source.copy()).cuda()
    x_s = torch.from_numpy(source.copy()).cuda()
    x_t = torch.from_numpy(target.copy()).cuda()

    initial_losses = torch.stack(
        [loss_fn(x_s[labels_s[:, f] == 1], x_t[labels_t[:, f] == 1]) for f in range(labels_s.shape[1])]
        + [loss_fn(x_s, x_t)]).detach()
    for t in range(iters):
        x_s = orig_source.clone() + m
        losses = torch.stack(
            [loss_fn(x_s[labels_s[:, f] == 1], x_t[labels_t[:, f] == 1]) for f in range(labels_s.shape[1])]
            + [loss_fn(x_s, x_t)]) / initial_losses
        loss = torch.sum(losses) # + 0.1 * losses[-1]
        print(f"Iter {t} loss: {loss.item()}")
        [g] = torch.autograd.grad(loss, [m])
        m.data -= lr * g

    x_s = x_s.detach().cpu().numpy()
    diff = x_s - source
    argsorted_diff = np.linalg.norm(diff, axis=0).argsort()[::-1]  # a feature-wise divergence array in decreasing order
    free_dim_mask = np.zeros(source.shape[1], dtype=bool)  # a mask where True mean that feature can be transported
    free_dim_mask[argsorted_diff[:n_features]] = True

    source_t = source.copy()
    source_t[:, free_dim_mask] = x_s[:, free_dim_mask]
    shift = m.data.detach().cpu().numpy()
    shift[~free_dim_mask] = 0
    return source_t, shift


def transform_samples(new_source, orig_source, transformed_source):
    transport = transformed_source - orig_source
    knn = KNeighborsClassifier(n_neighbors=1).fit(orig_source, np.arange(orig_source.shape[0]))
    closest = knn.kneighbors(new_source, return_distance=False).flatten()
    return new_source + transport[closest, :]


def transform_samples_kmeans(new_source, centroids, shifts):
    knn = KNeighborsClassifier(n_neighbors=1).fit(centroids, np.arange(centroids.shape[0]))
    closest = knn.kneighbors(new_source, return_distance=False).flatten()
    return new_source + shifts[closest, :]


def group_kmeans_shift_transport(
    source: np.ndarray,
    target: np.ndarray,
    source_groups: np.ndarray,
    target_groups: np.ndarray,
    n_features: int,
    clusters=4,
    lr=0.02,
    iters=90,
    init_clusters=None,
    tol=1e-6,
    blur=0.05,
    loss_type="max") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    labels_s = torch.from_numpy(source_groups)
    labels_t = torch.from_numpy(target_groups)
    features_keep = []
    for f in range(labels_s.shape[1]):
        if torch.sum(labels_t[:, f] == 1) < 1 or torch.sum(labels_s[:, f] == 1) < 1:
            continue
        features_keep.append(f)
    labels_s = labels_s[:, features_keep]
    labels_t = labels_t[:, features_keep]

    scaler = MaxAbsScaler().fit(source)
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(scaler.transform(source))
    cluster_members = kmeans.labels_
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    for c in range(clusters):
        print(f"Samples in cluster {c}: {np.sum(cluster_members == c)}")

    # One row for each cluster mean
    m = torch.rand((clusters, source.shape[1])).float()
    m *=  np.linalg.norm(source) / (100 * torch.linalg.norm(m))
    m = m.cuda()
    m.requires_grad = True
    m_prev = m.clone().detach()
    loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=blur, scaling=0.6)

    # optim = torch.optim.SGD([m], lr=lr)

    orig_source = torch.from_numpy(source.copy()).cuda().float()
    x_s = torch.from_numpy(source.copy()).cuda().float()
    x_t = torch.from_numpy(target.copy()).cuda().float()

    x_s = orig_source.clone()
    for c in range(clusters):
        x_s[cluster_members == c, :] += m[c, :]

    # initial_losses = torch.stack(
    #     [loss_fn(x_s[labels_s[:, f] == 1], x_t[labels_t[:, f] == 1]) for f in range(labels_s.shape[1])]
    #     + [loss_fn(x_s, x_t)]).detach()

    if init_clusters is not None:
        m.data = torch.from_numpy(init_clusters).cuda()
        m.requires_grad = True

    adv_probs = torch.ones(labels_t.shape[1] + 1).cuda() / labels_t.shape[1]
    adj = torch.zeros(labels_t.shape[1] + 1).float().cuda()
    for t in range(iters):
        # optim.zero_grad()
        x_s = orig_source.clone()
        for c in range(clusters):
            x_s[cluster_members == c, :] += m[c, :]
        losses = torch.stack(
            [loss_fn(x_s[labels_s[:, f] == 1], x_t[labels_t[:, f] == 1]) for f in range(labels_s.shape[1])])
            # + [loss_fn(x_s, x_t)]) #/ initial_losse

        if loss_type == "max":
            loss = torch.max(losses)
        elif loss_type == "sum":
            loss = torch.sum(losses)
        elif loss_type == "dro":
            ###
            adjusted_loss = losses
            if torch.all(adj>0):
                adjusted_loss += adj/torch.sqrt(torch.tensor(labels_s.shape[1]))
            adjusted_loss = adjusted_loss/(adjusted_loss.sum())
            adv_probs = adv_probs * torch.exp(0.01 * adjusted_loss.data)
            adv_probs = adv_probs/(adv_probs.sum())
            loss = losses @ adv_probs
            ###
        else:
            raise ValueError("Invalid loss type. Can only be 'max', 'sum', or 'dro'")

        print(f"Iter {t} loss: {loss.item()}")
        [g] = torch.autograd.grad(loss, [m])
        # m.grad = g
        m_prev = m.clone().detach()
        # loss.backward()
        # optim.step()
        m.data -= (m.shape[0] / (len(losses))) * g * lr

        # stop early if converged
        print(f"Change in x_s: {torch.norm(m.detach() - m_prev)}")
        if torch.allclose(m, m_prev, atol=tol):
            print("Converged at iteration", t)
            break

    x_s = x_s.detach().cpu().numpy()
    diff = x_s - source
    argsorted_diff = np.linalg.norm(diff, axis=0).argsort()[::-1]  # a feature-wise divergence array in decreasing order
    free_dim_mask = np.zeros(source.shape[1], dtype=bool)  # a mask where True mean that feature can be transported
    free_dim_mask[argsorted_diff[:n_features]] = True

    source_t = source.copy()
    source_t[:, free_dim_mask] = x_s[:, free_dim_mask]
    shift = m.data.detach().cpu().numpy()
    shift[:, ~free_dim_mask] = 0

    return source_t, centroids, shift


def group_kmeans_shift_transport2(
    source: np.ndarray,
    target: np.ndarray,
    source_groups: np.ndarray,
    target_groups: np.ndarray,
    n_features: int,
    clusters=4,
    lr=0.02,
    iters=90,
    init_clusters=None, tol=1e-3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    labels_s = torch.from_numpy(source_groups)
    labels_t = torch.from_numpy(target_groups)
    features_keep = []
    for f in range(labels_s.shape[1]):
        if torch.sum(labels_t[:, f] == 1) < 1 or torch.sum(labels_s[:, f] == 1) < 1:
            continue
        features_keep.append(f)
    labels_s = labels_s[:, features_keep]
    labels_t = labels_t[:, features_keep]

    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(MaxAbsScaler().fit_transform(source))
    cluster_members = kmeans.labels_
    centroids = kmeans.cluster_centers_
    for c in range(clusters):
        print(f"Samples in cluster {c}: {np.sum(cluster_members == c)}")

    # One row for each cluster mean
    m = torch.rand((clusters, source.shape[1])).cuda()
    m.requires_grad = True
    m_prev = m.clone().detach()
    loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.01, scaling=0.99)

    optim = torch.optim.SGD([m], lr=lr)

    orig_source = torch.from_numpy(source.copy()).cuda()
    x_s = torch.from_numpy(source.copy()).cuda()
    x_t = torch.from_numpy(target.copy()).cuda()

    x_s = orig_source.clone()
    for c in range(clusters):
        x_s[cluster_members == c, :] += m[c, :]

    initial_losses = torch.stack(
        [loss_fn(x_s[labels_s[:, f] == 1], x_t[labels_t[:, f] == 1]) for f in range(labels_s.shape[1])]
        + [loss_fn(x_s, x_t)]).detach()

    if init_clusters is not None:
        m.data = torch.from_numpy(init_clusters).cuda()
        m.requires_grad = True

    adv_probs = torch.ones(labels_t.shape[1] + 1).cuda() / labels_t.shape[1]
    adj = torch.zeros(labels_t.shape[1] + 1).float().cuda()
    for t in range(iters):
        x_s = orig_source.clone()
        for c in range(clusters):
            x_s[cluster_members == c, :] += m[c, :]
        # losses = torch.stack(
        #     [loss_fn(x_s[labels_s[:, f] == 1], x_t[labels_t[:, f] == 1]) for f in range(labels_s.shape[1])]
        #     + [loss_fn(x_s, x_t)]) / initial_losses
        # # loss = torch.sum(losses) # + 0.1 * losses[-1]
        # ###
        # adjusted_loss = losses
        # if torch.all(adj>0):
        #     adjusted_loss += adj/torch.sqrt(torch.tensor(labels_s.shape[1]))
        # adjusted_loss = adjusted_loss/(adjusted_loss.sum())
        # adv_probs = adv_probs * torch.exp(0.01 * adjusted_loss.data)
        # adv_probs = adv_probs/(adv_probs.sum())
        # loss = losses @ adv_probs
        losses = torch.stack(
            [loss_fn(x_s[labels_s[:, f] == 1], x_t[labels_t[:, f] == 1]) for f in range(labels_s.shape[1])])
        loss = torch.max(losses)
        ###
        print(f"Iter {t} loss: {loss.item()}")
        [g] = torch.autograd.grad(loss, [m])
        m.grad = g
        m_prev = m.clone().detach()
        optim.step()
        # m.data -= lr * g

        # stop early if converged
        print(f"Change in x_s: {torch.norm(m.detach() - m_prev)}")
        if torch.allclose(m, m_prev, atol=tol):
            print("Converged at iteration", t)
            break

    x_s = x_s.detach().cpu().numpy()
    diff = x_s - source
    argsorted_diff = np.linalg.norm(diff, axis=0).argsort()[::-1]  # a feature-wise divergence array in decreasing order
    free_dim_mask = np.zeros(source.shape[1], dtype=bool)  # a mask where True mean that feature can be transported
    free_dim_mask[argsorted_diff[:n_features]] = True

    source_t = source.copy()
    source_t[:, free_dim_mask] = x_s[:, free_dim_mask]
    shift = m.data.detach().cpu().numpy()
    shift[:, ~free_dim_mask] = 0

    return source_t, centroids, shift