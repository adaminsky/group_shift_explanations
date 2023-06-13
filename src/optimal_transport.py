from __future__ import annotations
from typing import Tuple
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


def group_feature_transport(
    source: np.ndarray,
    target: np.ndarray,
    source_groups: np.ndarray,
    target_groups: np.ndarray,
    n_features: int,
    lr=0.02,
    iters=90,
    init_x_s=None) -> np.ndarray:

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
            [loss_fn((x_s + shift)[labels_s[:, f] == 1], x_t[labels_t[:, f] == 1]) for f in range(labels_s.shape[1])]
            + [loss_fn(x_s + shift, x_t)]) / initial_losses
        # Can also use torch.max, but sum appears to work better
        # loss = torch.sum(losses) #/ (labels_s.shape[0] + labels_t.shape[0])
        ###
        adjusted_loss = losses
        if torch.all(adj>0):
            adjusted_loss += adj/torch.sqrt(torch.tensor(labels_s.shape[1]))
        adjusted_loss = adjusted_loss/(adjusted_loss.sum())
        adv_probs = adv_probs * torch.exp(0.01 * adjusted_loss.data)
        adv_probs = adv_probs/(adv_probs.sum())
        loss = losses @ adv_probs
        ###
        print(f"Iter {i} loss: {loss.item()}")
        [g] = torch.autograd.grad(loss, [shift])
        m = labels_s.shape[0]
        shift_prev = shift.clone().detach()
        shift.data -= lr * m * g

        # stop early if we've converged
        with torch.no_grad():
            print(f"Change in x_s: {torch.norm(shift.detach() - shift_prev)}")
            if torch.allclose(shift.detach() - shift_prev, torch.zeros_like(shift), atol=1e-3):
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
    init_clusters=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

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
        losses = torch.stack(
            [loss_fn(x_s[labels_s[:, f] == 1], x_t[labels_t[:, f] == 1]) for f in range(labels_s.shape[1])]
            + [loss_fn(x_s, x_t)]) / initial_losses
        # loss = torch.sum(losses) # + 0.1 * losses[-1]
        ###
        adjusted_loss = losses
        if torch.all(adj>0):
            adjusted_loss += adj/torch.sqrt(torch.tensor(labels_s.shape[1]))
        adjusted_loss = adjusted_loss/(adjusted_loss.sum())
        adv_probs = adv_probs * torch.exp(0.01 * adjusted_loss.data)
        adv_probs = adv_probs/(adv_probs.sum())
        loss = losses @ adv_probs
        ###
        print(f"Iter {t} loss: {loss.item()}")
        [g] = torch.autograd.grad(loss, [m])
        m.grad = g
        m_prev = m.clone().detach()
        optim.step()
        # m.data -= lr * g

        # stop early if converged
        print(f"Change in x_s: {torch.norm(m.detach() - m_prev)}")
        if torch.allclose(m, m_prev, atol=1e-3):
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