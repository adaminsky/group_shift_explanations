import torch


def worst_group_loss(x, loss_fn, groups):
    losses = loss_fn(x)
    group_vals = torch.unique(groups)
    return torch.max(torch.tensor([torch.sum(losses[groups == group]) for group in group_vals]))