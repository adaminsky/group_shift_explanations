import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Dataset, DataLoader


class GroupDataset(Dataset):
    def __init__(self, data, labels, groups):
        self.data = data
        self.labels = labels
        self.groups = groups

    def __getitem__(self, index):
        return (self.data[index], self.labels[index], self.groups[index])

    def __len__(self):
        return self.data.shape[0]

def regular_training(model, data, labels, iters, lr, l1=False, l1_lambda=0.1):
    """Optimize a pytorch model

    Inputs:
        - model: pytorch model
        - data: (n, d) numpy array of input data
        - labels: (n,) flattened float array of binary labels
        - iters: number of optimization iterations
        - lr: learning rate
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = torch.nn.BCELoss()
    print(torch.sum(labels) / labels.shape[0])

    group_data = GroupDataset(data, labels, np.ones(data.shape[0]))
    loader = DataLoader(group_data, batch_size=100, shuffle=True)
    model = model.cuda()

    for iteration in range(iters):
        total_loss = 0
        for i, (batch_data, batch_labels, _) in enumerate(loader):
            batch_data = batch_data.cuda()
            batch_labels = batch_labels.cuda()

            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            if l1:
                linear_params = torch.cat([x.view(-1) for x in model.parameters()])
                regular_term = l1_lambda * torch.norm(linear_params, 1)
                loss += regular_term
            loss.backward()
            total_loss += loss.detach().cpu().item()
            optimizer.step()
        avg_loss = total_loss/len(loader)
        if iteration % 10 == 0:
            print(iteration, avg_loss)

    model = model.cpu()
    with torch.no_grad():
        outputs = model(data)
        print(outputs.shape, labels.shape)
        print("Accuracy:", torch.sum(torch.round(outputs) == labels) / outputs.shape[0])
        # print("Accuracy:", torch.sum(torch.argmax(outputs, dim=1) == labels) / outputs.shape[0])

def dro_training(model, data, labels, groups, iters, lr, l1=False, l1_lambda = 0.1):
    """Perform DRO on a pytorch model

    Inputs:
        - model: pytorch model
        - data: (n, d) numpy array of input data
        - labels: (n,) flattened float array of binary labels
        - groups: (n, g+1) array of group membership for each sample plus the
                  last column should be all ones for the group of everything.
        - iters: number of optimization iterations
        - lr: learning rate
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = torch.nn.BCELoss(reduction='none')
    groups = groups
    group_denom = groups.T.sum(1).cuda()
    group_denom = group_denom + (group_denom == 0).float()
    adv_probs = torch.ones(groups.shape[1]).cuda() / groups.shape[1]
    adj = torch.zeros(groups.shape[1]).float().cuda()

    group_data = GroupDataset(data, labels, groups)
    loader = DataLoader(group_data, batch_size=100, shuffle=True)
    model = model.cuda()

    for epoch in range(iters):
        total_loss = 0
        for i, (batch_data, batch_labels, batch_groups) in enumerate(loader):
            optimizer.zero_grad()
            outputs = model(batch_data.cuda())
            losses = criterion(outputs, batch_labels.cuda()).double()
            loss = (batch_groups.T.cuda() @ losses.view(-1)) / group_denom
            adjusted_loss = loss
            if torch.all(adj>0):
                adjusted_loss += adj/torch.sqrt(batch_groups.shape[1])
            adjusted_loss = adjusted_loss / (adjusted_loss.sum())
            adv_probs = adv_probs * torch.exp(0.01 * adjusted_loss.data)
            adv_probs = adv_probs/(adv_probs.sum())

            robust_loss = loss @ adv_probs
            if l1:
                linear_params = torch.cat([x.view(-1) for x in model.parameters()])
                regular_term = l1_lambda * torch.norm(linear_params, 1)
                robust_loss += regular_term
            
            robust_loss.backward()
            optimizer.step()
            total_loss += robust_loss.detach().cpu().item()
            # if i % 1 == 0:
            #     print(i, robust_loss.item())
        avg_loss = total_loss/len(loader)
        if epoch % 10 == 0:
            print(epoch, avg_loss)


    model = model.cpu()
    with torch.no_grad():
        outputs = model(data)
        print("Accuracy:", torch.sum(np.round(outputs.flatten()) == labels.flatten()) / outputs.shape[0])