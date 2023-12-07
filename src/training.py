import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import Dataset, DataLoader
import sys


class GroupDataset(Dataset):
    def __init__(self, data, labels, groups):
        self.data = data
        self.labels = labels
        self.groups = groups

    def __getitem__(self, index):
        return (self.data[index], self.labels[index], self.groups[index])

    def __len__(self):
        return self.data.shape[0]

class Subset(torch.utils.data.Dataset):
    """
    Subsets a dataset while preserving original indexing.
    NOTE: torch.utils.dataset.Subset loses original indexing.
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

        # self.group_array = self.get_group_array(re_evaluate=True)
        # self.label_array = self.get_label_array(re_evaluate=True)
        

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def get_group_array(self, re_evaluate=True):
        """Return an array [g_x1, g_x2, ...]"""
        # setting re_evaluate=False helps us over-write the group array if necessary (2-group DRO)
        if re_evaluate:
            group_array = self.dataset.get_group_array()[self.indices]        
            assert len(group_array) == len(self)
            return group_array
        else:
            return self.group_array

    def get_label_array(self, re_evaluate=True):
        if re_evaluate:
            label_array = self.dataset.get_label_array()[self.indices]
            assert len(label_array) == len(self)
            return label_array
        else:
            return self.label_array


class ConcatDataset(torch.utils.data.ConcatDataset):
    """
    Concate datasets
    Extends the default torch class to support group and label arrays.
    """
    def __init__(self, datasets):
        super(ConcatDataset, self).__init__(datasets)

    def get_group_array(self):
        group_array = []
        for dataset in self.datasets:
            group_array += list(np.squeeze(dataset.get_group_array()))
        return group_array

    def get_label_array(self):
        label_array = []
        for dataset in self.datasets:
            label_array += list(np.squeeze(dataset.get_label_array()))
        return label_array

def regular_training(model, data, labels, groups, iters, lr, weight_decay=1e-2):
    """Optimize a pytorch model

    Inputs:
        - model: pytorch model
        - data: (n, d) numpy array of input data
        - labels: (n,) flattened float array of binary labels
        - iters: number of optimization iterations
        - lr: learning rate
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCELoss()
    print(torch.sum(labels) / labels.shape[0])
    group_denom = groups.T.sum(1)
    group_denom = group_denom + (group_denom == 0).float()

    group_data = GroupDataset(data, labels, np.ones(data.shape[0]))
    loader = DataLoader(group_data, batch_size=len(group_data), shuffle=True)
    model = model.cuda().train()

    for iteration in range(iters):
        for i, (batch_data, batch_labels, _) in enumerate(loader):
            batch_data = batch_data.cuda()
            batch_labels = batch_labels.cuda()

            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            if i % 1 == 0:
                print(iteration, loss.item(), file=sys.stderr)

    model = model.cpu().eval()
    with torch.no_grad():
        outputs = model(data)
        print(outputs.shape, labels.shape)
        print("Accuracy:", torch.sum(torch.round(outputs) == labels) / outputs.shape[0])
        print((groups.T.float() @ (torch.round(outputs.flatten()) == labels.flatten()).float()) / group_denom.cpu())
        # print("Accuracy:", torch.sum(torch.argmax(outputs, dim=1) == labels) / outputs.shape[0])
        # print((groups.T.float() @ (torch.argmax(outputs, dim=1) == labels).float()) / group_denom)

    
def jtt_training(model, data, labels, groups, iters, lr, weight_decay=1e-2):
    """Optimize a pytorch model

    Inputs:
        - model: pytorch model
        - data: (n, d) numpy array of input data
        - labels: (n,) flattened float array of binary labels
        - iters: number of optimization iterations
        - lr: learning rate
    """
    # First do regular training
    regular_training(model, data, labels, groups, 60, lr, weight_decay)

    # Get misclassified samples
    pred_wrong = model(data).round() != labels
    print(torch.nonzero(pred_wrong)[0].shape)
    print(torch.concat([data, data[torch.nonzero(pred_wrong)[0]]]).shape)
    data_upsampled = torch.concat([data] + ([data[torch.nonzero(pred_wrong)[0]]] * 50), dim=0)
    labels_upsampled = torch.concat([labels] + ([labels[torch.nonzero(pred_wrong)[0]]] * 50), dim=0)
    groups_upsampled = torch.concat([groups] + ([groups[torch.nonzero(pred_wrong)[0]]] * 50), dim=0)
    regular_training(model, data_upsampled, labels_upsampled, groups_upsampled, iters, lr, weight_decay)


def dro_training(model, data, labels, groups, iters, lr, weight_decay=1e-2, loss_type="max"):
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
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    # criterion = torch.nn.CrossEntropyLoss(reduction='none')
    criterion = torch.nn.BCELoss(reduction='none')
    groups = groups
    group_denom = groups.T.sum(1).cuda()
    group_denom = group_denom + (group_denom == 0).float()
    adv_probs = torch.ones(groups.shape[1]).cuda() / groups.shape[1]
    adj = torch.zeros(groups.shape[1]).float().cuda()

    group_data = GroupDataset(data, labels, groups)
    # group_array = []
    # y_array = []
    # for x,y,g in group_data:
    #     group_array.append(g)
    #     y_array.append(y)
    # _group_array = torch.LongTensor(group_array)
    # _y_array = torch.LongTensor(y_array)
    _group_array = torch.argmax(groups, dim=1)
    _y_array = labels
    _group_counts = (torch.arange(groups.shape[1]).unsqueeze(1)==_group_array).sum(1).float()
    group_weights = len(group_data)/_group_counts
    weights = group_weights[_group_array]

    batch_size = len(group_data)
    # batch_size = 1000
    sampler = WeightedRandomSampler(weights, batch_size, replacement=True)
    loader = DataLoader(group_data, batch_size=batch_size, shuffle=False, sampler=sampler)
    model = model.cuda()
    model.train()

    for epoch in range(iters):
        for i, (batch_data, batch_labels, batch_groups) in enumerate(loader):
            optimizer.zero_grad()
            outputs = model(batch_data.cuda())
            losses = criterion(outputs, batch_labels.cuda())
            batch_group_denom = batch_groups.sum(0).float().cuda()
            batch_group_denom = batch_group_denom + (batch_group_denom == 0).float()

            loss = (batch_groups.T.cuda() @ losses.view(-1)) / batch_group_denom

            if loss_type == "max":
                robust_loss = torch.max(loss)
            elif loss_type == "sum":
                robust_loss = torch.sum(loss)
            elif loss_type == "dro":
                adjusted_loss = loss
                if torch.all(adj>0):
                    adjusted_loss += adj/torch.sqrt(batch_groups.shape[1])
                adjusted_loss = adjusted_loss / (adjusted_loss.sum())
                adv_probs = adv_probs * torch.exp(0.01 * adjusted_loss.data)
                adv_probs = adv_probs/(adv_probs.sum())
                robust_loss = loss @ adv_probs
            else:
                raise ValueError("loss_type must be one of 'max', 'sum', or 'dro'")

            robust_loss.backward()
            optimizer.step()
            if i % 1 == 0:
                print(epoch, robust_loss.item(), file=sys.stderr)

    model = model.cpu().eval()
    with torch.no_grad():
        outputs = model(data)
        # print("Accuracy:", torch.sum(torch.argmax(outputs, dim=1) == labels) / outputs.shape[0])
        # print((groups.T.float() @ (torch.argmax(outputs, dim=1) == labels).float()) / group_denom.cpu())
        print("Accuracy:", torch.sum(torch.round(outputs.flatten()) == labels.flatten()) / outputs.shape[0])
        print((groups.T.float() @ (torch.round(outputs.flatten()) == labels.flatten()).float()) / group_denom.cpu())

def dro_training2(model, data, labels, groups, iters, lr, weight_decay=1e-2):
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
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    # criterion = torch.nn.CrossEntropyLoss(reduction='none')
    criterion = torch.nn.BCELoss(reduction='none')
    groups = groups
    group_denom = groups.T.sum(1).cuda()
    group_denom = group_denom + (group_denom == 0).float()
    adv_probs = torch.ones(groups.shape[1]).cuda() / groups.shape[1]
    adj = torch.zeros(groups.shape[1]).float().cuda()

    group_data = GroupDataset(data, labels, groups)
    # group_array = []
    # y_array = []
    # for x,y,g in group_data:
    #     group_array.append(g)
    #     y_array.append(y)
    # _group_array = torch.LongTensor(group_array)
    # _y_array = torch.LongTensor(y_array)
    _group_array = torch.argmax(groups, dim=1)
    _y_array = labels
    _group_counts = (torch.arange(groups.shape[1]).unsqueeze(1)==_group_array).sum(1).float()
    group_weights = len(group_data)/_group_counts
    weights = group_weights[_group_array]

    sampler = WeightedRandomSampler(weights, len(group_data), replacement=True)
    loader = DataLoader(group_data, batch_size=len(group_data), shuffle=False, sampler=sampler)
    model = model.cuda()
    model.train()

    for epoch in range(iters):
        for i, (batch_data, batch_labels, batch_groups) in enumerate(loader):
            optimizer.zero_grad()
            outputs = model(batch_data.cuda())
            losses = criterion(outputs, batch_labels.cuda())
            batch_group_denom = batch_groups.sum(0)
            batch_group_denom = batch_group_denom + (batch_group_denom == 0).float()
            loss = (batch_groups.float().T.cuda() @ losses.view(-1)) / batch_group_denom.cuda()
            # adjusted_loss = loss
            # if torch.all(adj>0):
            #     adjusted_loss += adj/torch.sqrt(batch_groups.shape[1])
            # adjusted_loss = adjusted_loss / (adjusted_loss.sum())
            # adv_probs = adv_probs * torch.exp(0.01 * adjusted_loss.data)
            # adv_probs = adv_probs/(adv_probs.sum())

            # robust_loss = loss @ adv_probs
            robust_loss = torch.max(loss)
            robust_loss.backward()
            optimizer.step()
            if i % 1 == 0:
                print(epoch, robust_loss.item(), file=sys.stderr)

    model = model.cpu().eval()
    with torch.no_grad():
        outputs = model(data)
        # print("Accuracy:", torch.sum(torch.argmax(outputs, dim=1) == labels) / outputs.shape[0])
        # print((groups.T.float() @ (torch.argmax(outputs, dim=1) == labels).float()) / group_denom.cpu())
        print("Accuracy:", torch.sum(torch.round(outputs.flatten()) == labels.flatten()) / outputs.shape[0])
        print((groups.T.float() @ (torch.round(outputs.flatten()) == labels.flatten()).float()) / group_denom.cpu())

def eval_acc(model, data, labels, groups):
    model.eval()
    group_denom = groups.T.sum(1)
    group_denom = group_denom + (group_denom == 0).float()
    with torch.no_grad():
        outputs = model(data)
        print("Accuracy:", torch.sum(np.round(outputs.flatten()) == labels.flatten()) / outputs.shape[0])
        # print("Accuracy:", torch.sum(torch.argmax(outputs, dim=1) == labels) / outputs.shape[0])
        # print((groups.T.float() @ (torch.argmax(outputs, dim=1) == labels).float()) / group_denom)
        print((groups.T.float() @ (torch.round(outputs.flatten()) == labels.flatten()).float()) / group_denom.cpu())