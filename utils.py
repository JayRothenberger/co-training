import random
from math import ceil, floor
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.multiprocessing import set_forkserver_preload, set_start_method
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from visualization import *


class EarlyStopper:
    def __init__(self,
                 metric: Literal['loss', 'accuracy'] = 'accuracy',
                 patience: int = 0,
                 min_delta: float = 0.0):
        self.metric = metric
        self.patience = patience
        self.min_delta = min_delta

        self.early_stop = False
        self.epochs_since_improvement = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0

    def step(self, val_acc: float, val_loss: float):
        self.epochs_since_improvement += 1
        if self.metric == 'loss' and val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            self.epochs_since_improvement = 0
        elif self.metric == 'accuracy' and val_acc > self.best_val_acc + self.min_delta:
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            self.epochs_since_improvement = 0
        if self.epochs_since_improvement > self.patience:
            self.early_stop = True


def create_imagefolder(data, samples, path, transform, new_path=None):
    imgfolder = datasets.ImageFolder(path, transform=transform)
    imgfolder.class_to_idx = data['class_map']
    imgfolder.classes = list(data['class_map'].keys())
    imgfolder.samples = samples

    if new_path is not None:
        imgfolder.root = new_path

    return imgfolder


def create_sampler_loader(rank: int, 
                          world_size: int, 
                          data: torch.utils.data.Dataset,
                          batch_size: int = 64,
                          cuda_kwargs: dict = {'num_workers': 12, 
                                               'pin_memory': True, 
                                               'shuffle': False}, 
                          shuffle=True,
                          persistent_workers=False):
    
    sampler = DistributedSampler(data, rank=rank, num_replicas=world_size, shuffle=shuffle)

    loader_kwargs = {'batch_size': batch_size, 
                     'sampler': sampler, 
                     'multiprocessing_context': 'forkserver', 
                     'persistent_workers': persistent_workers,
                     'drop_last': False}
    
    loader_kwargs.update(cuda_kwargs)

    loader = DataLoader(data, **loader_kwargs)

    return sampler, loader


def create_loader(rank: int, 
                          world_size: int, 
                          data: torch.utils.data.Dataset,
                          batch_size: int = 64,
                          cuda_kwargs: dict = {'num_workers': 12, 
                                               'pin_memory': True, 
                                               'shuffle': False}, 
                          shuffle=True,
                          persistent_workers=False):

    loader_kwargs = {'batch_size': batch_size, 
                     'multiprocessing_context': 'forkserver', 
                     'persistent_workers': persistent_workers,
                     'drop_last': False}
    
    loader_kwargs.update(cuda_kwargs)

    loader = DataLoader(data, **loader_kwargs)

    return loader


def create_samplers_loaders(rank: int, 
                            world_size: int, 
                            views: list[torch.utils.data.Dataset], 
                            batch_size: int = 64,
                            cuda_kwargs: dict = {'num_workers': 12, 
                                                 'pin_memory': True, 
                                                 'shuffle': False},
                            shuffle: bool = True,  
                            persistent_workers: bool = False):
    samplers = []
    loaders = []
    for i in range(len(views)):
        sampler, loader = create_sampler_loader(rank, world_size, views[i], 
                                                batch_size, cuda_kwargs, 
                                                shuffle, persistent_workers)
        samplers.append(sampler)
        loaders.append(loader)
    
    return samplers, loaders


def add_to_imagefolder(paths, labels, dataset):
    """
    Adds the paths with the labels to an image classification dataset

    :list paths: a list of absolute image paths to add to the dataset
    :list labels: a list of labels for each path
    :Dataset dataset: the dataset to add the samples to
    """

    new_samples = list(zip(paths, labels))

    dataset.samples += new_samples

    return dataset


def setup(rank, world_size):
    set_start_method('forkserver')
    set_forkserver_preload(['torch'])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def accuracy(output, target, topk=(1,)):
    output = output.to(torch.device('cpu'))
    target = target.to(torch.device('cpu'))
    maxk = max(topk)
    batch_size = target.shape[0]

    _, idx = output.sort(dim=1, descending=True)
    pred = idx.narrow(1, 0, maxk).t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(dim=0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def epoch_accuracy(loader_s, loader_t, student, teacher):
    out_ct = [((student(L_s.to(0)), y_s), (teacher(L_t.to(0)), y_t)) for (L_s, y_s), (L_t, y_t) in zip(iter(loader_s), iter(loader_t))]

    out_epoch_s = [accuracy(a[0], a[1])[0].cpu().item() for a, _ in out_ct]
    out_epoch_t = [accuracy(b[0], b[1])[0].cpu().item() for _, b in out_ct]
    out_epoch_ct = [accuracy(torch.nn.Softmax(dim=-1)(a[0])*torch.nn.Softmax(dim=-1)(b[0]), a[1])[0].cpu().item() for a, b in out_ct]
    
    return sum(out_epoch_s) / len(out_epoch_s), sum(out_epoch_t) / len(out_epoch_t), sum(out_epoch_ct) / len(out_epoch_t)


# TODO write multi-view implementation (...)
def train_test_split_samples(samples0, samples1, test_size, random_state=None):
    if random_state is not None:
        random.seed(random_state)

    assert test_size > 0 and test_size < 1, \
        'test_size should be a float between (0, 1)'

    assert len(samples0) == len(samples1), \
        'number of samples in samples0, samples1 are not equal'

    idx_samples = list(range(len(samples0)))
    idx_test = random.sample(idx_samples, floor(test_size * len(samples0)))
    idx_train = list(set(idx_samples) - set(idx_test))

    # convert to np array for convenient array indexing
    samples0_np = np.stack([np.array(a) for a in samples0])
    samples1_np = np.stack([np.array(a) for a in samples1])

    samples_train0 = [(str(a[0]), int(a[1]))
                      for a in list(samples0_np[idx_train])]
    samples_test0 = [(str(a[0]), int(a[1]))
                     for a in list(samples0_np[idx_test])]
    samples_train1 = [(str(a[0]), int(a[1]))
                      for a in list(samples1_np[idx_train])]
    samples_test1 = [(str(a[0]), int(a[1]))
                     for a in list(samples1_np[idx_test])]

    assert len(samples_train0) == len(samples_train1), \
            'sample sizes not equal after split'
    assert len(samples_test0) == len(samples_test1), \
            'sample sizes not equal after split'

    return samples_train0, samples_test0, samples_train1, samples_test1


def train_test_split_views(views: list, 
                           test_size: float, 
                           random_state: int = None) -> tuple[list, list]:
    if random_state is not None:
        random.seed(random_state)

    msg1 = 'test size should be a float between (0, 1)'
    assert test_size > 0 and test_size < 1, msg1

    msg2 = 'number of samples in views list is not equal'
    lens_views = set([len(view) for view in views])
    assert len(lens_views) == 1, msg2

    num_samples = len(views[0])
    idx_samples = list(range(num_samples))
    idx_test = random.sample(idx_samples, floor(test_size * num_samples))
    idx_train = list(set(idx_samples) - set(idx_test))

    samples_np = []
    for view in views:
        sample_np = np.stack([np.array(a) for a in view])
        samples_np.append(sample_np)
        
    train_views = []
    test_views = []
    for sample in samples_np:
        samples_train = [(str(a[0]), int(a[1]))
                         for a in list(sample[idx_train])]
        samples_test = [(str(a[0]), int(a[1]))
                        for a in list(sample[idx_test])]
        train_views.append(samples_train)
        test_views.append(samples_test)
    
    return train_views, test_views


def cascade_round(arr):
    s = 0.0
    arr_cp = np.zeros_like(arr)
    for i, a in enumerate(arr):
        s += a
        if s - (s // 1) > .5:
            arr_cp[i] = ceil(a)
        else:
            arr_cp[i] = floor(a)
    return arr_cp.astype(np.int32)


def cascade_round_subset(labels, percent):
    """
    labels: np.array of size (M, )
    
    return: mask of indicies to include if you want to respect stratification
    """
    unique, counts = np.unique(labels, return_counts=True)
    count_per_class = percent * counts
    # ok, but this is not exactly n% we will have some rounding to do here
    count_per_class = cascade_round(count_per_class)

    mask = np.hstack([
        np.random.choice(np.where(labels == unique[l])[0],
                         count_per_class[l],
                         replace=False) for l in range(unique.shape[0])
    ])

    return mask


def stratified_pseudolabel_subset(labels, logits, percent):
    """
    labels: np.array of size (M, )
    
    return: mask of indicies to include if you want to respect stratification
    """
    unique, counts = np.unique(labels, return_counts=True)
    count_per_class = (percent * counts) * (logits.shape[0] / labels.shape[0])
    # ok, but this is not exactly n% we will have some rounding to do here
    count_per_class = cascade_round(count_per_class)

    mask = np.hstack([np.where(labels == unique[l])[0][np.argsort(np.max(torch.nn.Softmax(logits), -1))][:count_per_class[l]] for l in range(unique.shape[0])])

    return mask


def progressive_supset_sample(views: list, 
                              percent_unlbl: float,
                              percent_val: float,
                              k: float = 0.05,
                              random_state: int = 13) -> tuple[list, list, list]:
    """constructs training, validation, and unlabeled sets such that increasing
    the number of samples to use for training will maintain a subset / superset relation 
    on the training and validation sets; the training and validation sets are constructed
    by repeatedly sampling k% of the dataset and splitting the k%

    :param views: list of samples
    :type views: list
    :param percent_unlbl: percentage of samples to hold out as unlabeled data
    :type percent_unlbl: float
    :param percent_val: percentage of labeled samples to hold out for validation
    :type percent_val: float
    :param k: percentage to sample for one iteration, defaults to 0.05
    :type k: float, optional
    :param random_state: seed for random number generator, defaults to 13
    :type random_state: int, optional
    :return: training, validation, unlabeled lists
    :rtype: tuple[list, list, list]
    """
    msg1 = 'percent_unlbl should be a float between (0, 1)'
    assert percent_unlbl > 0 and percent_unlbl < 1, msg1

    msg2 = 'percent_val should be a float between (0, 1)'
    assert percent_val > 0 and percent_val < 1, msg2

    msg3 = 'number of samples in views list is not equal'
    lens_views = set(len(view) for view in views)
    assert len(lens_views) == 1, msg3

    num_samples = len(views[0])
    k_samples = k * num_samples # want to sample these many per iteration?
    labels = np.array([l for _, l in views[0]])
    
    # number of iterations to sample k% of datapoints
    iters = round((1 - percent_unlbl) / k)

    lbl_views = []
    for i in range(len(views)):
        lbl_views.append([])

    unlbl_views = views
    for _ in range(iters):
        random.seed(random_state)
        np.random.seed(random_state)
        idx_samples = cascade_round_subset(labels, k_samples / len(labels) 
                                           if k_samples < len(labels) 
                                           else k)
        
        # mask to remove samples
        mask = np.ones(len(labels), dtype=bool)
        mask[idx_samples] = False
        
        for i in range(len(views)):
            samples_np = np.stack([np.array(a) for a in unlbl_views[i]])
            samples_lbl = [(str(a[0]), int(a[1])) for a in list(samples_np[idx_samples])]
            # no list concat cuz we eventually want to split each subset into 80/20
            lbl_views[i].append(samples_lbl)

            # actually remove the samples from the view
            unlbl_views[i] = [(str(a[0]), int(a[1])) for a in list(samples_np[mask])]
        
        labels = labels[mask]

    train_views = []
    val_views = []
    for i in range(len(views)):
        train_views.append([])
        val_views.append([])

    for i in range(len(views)):
        for partition in lbl_views[i]:
            random.seed(random_state) # massive paranoia
            # split partition into validation / training
            labeled_np = np.stack([np.array(a) for a in partition])
            idx_labeled = list(range(len(labeled_np)))
            # mask
            idx_val = random.sample(idx_labeled, floor(percent_val * len(idx_labeled)))
            idx_train = list(set(idx_labeled) - set(idx_val))
            # convert array to list for concat
            samples_tr = [(str(a[0]), int(a[1])) for a in list(labeled_np[idx_train])]
            samples_val = [(str(a[0]), int(a[1])) for a in list(labeled_np[idx_val])]
            # views update
            train_views[i] += samples_tr
            val_views[i] += samples_val

    return train_views, val_views, unlbl_views


def save_reliability_diagram(view_num: int, 
                             iteration: int,
                             percent_unlabeled: float, 
                             predictions: np.ndarray, 
                             labels: np.ndarray,
                             logits: bool = True):
    rel_diagram = ReliabilityDiagram()
    plt_test = rel_diagram.plot(predictions, labels, logits=logits, title="Reliability Diagram")
    plt_test.savefig(f'plots/unlbl{percent_unlabeled}_view{view_num}_iter{iteration}.png',
                     bbox_inches='tight')


# https://github.com/ai2es/miles-guess/blob/main/mlguess/torch/class_losses.py
import torch.nn.functional as F
import torch



"""

    Categorical losses and utilities

"""

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)


def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = get_device()
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def loglikelihood_loss(y, alpha, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, weights=None, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)

    if isinstance(weights, torch.Tensor):
        weights = weights.to(device)
        loglikelihood = (weights * y).sum(-1) * loglikelihood_loss(y, alpha, device=device)
    else:
        loglikelihood = loglikelihood_loss(y, alpha, device=device)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, weights=None, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    if not isinstance(weights, torch.Tensor):
        A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)
    else:
        weights = weights.to(device)
        A = torch.sum((weights * y).sum(-1) * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div

def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, weights=None, device=None):
    if device is None:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1

    loss = torch.mean(
        mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device)
    )
    return loss


def edl_digamma_loss(
    output, target, epoch_num, num_classes, annealing_step, weights=None, device=None
):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    
    loss = torch.mean(
        edl_loss(
            torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, weights, device
        )
    )
    return loss

def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, weights=None, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.log, target, alpha, epoch_num, num_classes, annealing_step, weights, device
        )
    )
    return loss

class EDLLossMSE:
    def __init__(self, num_classes, weights=None):
        self.epoch_num = 1
        self.annealing_step = 100
        self.num_classes = num_classes
        self.weights = weights
        self.CE = torch.nn.CrossEntropyLoss(weight=weights.to(get_device()).type(torch.float32))

    def __call__(self, output, target):
        # return edl_mse_loss(output, F.one_hot(target, self.num_classes), self.epoch_num, self.num_classes, self.annealing_step, weights=self.weights) + self.CE(output, target)
        return self.CE(output, target)

    def step_epoch(self):
        self.epoch_num += 1

    def step_annealing(self):
        self.annealing_step *= 1.01


import math
from collections import OrderedDict
import os

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class Cos(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X: torch.Tensor):
        return torch.cos(X)


class RandomFeatureGaussianProcess(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        backbone: nn.Module = nn.Identity(),
        n_inducing: int = 1024,
        momentum: float = 0.9,
        ridge_penalty: float = 1e-6,
        activation: nn.Module = Cos(),
        verbose: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_inducing = n_inducing
        self.momentum = momentum
        self.ridge_penalty = ridge_penalty
        self.verbose = verbose
        self.backbone = backbone

        # Random Fourier features (RFF) layer
        projection = nn.Linear(in_features, n_inducing)
        projection.weight.requires_grad_(False)
        projection.bias.requires_grad_(False)

        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L96
        nn.init.kaiming_normal_(projection.weight, a=math.sqrt(5))
        nn.init.uniform_(projection.bias, 0, 2 * math.pi)

        self.rff = nn.Sequential(
            OrderedDict(
                [
                    ("backbone", backbone),
                    ("projection", projection),
                    ("activation", activation),
                ]
            )
        )

        # Weights for RFF
        self.weight = nn.Linear(n_inducing, out_features, bias=False)
        # Should be normally distributed a priori
        nn.init.kaiming_normal_(self.weight.weight, a=math.sqrt(5))

        self.pipeline = nn.Sequential(self.rff, self.weight)

        # RFF precision and covariance matrices
        self.is_fitted = False
        self.covariance = Parameter(
            1 / self.ridge_penalty * torch.eye(self.n_inducing),
            requires_grad=False,
        )
        # Ridge penalty is used to stabilize the inverse computation
        self.precision_initial = self.ridge_penalty * torch.eye(
            self.n_inducing, requires_grad=False
        )
        self.precision = Parameter(
            self.precision_initial,
            requires_grad=False,
        )

    def forward(
        self,
        X: torch.Tensor,
        with_variance: bool = False,
        update_precision: bool = False,
    ):
        features = self.rff(X).detach()

        if update_precision:
            self.update_precision_(features)

        logits = self.weight(features)
        if not with_variance:
            return logits
        else:
            if not self.is_fitted:
                raise ValueError(
                    "`compute_covariance` should be called before setting "
                    "`with_variance` to True"
                )
            with torch.no_grad():
                variances = torch.bmm(
                    features[:, None, :],
                    (features @ self.covariance)[:, :, None],
                ).reshape(-1)

            return logits, variances

    def reset_precision(self):
        self.precision[...] = self.precision_initial.detach()

    def update_precision_(self, features: torch.Tensor):
        with torch.no_grad():
            if self.momentum < 0:
                # Use this to compute the precision matrix for the whole
                # dataset at once
                self.precision[...] = self.precision + features.T @ features
            else:
                self.precision[...] = (
                    self.momentum * self.precision
                    + (1 - self.momentum) * features.T @ features
                )

    def update_precision(self, X: torch.Tensor):
        with torch.no_grad():
            features = self.rff(X)
            
            features_list = [torch.zeros_like(features) for i in range(int(os.environ['WORLD_SIZE']))]
            torch.distributed.all_gather(features_list, features)
            features = torch.cat(features_list)

            self.update_precision_(features)

    def update_covariance(self):
        if not self.is_fitted:
            self.covariance[...] = (
                self.ridge_penalty * self.precision.cholesky_inverse()
            )
            self.is_fitted = True

    def reset_covariance(self):
        self.is_fitted = False
        self.covariance.zero_()


def dino_model():
    os.environ['TORCH_HOME'] = './'
    os.environ['TORCH_HUB'] = './'
    # DINOv2 vit-s (14) with registers
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
    # state = model.state_dict()
    # mymodel = vit_small(14, 4)
    # mymodel.load_state_dict(state)
    model.eval()

    return model.to('cpu')



def get_uncertainty(ds, model, batch_size=1024, verbose=True):
    dataloader = DataLoader(ds, batch_size=batch_size)
    uncs = []

    for (X, y) in tqdm(dataloader, total=len(dataloader), disable=not verbose):
        with torch.no_grad():
            _, unc = model(X, with_variance=True)
        uncs.append(unc.cpu())

    uncs = torch.concat(uncs)
    uncs = (uncs - uncs.min())
    uncs = (uncs / uncs.max()) ** 0.5
    return uncs.detach().cpu()


def dino_RFGP(num_classes=3):
    m = torch.nn.Sequential(dino_model(), torch.nn.BatchNorm1d(384)) # linear = torch.nn.Linear(384, num_classes)
    
    linear = RandomFeatureGaussianProcess(
                    in_features=384,
                    out_features=num_classes,
                    backbone=m,
                    n_inducing=1024,
                    momentum = 0.9,
                    ridge_penalty = 1e-6,
                    activation = Cos(),
                    verbose = False,
                )
    
    return linear


def remove_gaps(arr, remove=1):
    # remove the largest gap between numerical values in a 1-d array towards the heavier side
    s, _ = torch.sort(arr)
    a1 = torch.cat(s, torch.zeros(1,))
    a2 = torch.cat(torch.zeros(1,), s)
    # each coordinate represents the distance between this coordinate and the next
    gaps = (a2 - a1)[1:-1]
    order = torch.argsort(gaps, descending=True)

    for i in range(remove):
        coord = order[i]
        offset = gaps[coord]
        if coord > a1.shape[0] // 2:
            arr -= torch.where(arr > s[coord]).type(torch.float32) * offset
        else:
            arr += torch.where(arr < s[coord]).type(torch.float32) * offset

    return arr


class LinearProbe(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.m = dino_model()
        self.linear = torch.nn.Linear(384, num_classes)
    
    def forward(self, x):
        x = self.m(x).detach()
        return self.linear(x)