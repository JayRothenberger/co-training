import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from torchvision.models import resnet50

import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

import numpy as np

import argparse
import functools
import pickle
import copy
import random
from math import floor

from utils import add_to_imagefolder, train_test_split_samples

# takes in a Tensor of shape e.g. (# instances, # prob outputs) and returns a tuple
# (Tensor[top probabilities], Tensor[predicted labels], Tensor[instance indexes])
def get_topk_pred(pred, k):
    prob, label = torch.max(pred, 1)
    idx = torch.argsort(prob, descending=True)[:k]
    return prob[idx].cpu(), label[idx].cpu(), idx.cpu()

def remove_collisions(lbl_model0, lbl_model1, idx_model0, idx_model1):
    # find instances and indices of instances that have
    # been labeled as most confident by both model0, model1
    inter, idx_inter0, idx_inter1 = np.intersect1d(
                                        idx_model0,
                                        idx_model1,
                                        return_indices=True)

    print(f"Number of predictions (model0): {len(idx_model0)}")
    print(f"Number of predictions (model1): {len(idx_model1)}")
    print(f"Found {len(inter)} potential conflicting predictions")

    # bool mask to identify the conflicting predictions (collision)
    mask_coll = lbl_model0[idx_inter0] != lbl_model1[idx_inter1]
    collisions = inter[mask_coll]

    print(f"Found {len(collisions)} conflicting predictions")

    if (len(collisions) > 0):
        print(f"Collisions: {collisions}")
        # find where these collisions are actually at
        # in their respective lists, and remove them...
        # (maybe want to return this as well? ...)
        idx_coll0 = idx_inter0[mask_coll]
        idx_coll1 = idx_inter1[mask_coll]

        # masks to remove the instances with conflicting predictions
        mask0 = np.ones(len(idx_model0), dtype=bool)
        mask0[idx_coll0] = False
        mask1 = np.ones(len(idx_model1), dtype=bool)
        mask1[idx_coll1] = False

        lbl_model0 = lbl_model0[mask0]
        lbl_model1 = lbl_model1[mask1]
        idx_model0 = idx_model0[mask0]
        idx_model1 = idx_model1[mask1]

    return lbl_model0, lbl_model1, idx_model0, idx_model1

def predict(loader, model, rank):
    model.eval()
    predictions = []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(rank % torch.cuda.device_count()), y.to(rank % torch.cuda.device_count())
            output = model(X)
            predictions.append(output)
    return torch.cat(predictions) # output shape (# instances, # outputs)

# train two models on two different views
# then add top k% of predictions on the unlabeled set
# to the labeled datasets
def cotrain(loader0, loader1, loader_unlbl,
            model0, model1, k, device):

    pred_model0 = predict(loader_unlbl, model0, device)
    pred_model1 = predict(loader_unlbl, model1, device)

    # get top-k predictions (labels, instance indexes in the dataset)
    _, lbl_topk0, idx_topk0 = get_topk_pred(
                                    pred_model0,
                                    k if k <= len(pred_model0) 
                                    else len(pred_model0))
    _, lbl_topk1, idx_topk1 = get_topk_pred(
                                    pred_model1, 
                                    k if k <= len(pred_model1) 
                                    else len(pred_model1))

    print(f"Number of unlabeled instances: {len(loader_unlbl.dataset)}")

    # what if two models predict confidently on the same instance?
    # find and remove conflicting predictions from the lists
    # (may want to return the indices of the collisions too...?)
    lbl_topk0, lbl_topk1, idx_topk0, idx_topk1 = \
    remove_collisions(lbl_topk0, lbl_topk1, idx_topk0, idx_topk1)

    # convert from list to array for the convenient numpy indexing
    samples_unlbl = np.stack([np.array(a) for a in loader_unlbl.dataset.samples])
    list_samples0 = [(str(a[0]), int(a[1])) for a in list(samples_unlbl[idx_topk0])]
    list_samples1 = [(str(a[0]), int(a[1])) for a in list(samples_unlbl[idx_topk1])] 

    paths0 = [i for i, _ in list_samples0]
    paths1 = [i for i, _ in list_samples1]

    # add pseudolabeled instances to the labeled datasets
    loader0.dataset.samples = add_to_imagefolder(paths1, list(lbl_topk1), loader0.dataset)
    loader1.dataset.samples = add_to_imagefolder(paths0, list(lbl_topk0), loader1.dataset)

    # remove instances from unlabeled dataset
    mask_unlbl = np.ones(len(loader_unlbl.dataset), dtype=bool)
    mask_unlbl[idx_topk0] = False
    mask_unlbl[idx_topk1] = False
    print(f"Number of unlabeled instances to remove: {(~mask_unlbl).sum()}")
    samples_unlbl = samples_unlbl[mask_unlbl]
    list_unlbl = [(str(a[0]), int(a[1])) for a in list(samples_unlbl)]
    loader_unlbl.dataset.samples = list_unlbl

def train(args, rank, world_size, loader, model, optimizer, epoch,
          sampler=None):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    if sampler:
        sampler.set_epoch(epoch)
    ddp_loss = torch.zeros(3).to(rank % torch.cuda.device_count())
    for batch, (X, y) in enumerate(loader):
        X, y = X.to(rank % torch.cuda.device_count()), y.to(rank % torch.cuda.device_count())
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += (output.argmax(1) == y).type(torch.float).sum().item()
        ddp_loss[2] += len(batch)
    
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        print('Train Epoch: {} \tAccuracy: {:.2f}% \tLoss: {:.6f}'
              .format(epoch, 
                      100*(ddp_loss[1] / ddp_loss[2]), 
                      ddp_loss[0] / ddp_loss[2]))


def test(args, rank, world_size, loader, model):
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    ddp_loss = torch.zeros(3).to(rank % torch.cuda.device_count())
    for batch, (X, y) in enumerate(loader):
        X, y = X.to(rank % torch.cuda.device_count()), y.to(rank % torch.cuda.device_count())
        output = model(X)
        loss = loss_fn(output, y)
        ddp_loss[0] += loss.item()
        ddp_loss[1] += (output.argmax(1) == y).type(torch.float).sum().item()
        ddp_loss[2] += len(batch)
    
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        print('Test error: \tAccuracy: {:.2f}% \tAverage loss: {:.6f}'
              .format(100*(ddp_loss[1] / ddp_loss[2]), 
                      ddp_loss[0] / ddp_loss[2]))

def training_process(args, rank, world_size):
    with open('cotraining_samples_lists_fixed.pkl', 'rb') as fp:
        data = pickle.load(fp)

    # split samples into labeled, unlabeled (25/75 split)
    samples_train0, samples_train1, samples_unlbl0, samples_unlbl1 = \
    train_test_split_samples(data['labeled'], data['inferred'],
                             test_size=0.75, random_state=13)
    
    # split the data so we get 70/10/20 train/val/test
    samples_train0, samples_train1, samples_test0, samples_test1 = \
        train_test_split_samples(samples_train0, samples_train1,
                                 test_size=0.2, random_state=13)

    samples_train0, samples_train1, samples_val0, samples_val1 = \
        train_test_split_samples(samples_train0, samples_train1,
                                 test_size=0.125, random_state=13)

    # ResNet50 wants 224x224 images
    trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
    ])

    # Create ImageFolder objects/datasets for first view
    data_train0 = datasets.ImageFolder('/ourdisk/hpc/ai2es/jroth/data/labeled', transform=trans)
    data_train0.class_to_idx = data['class_map']
    data_train0.classes = list(data['class_map'].keys())

    data_unlbl0 = copy.deepcopy(data_train0)
    data_unlbl0.samples = samples_unlbl0
    data_val0 = copy.deepcopy(data_train0)
    data_val0.samples = samples_val0
    data_test0 = copy.deepcopy(data_train0)
    data_test0.samples = samples_test0

    # Create ImageFolder objects/datasets for second view
    data_train1 = copy.deepcopy(data_train0)
    data_train1.root = '/ourdisk/hpc/ai2es'
    
    data_unlbl1 = copy.deepcopy(data_train1)
    data_unlbl1.samples = samples_unlbl1
    data_val1 = copy.deepcopy(data_train1)
    data_val1.samples = samples_val1
    data_test1 = copy.deepcopy(data_train1)
    data_test1.samples = samples_test1

    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=10_000_000
    )

    torch.cuda.set_device(rank % torch.cuda.device_count())

    # TODO define some dictionaries to do dict unpacking

    batch_size = args.batch_size
    loader_train0 = DataLoader(data_train0, batch_size, False)
    loader_unlbl0 = DataLoader(data_unlbl0, batch_size, False)
    loader_val0 = DataLoader(data_val0, batch_size, False)
    loader_test0 = DataLoader(data_test0, batch_size, False)
    
    loader_train1 = DataLoader(data_train1, batch_size, False)
    loader_unlbl1 = DataLoader(data_unlbl1, batch_size, False)
    loader_val1 = DataLoader(data_val1, batch_size, False)
    loader_test1 = DataLoader(data_test1, batch_size, False)

    # device = torch.device("cuda" 
    #                       if torch.cuda.is_available()
    #                       else "cpu")
    # print(f"using {device}")

    model0, model1 = resnet50(), resnet50()

    model0.to(rank % torch.cuda.device_count())
    model1.to(rank % torch.cuda.device_count())
    
    optimizer0 = optim.SGD(model0.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer1 = optim.SGD(model1.parameters(), lr=args.lr, momentum=args.momentum)
    
    scheduler0 = ReduceLROnPlateau(optimizer0)
    scheduler1 = ReduceLROnPlateau(optimizer1)

    # TODO return dict of states, metrics

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(args, rank, world_size):

    setup(rank, world_size)

    states, metric = training_process(args, rank, world_size)

    if rank == 0:
        torch.save(states, '/home/scratch/tiffanyle/cotraining/state.pth')

    cleanup()
        
def create_parser():
    parser = argparse.ArgumentParser(description='co-training')
    
    parser.add_argument('-e', '--epochs', type=int, default=10, 
                        help='training epochs (default: 10)')
    parser.add_argument('-b', '--batch_size', type=int, default=64, 
                        help='batch size for training (default: 64)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3,
                        help='learning rate for SGD (default 1e-3)')
    parser.add_argument('-m', '--momentum', type=float, default=0.9,
                        help='momentum for SGD (default 0.9')
    #blah add more here
    
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    
    main(args)
    