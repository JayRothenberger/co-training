from itertools import zip_longest

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from copy import deepcopy as copy

from utils import cascade_round, create_sampler_loader, create_loader, k_model, add_to_imagefolder, create_samplers_loaders, EarlyStopper


def train_ddp(rank, device, epoch, model, loader, loss_fn, optimizer):
    ddp_loss = torch.zeros(3).to(device)
    model.train()
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X, update_precision=True)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += (output.argmax(1) == y).type(torch.float).sum().item()
        ddp_loss[2] += len(X)
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    train_acc = ddp_loss[1] / ddp_loss[2] 
    train_loss = ddp_loss[0] / ddp_loss[2]

    if rank == 0:
        print('Train Epoch: {} \tAccuracy: {:.2f}% \tAverage Loss: {:.6f}'
              .format(epoch, 
                      100*(ddp_loss[1] / ddp_loss[2]), 
                      ddp_loss[0] / ddp_loss[2]))

    return train_acc, train_loss 


def test_ddp(rank, device, model, loader, loss_fn):
    ddp_loss = torch.zeros(3).to(device)
    model.eval()
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = loss_fn(output, y)
            ddp_loss[0] += loss.item()
            ddp_loss[1] += (output.argmax(1) == y).type(torch.float).sum().item()
            ddp_loss[2] += len(X)
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    test_acc = ddp_loss[1] / ddp_loss[2] 
    test_loss = ddp_loss[0] / ddp_loss[2]

    if rank == 0:
        print('Test error: \tAccuracy: {:.2f}% \tAverage loss: {:.6f}'
              .format(100*test_acc, test_loss))

    return test_acc, test_loss


def predict_ddp(world_size, device, model, loader, batch_size, num_classes, with_variance=False):
    model.eval()
    predictions = []
    labels = []
    variances = []

    with torch.no_grad():
        for X, y in loader:
            tensor_list = [torch.full((batch_size, num_classes), -99,
                                      dtype=torch.float16).to(device) 
                           for _ in range(world_size)]
            
            var_list = [torch.full((batch_size,), -99,
                                      dtype=torch.float32).to(device) 
                           for _ in range(world_size)]
            if with_variance:
                output, variance = model(X.to(device), with_variance=True)
            else:
                output = model(X.to(device))

            # pad the output so that it contains the same 
            # number of rows as specified by the batch size
            pad = torch.full((batch_size, num_classes), -1, 
                             dtype=torch.float16).to(device)
            pad[:output.shape[0]] = output

            # same as above but we need the labels
            label_list = [torch.full((batch_size,), -1, 
                                     dtype=torch.float16).to(device)
                                     for _ in range(world_size)]

            pad2 = torch.full((batch_size,), -1, 
                               dtype=torch.float16).to(device)
            pad2[:y.shape[0]] = y

            if with_variance:
                pad3 = torch.full((batch_size,), -1, 
                                dtype=torch.float32).to(device)
                pad3[:variance.shape[0]] = variance

            # all-gather the full list of predictions across all processes
            dist.all_gather(tensor_list, pad)
            batch_outputs = torch.cat(tensor_list)

            # all-gather the full list of labels
            dist.all_gather(label_list, pad2)
            batch_labels = torch.cat(label_list)

            if with_variance:
                dist.all_gather(var_list, pad3)
                batch_variance = torch.cat(var_list)

            # remove all rows of the tensor that contain a -1
            # (as this is not a valid value anywhere)
            mask = ~(batch_outputs == -99).any(-1)
            if mask.shape[0] < batch_size:
                print(mask)
            batch_outputs = batch_outputs[mask]
            predictions.append(batch_outputs)

            batch_labels = batch_labels[mask]
            labels.append(batch_labels)

            if with_variance:
                batch_variance = batch_variance[mask]
                variances.append(batch_variance)
    if with_variance:
        return torch.cat(predictions), torch.cat(labels), torch.cat(variances)
    else:
        return torch.cat(predictions), torch.cat(labels)


def c_test(rank, model0, model1, loader0, loader1, device):
    ddp_loss = torch.zeros(3).to(device)
    model0.eval()
    model1.eval()
    with torch.no_grad():
        for batch, ((X0, y0), (X1, y1)) in enumerate(zip(iter(loader0), iter(loader1))):
            X0, y0 = X0.to(device), y0.to(device)
            X1, y1 = X1.to(device), y1.to(device)

            output = torch.nn.Softmax(-1)(
                torch.clamp(model0(X0) + model1(X1), -(2**14), 2**14)
            )

            ddp_loss[1] += (output.argmax(1) == y0).type(torch.float).sum().item()
            ddp_loss[2] += len(X0)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    test_acc = ddp_loss[1] / ddp_loss[2]

    print(f"ddp_loss1: {ddp_loss[1]}, ddp_loss2: {ddp_loss[2]}")

    if rank == 0:
        print("Test error: \tCo-Accuracy: {:.2f}%".format(100 * test_acc))

    return test_acc

def get_topk(prediction, u, k, frequencies):
    prob, label = torch.max(prediction, 1)

    unique, counts = np.unique(label.cpu().numpy(), return_counts=True)

    idx = [[] for l in unique]
    u = u.squeeze()
    print('u', u)

    hist, edges = np.histogram(u.clone().cpu().numpy())
    print(hist, edges)

    dec_conf = torch.argsort(u, descending=False)
    # ascending uncertainty
    for i in dec_conf:
        for unq, l in enumerate(unique):
            # append the label to the appropriate list (could skip this for if label == idx)
            if label[i] == l:
                idx[unq].append(i)
                
    idx = [torch.tensor(inds) for inds in idx]
    
    # stratified balance
    ratios = (frequencies / frequencies.sum())
    # the percent of observations we can use for each class
    count_per_class = ratios * k
    
    idxs = []
    print(f'gathering k... {k}')
    while len(idxs) <= k:
        idxs = torch.cat([idx[l][:cascade_round(count_per_class)[l]] for l in range(len(unique))])
        count_per_class += (1 / frequencies.shape[0])

    idxs = idxs[:k]
    print('idx shape', idxs.shape)

    unique, counts = np.unique(label[idxs].cpu().numpy(), return_counts=True)
    print(counts, frequencies)

    return prob[idxs], label[idxs], idxs


def get_k(prediction, u, epsilon, frequencies):
    prob, label = torch.max(prediction, 1)


    unique, counts = np.unique(label.cpu().numpy(), return_counts=True)

    # ok, but this is not exactly n% we will have some rounding to do here
    idx = [[] for l in unique]
    u = u.squeeze()
    print(u.min(), u.max())

    print('u shape', u.shape)

    dec_conf = torch.argsort(u, descending=False)
    # ascending uncertainty
    for i in dec_conf:
        # for each label
        if u[i] > epsilon:
            break
        for unq, l in enumerate(unique):
            # append the label to the appropriate list (could skip this for if label == idx)
            if label[i] == l:
                idx[unq].append(i)
                
    idx = [torch.tensor(inds) for inds in idx]
    # stratified balance
    ratios = (frequencies / frequencies.sum())
    # normalize by the balance
    print(frequencies)
    print(ratios)
    print(torch.tensor([inds.shape[0] for inds in idx]))
    k = int((torch.tensor([inds.shape[0] for inds in idx]) / ratios).min())
    print('k', k)
    return k


# some models may end training earlier than others,
# but we would like to have training logged on the same step
def merge_wandb_logs(iteration, epochs, wandb_logs) -> list:
    logs_with_steps = []
    padded_logs = list(zip_longest(*wandb_logs, fillvalue=None))
    for epoch, logs in enumerate(padded_logs):
        log_merged = {}
        for log in logs:
            if log is None: log = {}
            log_merged = log_merged | log
        step = epoch + iteration * epochs
        logs_with_steps.append((log_merged, step))
    return logs_with_steps


class CoTrainingModel:
    def __init__(self, 
                 rank: int, 
                 world_size: int, 
                 models: list) -> None:
        self.rank = rank
        self.world_size = world_size
        self.models = models
        self.logs = []
        self.frequencies = np.array([])
        self.epsilon = 1.0

    def predict(self, device, unlbl_views, num_classes, batch_size, softmax_logits=False):
        samplers_unlbl = []
        loaders_unlbl = []
        for i in range(len(unlbl_views)):
            sampler_unlbl, loader_unlbl = create_sampler_loader(self.rank, 
                                                                self.world_size,
                                                                unlbl_views[i],
                                                                batch_size,
                                                                shuffle=False)
            samplers_unlbl.append(sampler_unlbl)
            loaders_unlbl.append(loader_unlbl)

        if self.rank == 0:
            print("making predictions on unlabeled set...")

        # make predictions for everything in unlabeled set, for all models
        preds_softmax = []
        for i, model in enumerate(self.models):
            preds, labels = predict_ddp(self.world_size, device, 
                               model, loaders_unlbl[i], 
                               batch_size, 
                               num_classes, 
                               softmax_logits=softmax_logits)
            preds = preds[:len(unlbl_views[i])]
            labels = labels[:(len(unlbl_views[i]))]
            # ensure all processes agree on the list of predictions
            dist.broadcast(preds, 0)
            preds_softmax.append(preds)
        dist.broadcast(labels, 0)

        if self.rank == 0:
            print(f"shape of predictions {preds.shape} of labels {labels.shape}")

        # num views x num instances x num classes
        preds_softmax = torch.stack(preds_softmax)

        return preds_softmax, labels

    def predict_uncertainty(self, device, unlbl_views, num_classes, batch_size):

        samplers_unlbl = []
        loaders_unlbl = []
        for i in range(len(unlbl_views)):
            loader_unlbl = create_loader(self.rank, 
                                        self.world_size,
                                        unlbl_views[i],
                                        batch_size,
                                        shuffle=False)
            loaders_unlbl.append(loader_unlbl)

        us = []
        preds = []
        labels = []

        for model, loader in zip(self.models, loaders_unlbl):
            uncs = []
            pred = []
            labs = []

            with torch.no_grad():
                for X, y in loader:
                    output, variance = model(X.to(device), with_variance=True)
                    uncs.append(variance)
                    pred.append(output)
                    labs.append(y)

            for u in uncs:
                try:
                    assert len(u.shape)
                except Exception as e:
                    print(f'expected unc to have non-zero number of dimensions, found {u.shape}')
                    u.unsqueeze(0)

            uncs = torch.cat(uncs)
            pred = torch.cat(pred)
            dist.broadcast(uncs, 0)
            dist.broadcast(pred, 0)
            labels = torch.cat(labs)

            uncs.type(torch.float32)

            uncs = uncs - uncs.min()
            uncs = torch.nn.ReLU()(uncs)

            uncs = (uncs / uncs.max()) ** 0.5

            us.append(uncs)
            preds.append(torch.nn.Softmax(-1)(pred))
            

        return torch.stack(preds)[:,:len(unlbl_views[0])], us[:len(unlbl_views[0])], labels[:len(unlbl_views[0])]

    def update(self,
               preds_softmax: torch.Tensor,
               u: list,
               train_views: list, 
               unlbl_views: list,
               k_total: float) -> None:
        msg = (f'number of views and number of models must be the same -- '
               f'got train: {len(train_views)}, unlabeled: {len(unlbl_views)}, '
               f'models: {len(self.models)}')
        assert len(train_views) == len(unlbl_views) == len(self.models), msg

        # assuming that k is # of total dataset to bring in
        # we'll have each model take in top-(k // len(models)) predictions
        # (though we may bring in less than this.) (?)
        if self.rank == 0:
            print("updating datasets...")

        # if only 2 views, we'll filter out conflicting predictions
        # and then update the datasets by swapping labels
        if len(self.models) == 2:
            lbls_topk = []
            idxes_topk = []
            ks = []

            for idz, pred, in enumerate(preds_softmax):
                k = get_k(pred,
                        u[idz],
                        self.epsilon,
                        self.frequencies)
                ks.append(k)
            k = min(ks)
            print(k)
            for idz, pred in enumerate(preds_softmax):
                _, lbl_topk, idx_topk = get_topk(pred,
                                                 u[idz],
                                                 k if k <= len(pred) else len(pred),
                                                 self.frequencies)
                lbls_topk.append(lbl_topk.detach().cpu().numpy())
                idxes_topk.append(idx_topk.detach().cpu().numpy())

            self._resolve_conflicts_twoview(lbls_topk, idxes_topk)
            self._update_datasets_twoview(lbls_topk, idxes_topk,
                                         train_views, unlbl_views)
        # if views > 2, aggregate the predictions and call it a day
        else:
            self._update_datasets_multiview(preds_softmax, k_model, 
                                            train_views, unlbl_views)

    # 2 views, no ensembling
    def _resolve_conflicts_twoview(self, lbls_topk, idxes_topk):
        msg = (f'expected predictions for 2 views, got: '
               f'labels {len(lbls_topk)}, indices {len(idxes_topk)}')
        assert len(lbls_topk) == len(idxes_topk) == 2, msg

        inter, idx_inter0, idx_inter1 = np.intersect1d(idxes_topk[0],
                                                       idxes_topk[1],
                                                       return_indices=True)
        
        mask_colls = lbls_topk[0][idx_inter0] != lbls_topk[1][idx_inter1]
        idx_colls = inter[mask_colls]

        if len(idx_colls) > 0:
            if self.rank == 0:
                print(f"number of conflicting predictions: {len(idx_colls)}")
            idx_coll0 = idx_inter0[mask_colls]
            idx_coll1 = idx_inter1[mask_colls]
            
            mask = np.ones(len(idxes_topk[0]), dtype=bool)
            mask[idx_coll0] = False
            mask[idx_coll1] = False

            for i in range(len(lbls_topk)):
                lbls_topk[i] = lbls_topk[i][mask]
                idxes_topk[i] = idxes_topk[i][mask]

    def _update_datasets_twoview(self, lbls_topk, idxes_topk,
                                train_views, unlbl_views):
        samples_unlbl0 = np.stack([np.array(a) 
                                   for a in unlbl_views[0].samples])
        samples_unlbl1 = np.stack([np.array(a) 
                                   for a in unlbl_views[1].samples])

        # retrieve the instances that have been labeled 
        # with high confidence by the other model
        list_samples0 = [(str(a[0]), int(a[1])) 
                         for a in list(samples_unlbl0[idxes_topk[1]])]
        list_samples1 = [(str(a[0]), int(a[1])) 
                         for a in list(samples_unlbl1[idxes_topk[0]])]
        
        # image paths for both
        paths0 = [i for i, _ in list_samples0]
        paths1 = [i for i, _ in list_samples1]

        # update imagefolders
        train_views[0] = add_to_imagefolder(paths0, lbls_topk[1].tolist(), 
                                            train_views[0])
        train_views[1] = add_to_imagefolder(paths1, lbls_topk[0].tolist(), 
                                            train_views[1])
        
        unique, counts = np.unique(np.array([y for (x, y) in train_views[0].samples]), return_counts=True)
        print(unique, counts)
        self.frequencies += counts
        unique, counts = np.unique(np.array([y for (x, y) in train_views[1].samples]), return_counts=True)
        print(unique, counts)
        self.frequencies += counts

        # remove instances from unlabeled dataset
        mask = np.ones(len(unlbl_views[0]), dtype=bool)
        for idx_topk_i in idxes_topk:
            mask[idx_topk_i] = False
        
        if self.rank == 0:
            print(f"number of unlabeled instances to remove: {(~mask).sum()}")

        samples_unlbl0 = samples_unlbl0[mask]
        samples_unlbl1 = samples_unlbl1[mask]

        list_unlbl0 = [(str(a[0]), int(a[1])) for a in list(samples_unlbl0)]
        list_unlbl1 = [(str(a[0]), int(a[1])) for a in list(samples_unlbl1)]

        unlbl_views[0].samples = list_unlbl0
        unlbl_views[1].samples = list_unlbl1

        if self.rank == 0:
            print(f"remaining number of unlabeled instances: {len(unlbl_views[0])}")

    def _update_datasets_multiview(self, k_model, preds_softmax,
                                  train_views, unlbl_views):
        num_views = len(self.models)
        num_instances = preds_softmax.shape[1]
        mask = np.ones(num_instances, dtype=bool)

        for i in range(num_views):
            samples_unlbl = np.stack([np.array(a)
                                      for a in unlbl_views[i].samples])
            
            # mask out softmax predictions to exclude "student" predictions
            mask_preds = torch.ones(num_views, dtype=torch.bool)
            mask_preds[i] = False
            preds_agg = torch.prod(preds_softmax[mask_preds], dim=0) / num_views
            _, lbl_topk, idx_topk = get_topk(preds_agg, k_model 
                                             if k_model <= num_instances
                                             else num_instances, self.frequencies)

            # training set update
            list_samples = [str(a[0], int(a[1]))
                            for a in list(samples_unlbl[idx_topk])]
            paths = [i for i, _ in list_samples]
            train_views[i] = add_to_imagefolder(paths, lbl_topk.tolist(), 
                                                train_views[i])

            # update mask to remove now-pseudolabeled instances (later)
            mask[idx_topk] = False

        # remove pseudolabeled instances from unlabeled set
        for i in range(num_views):
            samples_unlbl = np.stack([np.array(a) 
                                      for a in unlbl_views[i].samples])
            samples_unlbl = samples_unlbl[mask]
            list_unlbl = [(str(a[0]), int(a[1])) for a in list(samples_unlbl)]
            unlbl_views[i].samples = list_unlbl
        
        if self.rank == 0:
            print(f"number of unlabeled instances to remove: {(~mask).sum()}")
            print(f"remaining number of unlabeled instances: {len(unlbl_views[0])}")

    # one full pass of co-training without dataset updates
    def train(self,
              device: torch.device,
              iteration: int, 
              epochs: int,
              states: dict,
              train_views: list,
              val_views: list,
              test_views: list,
              batch_size: int = 64,
              test_batch_size: int = 64,
              optimizer: torch.optim.Optimizer = Adam,
              optimizer_kwargs: dict = {'lr':1e-3,
                                        'momentum': 0.9},
              stopper_kwargs: dict = {'metric': 'accuracy',
                                      'patience': 32,
                                      'min_delta': 1e-3},
              lr_scheduler: torch.optim.lr_scheduler = ReduceLROnPlateau) -> None:
        msg = (f'number of views and number of models must be the same -- '
               f'got train: {len(train_views)}, val: {len(val_views)}, '
               f'test: {len(test_views)}, models: {len(self.models)}')
        assert len(train_views) == len(val_views) == len(test_views) == len(self.models), msg

        optimizers = []
        schedulers = []
        stoppers = []

        for i in range(len(self.models)):
            stoppers.append(EarlyStopper(**stopper_kwargs))
            optimizers.append(optimizer(self.models[i].parameters(), 
                                        **optimizer_kwargs))
            # first iteration won't have any optimizer states in dict (...)
            if f'optimizer{i}_state' not in states:
                states[f'optimizer{i}_state'] = optimizers[i].state_dict()

        if lr_scheduler is not None:
            for i, opt in enumerate(optimizers):
                schedulers.append(lr_scheduler(opt))
        else:
            for i, opt in enumerate(optimizers):
                schedulers.append(ReduceLROnPlateau(opt, 'max', factor=0.5, patience=10))

        samplers_train, loaders_train = create_samplers_loaders(self.rank, self.world_size,
                                                                train_views, batch_size, 
                                                                persistent_workers=True)
        
        samplers_val, loaders_val = create_samplers_loaders(self.rank, self.world_size,
                                                            val_views, test_batch_size, 
                                                            persistent_workers=True)
        
        samplers_test, loaders_test = create_samplers_loaders(self.rank, self.world_size,
                                                              test_views, test_batch_size, 
                                                              persistent_workers=True)

        # loss_fn = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        best_val_loss = float('inf')
        iteration_logs = []
        for i in range(len(self.models)):
            best_val_acc = 0.0
            best_val_loss = float('inf')
            weights=-1*torch.log(torch.tensor(self.frequencies / (self.frequencies.sum() * 2)))
            loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(torch.cuda.current_device()).type(torch.float32))
            if self.rank == 0:
                print(f"training view{i}...")
            step = 0 * iteration
            model_logs = []
            for epoch in range(epochs):
                step += 1
                samplers_train[i].set_epoch(epoch)
                train_acc, train_loss = train_ddp(self.rank, device, epoch, 
                                               self.models[i], loaders_train[i], 
                                               loss_fn, optimizers[i])
                val_acc, val_loss = test_ddp(self.rank, device, self.models[i], 
                                         loaders_val[i], loss_fn)
                test_acc, test_loss = test_ddp(self.rank, device, self.models[i],
                                               loaders_test[i], loss_fn)

                stoppers[i].step(val_acc, val_loss)
                if stoppers[i].epochs_since_improvement == 0 and (stoppers[i].best_val_acc > states[f'model{i}_best_acc']):
                    states[f'model{i}_state'] = copy(self.models[i].state_dict())
                    states[f'optimizer{i}_state'] = optimizers[i].state_dict()
                    best_val_acc = max(best_val_acc, stoppers[i].best_val_acc)
                    best_val_loss = min(best_val_loss, stoppers[i].best_val_loss)
                    states[f'model{i}_best_acc'] = max(states[f'model{i}_best_acc'], stoppers[i].best_val_acc)
                    states[f'model{i}_best_loss'] = min(states[f'model{i}_best_loss'], stoppers[i].best_val_loss)
                    print('best val acc', best_val_acc)
                
                model_logs.append({f'train_acc{i}': train_acc,
                                  f'train_loss{i}': train_loss,
                                  f'val_acc{i}': val_acc,
                                  f'val_loss{i}': val_loss,
                                  f'test_acc{i}': test_acc,
                                  f'test_loss{i}': test_loss,
                                  f'best_val_acc{i}': stoppers[i].best_val_acc,
                                  f'best_val_loss{i}': stoppers[i].best_val_loss
                                  })

                if stoppers[i].early_stop:
                    break
                
                schedulers[i].step(val_loss)
                loss_fn.step_epoch()
                loss_fn.step_annealing()

            # need to manually shut down the workers as they will persist otherwise
            loaders_train[i]._iterator._shutdown_workers()
            loaders_val[i]._iterator._shutdown_workers()
            loaders_test[i]._iterator._shutdown_workers()

            iteration_logs.append(model_logs)
            self.models[i].module.update_covariance()


        self.logs += merge_wandb_logs(iteration, epochs, iteration_logs)
    
        return best_val_acc, best_val_loss
