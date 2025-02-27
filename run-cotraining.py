import argparse
import functools
import os
import shutil
import pickle
import random
from math import floor
import time
from PIL import Image, ImageFile
from copy import deepcopy as copy

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torchvision import transforms

import cotraining.metrics as metrics
import cotraining.recalibration as recalibration
from cotraining.ct_model import stacked_dino_ENS, CoTrainingModel, test_ddp, c_test
from cotraining.dahps import DistributedAsynchronousGridSearch
from cotraining.dahps import sync_parameters
from cotraining.utils import (
    train_test_split_views,
    progressive_supset_sample,
    create_imagefolder,
    create_sampler_loader,
    setup,
    cleanup
)

import wandb
import albumentations as A

ImageFile.LOAD_TRUNCATED_IMAGES = True


Atransforms = A.Compose(
    [
        A.augmentations.geometric.rotate.Rotate(limit=15, p=0.5),
        A.Perspective(scale=[0, 0.1], keep_size=False, fit_output=False, p=1),
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.GaussNoise(std_range=(2.5, 3.3), mean_range=(0.0, 0.0)),
        A.RandomToneCurve(scale=0.5, p=1),
        A.augmentations.dropout.coarse_dropout.CoarseDropout(),
    ]
)


def Atransforms_fn(img):
    return Atransforms(image=np.array(img))["image"]


def create_model(auto_wrap_policy, device, num_classes, random_state=None):
    if random_state is not None:
        torch.manual_seed(random_state)

    model = stacked_dino_ENS(num_classes).to(device)

    model = DDP(model, device_ids=[device], find_unused_parameters=True)

    return model


def training_process(args, rank, world_size):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open("cotraining_samples_lists_fixed.pkl", "rb") as fp:
        data = pickle.load(fp)

    views = [data["labeled"], data["inferred"]]

    samples_unlbl, samples_test = train_test_split_views(
        views, args.percent_test, random_state=args.seed
    )
    samples_train, samples_val, samples_unlbl = progressive_supset_sample(
        samples_unlbl,
        args.percent_unlabeled,
        args.percent_val,
        args.k,
        random_state=args.seed,
    )

    unique, counts = np.unique(
        np.array([y for (x, y) in samples_train[0] + samples_val[0]]),
        return_counts=True,
    )

    with open("/ourdisk/hpc/ai2es/jroth/unlabeled_samples_lists.pkl", "rb") as fp:
        data_u = pickle.load(fp)

    if rank == 0:
        os.mkdir(os.environ["LSCRATCH"] + "/data")
        shutil.copytree(
            "/ourdisk/hpc/ai2es/jroth/data/labeled",
            os.environ["LSCRATCH"] + "/data/labeled",
        )
        shutil.copy(
            "/ourdisk/hpc/ai2es/jroth/data/NYSDOT_m4er5dez4ab.tar.gz",
            os.environ["LSCRATCH"] + "/data/NYSDOT_m4er5dez4ab.tar.gz",
        )
        shutil.copy(
            "/ourdisk/hpc/ai2es/jroth/data/Skyline_6464.tar.gz",
            os.environ["LSCRATCH"] + "/data/Skyline_6464.tar.gz",
        )
        print("copied")
        os.system(
            f"tar -xzf {os.environ['LSCRATCH'] + '/data/NYSDOT_m4er5dez4ab.tar.gz'} -C {os.environ['LSCRATCH'] + '/data/'}"
        )
        os.system(
            f"tar -xzf {os.environ['LSCRATCH'] + '/data/Skyline_6464.tar.gz'} -C {os.environ['LSCRATCH'] + '/data/'}"
        )
        print("extracted")
        os.system("touch done.txt")
    else:
        while not os.path.isfile("done.txt"):
            time.sleep(10)
    # shutil.copy(Skyline_6464.tar.gz)

    for k, s in enumerate(samples_train):
        scratch = os.environ["LSCRATCH"] + "/"
        for i, (x, y) in enumerate(s):
            s[i] = (
                x.replace("/ourdisk/hpc/ai2es/jroth/", os.environ["LSCRATCH"] + "/"),
                y,
            )
        samples_train[k] = s

    for j, (l, i) in enumerate(zip(data_u["labeled"], data_u["inferred"])):
        print(j, end="\r")
        scratch = os.environ["LSCRATCH"] + "/"
        # scratch = '/ourdisk/hpc/ai2es/jroth/'
        # shutil.copy(l[0].replace('./', '/ourdisk/hpc/ai2es/jroth/'), l[0].replace('./', scratch))
        data_u["labeled"][j] = (l[0].replace("./", scratch), l[1])
        # shutil.copy(i[0].replace('./', '/ourdisk/hpc/ai2es/jroth/'), i[0].replace('./', scratch))
        data_u["inferred"][j] = (i[0].replace("./", scratch), i[1])

        # im = Image.open(l[0].replace('./', '/ourdisk/hpc/ai2es/jroth/'))
        # im.save(l[0].replace('./', '/ourdisk/hpc/ai2es/jroth/'))
        # im = Image.open(i[0].replace('./', '/ourdisk/hpc/ai2es/jroth/'))
        # im.save(i[0].replace('./', '/ourdisk/hpc/ai2es/jroth/'))

    samples_unlbl[0] += data_u["labeled"]
    samples_unlbl[1] += data_u["inferred"]

    assert len(samples_unlbl[0]) == len(samples_unlbl[1])

    # ResNet50 wants 224x224 images

    trans_train = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            Atransforms_fn,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    trans = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create ImageFolder objects for first view
    dummy_path = "/ourdisk/hpc/ai2es/jroth/data/labeled"
    data_train0 = create_imagefolder(data, samples_train[0], dummy_path, trans_train)
    data_unlbl0 = create_imagefolder(data, samples_unlbl[0], dummy_path, trans)
    data_val0 = create_imagefolder(data, samples_val[0], dummy_path, trans)
    data_test0 = create_imagefolder(data, samples_test[0], dummy_path, trans)

    # Create ImageFolder objects for second view (we will also update the root/path)
    new_path = "/ourdisk/hpc/ai2es"
    data_train1 = create_imagefolder(
        data, samples_train[1], dummy_path, trans_train, new_path
    )
    data_unlbl1 = create_imagefolder(
        data, samples_unlbl[1], dummy_path, trans, new_path
    )
    data_val1 = create_imagefolder(data, samples_val[1], dummy_path, trans, new_path)
    data_test1 = create_imagefolder(data, samples_test[1], dummy_path, trans, new_path)

    train_views = [data_train0, data_train1]
    copy_train_views = copy(train_views)
    unlbl_views = [data_unlbl0, data_unlbl1]
    copy_unlbl_views = copy(unlbl_views)
    val_views = [data_val0, data_val1]
    test_views = [data_test0, data_test1]

    num_classes = 3
    num_views = len(views)

    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=1_000_000
    )

    device = torch.device(rank % torch.cuda.device_count())
    torch.cuda.set_device(device)

    models = [
        create_model(auto_wrap_policy, device, num_classes) for _ in range(num_views)
    ]

    states = {
        "model0_state": models[0].state_dict(),
        "model1_state": models[1].state_dict(),
        "model0_best_acc": 0.0,
        "model0_best_loss": float("inf"),
        "model1_best_acc": 0.0,
        "model1_best_loss": float("inf"),
    }

    ct_model = CoTrainingModel(rank, world_size, models)
    ct_model.frequencies = counts

    if rank == 0:
        wandb.init(
            project="co-training-deep-ens-0075",
            entity="ai2es",
            name=f"Co-Training DINO ens",
            config={"args": vars(args)},
        )

    if rank == 0:
        print(vars(args))
        print(
            "Length of datasets:\n Train: {} \tUnlabeled: {} \tVal: {} \tTest: {}".format(
                len(samples_train[0]),
                len(samples_unlbl[0]),
                len(samples_val[0]),
                len(samples_test[0]),
            )
        )

    weights = -1 * torch.log(
        torch.nn.Softmax(-1)(torch.tensor(counts).type(torch.float32))
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(torch.cuda.current_device()).type(torch.float32))

    # calibration metrics
    ece_criterion = metrics.ECELoss()
    ace_criterion = metrics.ACELoss()
    mace_criterion = metrics.MACELoss()

    k = floor(len(data_unlbl0) * args.k)
    if rank == 0:
        print(f"k: {k}")

    best_val_acc = 0.0
    best_val_loss = float("inf")
    c_iter_logs = []
    for c_iter in range(args.cotrain_iters):
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(torch.cuda.current_device()).type(torch.float32))
        if len(data_unlbl0) == 0 and len(data_unlbl1) == 0:
            break
        if args.from_scratch and c_iter > 0:
            models = [
                create_model(auto_wrap_policy, device, num_classes)
                for _ in range(num_views)
            ]
            ct_model = CoTrainingModel(rank, world_size, models)
            ct_model.frequencies = counts

        if rank == 0:
            print(f"co-training iteration: {c_iter}")
            print(
                "train: {} unlabeled: {}".format(
                    len(train_views[0]), len(unlbl_views[0])
                )
            )

        train_kwargs = {
            "device": device,
            "iteration": c_iter,
            "epochs": args.epochs,
            "states": states,
            "train_views": train_views,
            "val_views": val_views,
            "test_views": test_views,
            "batch_size": args.batch_size,
            "test_batch_size": args.test_batch_size,
        }

        opt_kwargs = {"lr": args.learning_rate}

        stopper_kwargs = {
            "metric": args.stopping_metric,
            "patience": args.patience,
            "min_delta": args.min_delta,
        }

        best_val_acc_i, best_val_loss_i = ct_model.train(
            **train_kwargs, optimizer_kwargs=opt_kwargs, stopper_kwargs=stopper_kwargs
        )
        # update best val_acc
        best_val_acc = max(best_val_acc, best_val_acc_i)
        best_val_loss = min(best_val_loss, best_val_loss_i)

        ct_model.epsilon = 0.075

        print("epsilon", ct_model.epsilon)

        print("best val acc:", best_val_acc)

        # load best states for this iteration
        for i, model in enumerate(ct_model.models):
            model.load_state_dict(states[f"model{i}_state"])

        print(len(data_val0), len(data_val1))
        # no persistent workers as we're only iterating through it a few times from here
        sampler_val0, loader_val0 = create_sampler_loader(
            rank, world_size, data_val0, args.test_batch_size
        )
        sampler_test0, loader_test0 = create_sampler_loader(
            rank, world_size, data_test0, args.test_batch_size
        )
        sampler_val1, loader_val1 = create_sampler_loader(
            rank, world_size, data_val1, args.test_batch_size
        )
        sampler_test1, loader_test1 = create_sampler_loader(
            rank, world_size, data_test1, args.test_batch_size
        )

        # co-training val/test accuracy
        c_acc_val = c_test(rank, models[0], models[1], loader_val0, loader_val1, device)
        c_acc_test = c_test(
            rank, models[0], models[1], loader_test0, loader_test1, device
        )

        # TODO clean this a bit
        ct_model.models[0] = recalibration.ModelWithTemperature(models[0])
        ct_model.models[0].set_temperature(
            world_size, device, loader_val0, args.test_batch_size, num_classes
        )

        ct_model.models[1] = recalibration.ModelWithTemperature(models[1])
        ct_model.models[1].set_temperature(
            world_size, device, loader_val1, args.test_batch_size, num_classes
        )

        # prediction
        preds_softmax, u, labels = ct_model.predict_uncertainty(
            device, unlbl_views, num_classes, args.batch_size
        )

        # reset the datasets
        train_views = copy(copy_train_views)
        unlbl_views = copy(copy_unlbl_views)
        # update datasets
        ct_model.update(preds_softmax, u, train_views, unlbl_views, k)

        if rank == 0:
            print("testing after dataset update and calibration...")

        # test individual models after co-training update and calibration
        test_acc0, test_loss0 = test_ddp(
            rank, device, ct_model.models[0], loader_test0, loss_fn
        )
        test_acc1, test_loss1 = test_ddp(
            rank, device, ct_model.models[1], loader_test1, loss_fn
        )

        c_log = {
            "test_acc0": test_acc0,
            "test_loss0": test_loss0,
            "test_acc1": test_acc1,
            "test_loss1": test_loss1,
            "c_acc_val": c_acc_val,
            "c_acc_test": c_acc_test,
        }

        c_iter_logs += ct_model.logs
        c_iter_logs[-1] = ({**c_iter_logs[-1][0], **c_log}, c_iter_logs[-1][1])

    dist.barrier()

    # wandb.finish()

    return states, best_val_acc, c_iter_logs


def main(args, rank, world_size):
    setup(rank, world_size)

    search_space = ["k", "percent_unlabeled"]

    agent = sync_parameters(args, rank, search_space, DistributedAsynchronousGridSearch)

    args = agent.to_namespace(agent.combination)

    states, metric, logs = training_process(args, rank, world_size)

    if rank == 0:
        print("saving checkpoint")
        agent.save_checkpoint(states)
        for log, epoch in logs:
            wandb.log(log, step=epoch)
        wandb.finish()

    print("finishing combination")
    agent.finish_combination(float(metric))

    cleanup()


def create_parser():
    parser = argparse.ArgumentParser(description="co-training")

    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=512,
        help="training epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=128,
        help="batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "-tb",
        "--test_batch_size",
        type=int,
        default=128,
        help=" batch size for testing (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=1e-3,
        help="learning rate for SGD (default: %(default)s)",
    )
    parser.add_argument(
        "-m",
        "--momentum",
        type=float,
        default=0.9,
        help="momentum for SGD (default: %(default)s",
    )
    parser.add_argument(
        "-p",
        "--patience",
        type=float,
        default=64,
        help="number of epochs to train for without improvement (default: %(default)s)",
    )
    parser.add_argument(
        "-md",
        "--min_delta",
        type=float,
        default=1e-3,
        help="minimum delta for early stopping metric (default: %(default)s)",
    )
    parser.add_argument(
        "--cotrain_iters",
        type=int,
        default=100,
        help="max number of iterations for co-training (default: %(default)s)",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=[0.01],
        help="percentage of unlabeled samples to bring in each \
                            co-training iteration (default: 0.025)",
    )
    parser.add_argument(
        "--percent_unlabeled",
        type=float,
        default=[0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.96, 0.97, 0.98],
        help="percentage of unlabeled samples to start with (default: 0.9",
    )
    parser.add_argument(
        "--percent_test",
        type=float,
        default=0.2,
        help="percentage of samples to use for testing (default: %(default)s)",
    )
    parser.add_argument(
        "--percent_val",
        type=float,
        default=0.75,
        help="percentage of labeled samples to use for validation (default: %(default)s)",
    )
    parser.add_argument(
        "--stopping_metric",
        type=str,
        default="accuracy",
        choices=["loss", "accuracy"],
        help="metric to use for early stopping (default: %(default)s)",
    )
    parser.add_argument(
        "--from_scratch",
        action="store_true",
        help="whether to train a new model every co-training iteration (default: False)",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="/ourdisk/hpc/ai2es/jroth/co-training/co-training_deep_ens_plus_plus/",
        help="path for hparam search directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="seed for random number generator (default: %(default)s)",
    )
    parser.add_argument("--tag", type=str, help="tag for wandb run logging")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

    main(args, rank, world_size)
