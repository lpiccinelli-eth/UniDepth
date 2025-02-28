import argparse
import json
import os
import random
import uuid
from contextlib import nullcontext
from datetime import datetime as dt
from functools import partial
from time import time
from typing import Any, Dict

import git
import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.utils.data.distributed
import wandb
from torch import distributed as dist
from torch import optim
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

import unidepth.datasets as datasets
from unidepth.datasets import (ConcatDataset, DistributedSamplerNoDuplicate,
                               collate_fn, get_weights)
from unidepth.models import UniDepthV1, UniDepthV2
from unidepth.ops.scheduler import CosineScheduler
from unidepth.utils.distributed import (barrier, create_local_process_group,
                                        is_main_process,
                                        local_broadcast_process_authkey,
                                        setup_multi_processes, setup_slurm,
                                        sync_string_across_gpus,
                                        sync_tensor_across_gpus)
from unidepth.utils.ema_torch import (DummyExponentialMovingAverage,
                                      ExponentialMovingAverage)
from unidepth.utils.misc import calculate_mean_values, format_seconds
from unidepth.utils.validation import validate

EMA_INTERVAL = 10


def aggregate_sync_losses(dict_: dict[str, torch.Tensor], device):
    keys = list(dict_.keys())
    values = torch.tensor(list(dict_.values()), device=device)
    keys = sync_string_across_gpus(keys, device)
    values = sync_tensor_across_gpus(values, dim=0).cpu().tolist()
    dict_ = calculate_mean_values(keys, values)
    return dict_


def main_worker(config: Dict[str, Any], args: argparse.Namespace):

    current_process = psutil.Process(os.getpid())
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    seed = config["generic"]["seed"]

    if not args.distributed:
        args.rank = 0
        args.local_rank = 0
        args.world_size = 1
    else:
        # initializes the distributed backend which will take care of synchronizing nodes/GPUs
        setup_multi_processes(config)
        is_slurm = "SLURM_PROCID" in os.environ
        if is_slurm:
            setup_slurm("nccl", port=args.master_port)
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = device = int(os.environ["LOCAL_RANK"])
        if not is_slurm:
            dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)
            torch.cuda.set_device(device)
        create_local_process_group()
        local_broadcast_process_authkey()
        print(
            f"Start running DDP on: {args.rank} (local: {args.local_rank}) with seed {seed + args.rank}."
        )
        config["training"]["batch_size"] = int(
            config["training"]["batch_size"] / args.world_size
        )
        dist.barrier()

    # Fix seed
    # Different for every machine to avoid sampling
    # the same element across machines
    seed = seed + args.rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    batch_size = config["training"]["batch_size"]
    if is_main_process():
        print("Config: ", args.config_file)
        print(
            f"Torch version:{torch.__version__}, cuda:{torch.version.cuda}, cudnn:{torch.backends.cudnn.version()}, threads:{torch.get_num_threads()}"
        )
        print("BatchSize per GPU: ", batch_size)
        print(
            f"Divided into {config['training']['nsteps_accumulation_gradient']} accumulation step"
        )

    ##############################
    ########### MODEL ############
    ##############################
    # Build model
    model = eval(config["model"]["name"])(config).to(device)
    model.eval()
    print(f"MODEL: {model.__class__.__name__} at {model.device}")
    torch.cuda.empty_cache()
    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # Before setting find_unused_parameters=True, be sure no gradient checkpointing is on
        # => DDP with double pass (unused params) is incompatible with gradient checkpointing
        # My guess: CheckpointFunction does not have a pointer pointing to the original model
        # params (?), so DDP Reducer will treat those model parameters as unused, and mark them
        # as ready. Later, when the forward-backward is run within CheckpointFunction, those
        # parameters will be marked as ready again, which can also trigger this error.

        # Possible workaround (not sure at all):
        # with model.no_sync():
        #     model(inputs).backward()
        # works = [dist.all_reduce(p.grad, async_op=True) for p in model.parameters()] # Manual sync
        # ideally one can organize grads to larger buckets to make allreduce more efficient
        # for work in works:
        #     work.wait()
        model = DistributedDataParallel(
            model,
            find_unused_parameters=False,
            device_ids=[device],
            output_device=device,
        )

    ##############################
    ######### OPTIMIZER ##########
    ##############################
    f16 = config["training"].get("f16", False)
    clipping = config["training"].get("clipping", None)

    # Optimize
    ddp_model = model.module if args.distributed else model
    params = ddp_model.get_params(config)
    optimizer = optim.AdamW(params)

    # Load Model:
    step = 0
    if config["training"].get("pretrained", None) is not None:
        ddp_model.load_pretrained(config["training"]["pretrained"])
        pretrained = torch.load(
            config["training"]["pretrained"], map_location="cpu", weights_only=False
        )
        try:
            optimizer.load_state_dict(pretrained["optimizer"])
        except Exception as e:
            if is_main_process():
                print("Could not load optimizer state dict:", e)
        step = pretrained.get("step", step)

    # EMA
    ema_class = (
        ExponentialMovingAverage
        if config["training"]["ema"]
        else DummyExponentialMovingAverage
    )
    ema_handle = ema_class(
        model.parameters(),
        1 - (1 - 0.9995) * EMA_INTERVAL,
        update_after_step=75000 // EMA_INTERVAL,
        tau=20000 // EMA_INTERVAL,
    )
    setattr(ema_handle, "num_updates", step // EMA_INTERVAL)

    ##############################
    ######### GENERICS ###########
    ##############################
    resize_method = config["data"].get("resize_method", "hard")
    crop = config["data"].get("crop", "garg")
    augmentations_db = config["data"].get("augmentations", {})
    image_shape = config["data"]["image_shape"]
    nsteps_accumulation_gradient = config["training"]["nsteps_accumulation_gradient"]
    batch_size = config["training"]["batch_size"] // config["data"]["num_copies"]

    is_shell = int(os.environ.get("SHELL_JOB", 0))
    run_id = sync_string_across_gpus(
        [f"{dt.now().strftime('%d-%h_%H-%M')}-{uuid.uuid4()}"], device
    )[0]

    if not is_shell and is_main_process():
        repo_folder = os.path.dirname(os.path.realpath(__file__))
        try:
            repo = git.Repo(repo_folder)
            current_head = repo.head if repo.head.is_detached else repo.active_branch
            notes = f"MESSAGE: {current_head.commit.message} HASH:{current_head.commit.hexsha} BRANCH:{current_head.name}"
        except Exception as e:
            print(
                f"problem with {repo_folder}, does it exist? (original excpetion: {e})"
            )
            notes = ""

        # restore the original batchsize, not acquired by other calls from now on
        if args.distributed:
            config["training"]["batch_size"] = (
                config["training"]["batch_size"]
                * args.world_size
                * config["data"]["num_copies"]
            )
        wandb.init(
            project="UniDepth",
            name=run_id,
            config=config,
            tags=None,
            notes=notes,
            dir=os.environ.get("WANDB_HOME", os.environ.get("TMPDIR", "/tmp")),
        )
        wandb.watch(model)

    ##############################
    ########## DATASET ###########
    ##############################
    # Datasets loading
    train_datasets, val_datasets, dims = {}, {}, 0
    if is_main_process():
        print("Loading training datasets...")
    for dataset in config["data"]["train_datasets"]:
        assert hasattr(datasets, dataset), f"{dataset} not a custom dataset"
        train_dataset: datasets.BaseDataset = getattr(datasets, dataset)
        train_datasets[dataset] = train_dataset(
            image_shape=image_shape,
            split_file=train_dataset.train_split,
            test_mode=False,
            crop=crop,
            augmentations_db=augmentations_db,
            normalize=config["data"].get("normalization", "imagenet"),
            resize_method=resize_method,
            mini=1.0,
            num_copies=config["data"]["num_copies"],
            num_frames=1,
            fps_range=[1, 30],
        )
        dim = (
            train_datasets[dataset].dataset._addr.numel() * 8
            + train_datasets[dataset].dataset._lst.numel()
        ) / (2**20)
        if hasattr(train_datasets[dataset], "sequences"):
            dim += (
                train_datasets[dataset].sequences._addr.numel() * 8
                + train_datasets[dataset].sequences._lst.numel()
            ) / (2**20)
        if is_main_process():
            print(f"{dataset}: {dim:.1f}MB")
        dims += dim

    if is_main_process():
        print(f"All training datasets loaded, with total size: {dims:.1f}MB")

    barrier()

    assert batch_size % nsteps_accumulation_gradient == 0
    batch_chunk = batch_size // nsteps_accumulation_gradient

    train_dataset = ConcatDataset(
        [t for t in train_datasets.values()],
        shape_constraints=config["data"]["augmentations"]["shape_constraints"],
    )

    if is_main_process():
        print("Loading validation datasets...")
    for dataset in config["data"]["val_datasets"]:
        val_dataset: datasets.BaseDataset = getattr(datasets, dataset)
        val_datasets[dataset] = val_dataset(
            image_shape=image_shape,
            split_file=val_dataset.test_split,
            test_mode=True,
            crop=crop,
            augmentations_db=augmentations_db,
            normalize=config["data"].get("normalization", "imagenet"),
            resize_method=resize_method,
            num_frames=1,
            mini=1.0,
        )

    # Dataset samplers, create distributed sampler pinned to rank
    if args.distributed:
        weights, num_samples = get_weights(train_datasets, config["data"]["sampling"])
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples, replacement=True
        )
        valid_samplers = {
            k: DistributedSamplerNoDuplicate(
                v,
                num_replicas=args.world_size,
                rank=args.rank,
                shuffle=False,
                drop_last=False,
            )
            for k, v in val_datasets.items()
        }
    else:
        train_sampler = RandomSampler(train_dataset)
        valid_samplers = {k: SequentialSampler(v) for k, v in val_datasets.items()}

    train_sampler = torch.utils.data.BatchSampler(
        train_sampler, batch_size=batch_size, drop_last=True
    )

    # DATASET LOADERS
    val_batch_size = 1
    num_workers = 0  # int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    train_loader = DataLoader(
        train_dataset,
        num_workers=num_workers,
        sampler=train_sampler,
        pin_memory=True,
        collate_fn=partial(collate_fn, is_batched=True),
        persistent_workers=num_workers > 0,
    )
    val_loaders = {
        name_dataset: DataLoader(
            dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=valid_samplers[name_dataset],
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn, is_batched=False),
        )
        for name_dataset, dataset in val_datasets.items()
    }

    # SCHEDULERS
    scheduler_wd = CosineScheduler(
        optimizer,
        key="weight_decay",
        init_value=config["training"]["wd"],
        base_value=config["training"]["wd"],
        final_value=config["training"]["wd_final"],
        warmup_iters=0,
        total_iters=config["training"]["n_iters"],
        step_init=step - 1,
    )
    scheduler_lr = CosineScheduler(
        optimizer,
        key="lr",
        init_value=config["training"]["lr"] * config["training"].get("lr_warmup", 1.0),
        final_value=config["training"]["lr_final"],
        warmup_iters=config["training"]["warmup_iters"],
        total_iters=config["training"]["n_iters"],
        step_init=step - 1,
    )
    scheduler_betas = CosineScheduler(
        optimizer,
        key="betas",
        init_value=0.95 if config["training"].get("cycle_betas", True) else 0.9,
        base_value=0.85 if config["training"].get("cycle_betas", True) else 0.9,
        final_value=0.95 if config["training"].get("cycle_betas", True) else 0.9,
        warmup_iters=config["training"]["warmup_iters"],
        total_iters=config["training"]["n_iters"],
        step_init=step - 1,
    )

    # Set loss scaler for half precision training + sanity zeroing grads
    # dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    dtype = torch.float16  # bfloat has some issue in terms of resolution...
    context = torch.autocast(device_type="cuda", dtype=dtype, enabled=f16)
    optimizer.zero_grad(set_to_none=True)

    ##############################
    ########## TRAINING ##########
    ##############################
    # Remember that if i-th layer is frozen, this will break gradient checkpointing
    # in layer i+1-th. This is because CheckpointFunction treats the i+1-th input as
    # without gradient, thus the i+1-th layer does not have grads (?). To solve it,
    # just add requires_grad_() to the inputs coming from the frozen layer
    ddp_model.train()
    start = time()
    n_steps = config["training"]["n_iters"]
    init_steps = int(step)
    track_pbar = is_shell

    if is_main_process():
        print("Is a shell job?", is_shell)
        print("Use dtype:", dtype if f16 else torch.float32)
        print(
            f'Train for {config["training"]["n_iters"]} steps, validate every {config["training"]["validation_interval"]} steps'
        )
        print(f"START with {num_workers} workers")
        if track_pbar:
            pbar = tqdm(total=n_steps - init_steps)

    scaler = torch.amp.GradScaler("cuda", enabled=f16)
    track_losses = {}
    system_memory = dict(psutil.virtual_memory()._asdict())["available"] / 2**30
    cpid_memory = current_process.memory_info()[0] / 2.0**30
    gpu_mem = (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 2**30

    while True:
        for j, batches in enumerate(train_loader):

            system_memory = (
                0.99 * system_memory
                + 0.01 * dict(psutil.virtual_memory()._asdict())["available"] / 2**30
            )
            cpid_memory = (
                0.99 * cpid_memory + 0.01 * current_process.memory_info()[0] / 2.0**30
            )
            gpu_mem = (
                0.99 * gpu_mem
                + 0.01
                * (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0])
                / 2**30
            )
            if j % 1000 == 0 and is_main_process():
                print(f"System information at step {j}")
                print(f"System-wide RAM available: {system_memory:.2f}GB")
                print(f"CPU utilization: {psutil.cpu_percent(interval=None)}%")
                print(f"GPU memory utilized: {gpu_mem:.2f}GB")

            batches["data"] = {
                k: v.to(model.device, non_blocking=True)
                for k, v in batches["data"].items()
            }
            for idx in range(nsteps_accumulation_gradient):
                batch = {}
                batch_slice = slice(idx * batch_chunk, (idx + 1) * batch_chunk)
                batch["data"] = {k: v[batch_slice] for k, v in batches["data"].items()}
                batch["img_metas"] = batches["img_metas"][batch_slice]

                # remove temporal dimension of the dataloder, here is always 1!!
                batch["data"] = {
                    k: torch.flatten(v, 0, 1) for k, v in batch["data"].items()
                }
                batch["img_metas"] = [
                    {k: v for k, v in meta.items() if isinstance(v, list)}
                    for meta in batch["img_metas"]
                ]

                with (
                    model.no_sync()
                    if idx < nsteps_accumulation_gradient - 1
                    else nullcontext()
                ):
                    with context:
                        preds, losses = model(batch["data"], batch["img_metas"])
                    loss = sum(losses["opt"].values()) / nsteps_accumulation_gradient
                    scaler.scale(loss).backward()

                losses_dict = {
                    k: v.detach() for loss in losses.values() for k, v in loss.items()
                }
                track_losses.update(
                    {
                        k: 0.99 * track_losses.get(k, v)
                        + 0.01 * torch.nan_to_num(v, nan=1e5, posinf=1e5, neginf=1e5)
                        for k, v in losses_dict.items()
                    }
                )

            if clipping is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)

            scaler.step(optimizer)
            scaler.update()

            scheduler_wd.step()
            scheduler_lr.step()
            scheduler_betas.step()
            optimizer.zero_grad(set_to_none=True)
            if step % EMA_INTERVAL == 0:
                ema_handle.update()

            if is_main_process() and track_pbar:
                pbar.update(1)

            step += 1

            # LOGGING
            if step % 50 == 0:
                track_losses = aggregate_sync_losses(track_losses, device=model.device)
                if is_main_process():
                    try:
                        wandb.log(
                            {
                                **{f"Train/{k}": v for k, v in track_losses.items()},
                                **{f"Train/lr": scheduler_lr.get()[-1]},
                                **{f"Train/wd": scheduler_wd.get()[-2]},
                                **{f"Train/scale_f16": scaler.get_scale()},
                            },
                            step=step,
                        )
                    except Exception as e:
                        print("Not logging loss because of:", e)
                        pass

            if step % 100 == 0 and is_main_process():
                log_loss_dict = {f"Train/{k}": v for k, v in track_losses.items()}
                elapsed = int(time() - start)
                eta = int(elapsed * (n_steps - step) / max(1, step - init_steps))
                print(
                    f"Loss at {step}/{n_steps} [{format_seconds(elapsed)}<{format_seconds(eta)}]:"
                )
                print(", ".join([f"{k}: {v:.5f}" for k, v in log_loss_dict.items()]))

            # Validation
            is_last_step = step >= config["training"]["n_iters"]
            is_validation = step % config["training"]["validation_interval"] == 0
            if is_last_step or is_validation:
                torch.cuda.empty_cache()
                barrier()
                if is_main_process():
                    print(f"Validation at {step}th step...")

                ddp_model.eval()
                start_validation = time()
                with torch.no_grad(), ema_handle.average_parameters():
                    validate(
                        model, test_loaders=val_loaders, step=step, context=context
                    )

                if is_main_process():
                    print(f"Elapsed: {format_seconds(int(time() - start_validation))}")
                ddp_model.train()
                torch.cuda.empty_cache()

            if step >= config["training"]["n_iters"]:
                if is_main_process() and track_pbar:
                    pbar.close()
                wandb.finish(0)
                dist.destroy_process_group()
                return 0


if __name__ == "__main__":
    # dummy folder to avoid Triton conflicts on multi-node training with slurm
    if "SLURM_PROCID" in os.environ:
        os.environ["TRITON_CACHE_DIR"] = "/tmp"

    parser = argparse.ArgumentParser(
        description="Training script", conflict_handler="resolve"
    )
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--master-port", type=str)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        config = json.load(f)

    deterministic = config["generic"].get("deterministic", True)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    # set to false otw f16 + efficient memory is not really efficient... i.e. mem is greater than with
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.set_num_threads(1)
    main_worker(config, args)
