# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch
import torch.amp

import util.misc as misc
import util.lr_sched as lr_sched
from util.masking_scheduler import MaskingScheduler

import json

def _parse_masking_args(arg_str: str):
    # strip single quotes if present
    arg_str = arg_str.strip("'")
    try:
        masking_args = json.loads(arg_str)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid masking_args: {arg_str}")
    return masking_args

def parse_masking_sched_params(param_string):
    if param_string is None:
        return None
    
    params = {}
    for pair in param_string.split(','):
        key, value = pair.split('=')
        key = key.strip()
        value = value.strip()
        
        # Convert to appropriate type
        if key in ['initial_ratio', 'final_ratio']:
            params[key] = float(value)
        elif key in ['warmup_epochs']:
            params[key] = int(value)
        else:
            raise ValueError(f"Unknown parameter: {key}")
    
    return params
    


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = len(data_loader) // 5 #if not epoch == 0 else 20
    masking_args = _parse_masking_args(args.masking_args)
    nan_count = 0

    accum_iter = args.accum_iter

    optimizer.zero_grad()
    
    #masking ratio scheduler
    masking_sched_params = parse_masking_sched_params(args.use_masking_sched)

    if masking_sched_params is not None:
        masking_scheduler = MaskingScheduler(**masking_sched_params)
        masking_scheduler.steps_per_epoch = len(data_loader)
        print(f"Starting epoch {epoch} with a masking ratio of {masking_scheduler((epoch)*len(data_loader))} ending with {masking_scheduler((epoch+1)*len(data_loader))}")

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        
        if masking_sched_params is not None:
            # TODO check whether key "masking_ratio" exists
            masking_args["masking_ratio"] = masking_scheduler((data_iter_step)+(epoch)*len(data_loader))
            
        

        with torch.amp.autocast('cuda'):
            loss, _, _ = model(samples, masking_type=args.masking_type, entropy_weighting=args.entropy_weighting, **masking_args)

        loss_value = loss.item()

        # Playing NaN Limbo
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, skipping this iteration")
            nan_count += 1
            continue

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # Print NaN count
    print(f'Epoch {epoch} NaN count: {nan_count}')

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}