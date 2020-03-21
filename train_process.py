##########train_step_GAN
#train.train_G -> train_step_GAN
import contextlib
from itertools import chain
import logging
import math
import os
import sys
from typing import Any, Dict, List
import torch

from fairseq import checkpoint_utils, distributed_utils, metrics, models, optim, utils
from fairseq.file_io import PathManager
from fairseq.meters import AverageMeter, StopwatchMeter, TimeMeter
from fairseq.optim import lr_scheduler
from fairseq.utils import new_arange
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, load_langpair_dataset

from NAT_GAN.criterion_GAN import criterion_GAN


def train_step_GAN(args, trainer, samples, discriminator,raise_oom=False):
    """Do forward, backward and parameter update."""
    logger = logging.getLogger(__name__)



    if trainer._dummy_batch is None:
        trainer._dummy_batch = samples[0]

    trainer._set_seed()
    trainer.model.train()
    trainer.criterion.train()
    trainer.zero_grad()
    metrics.log_start_time("train_wall", priority=800, round=0)
    logging_outputs, sample_size, ooms = [], 0, 0
    for i, sample in enumerate(samples):
        sample = trainer._prepare_sample(sample)
        if sample is None:
          # when sample is None, run forward/backward on a dummy batch
          # and ignore the resulting gradients
            sample = trainer._prepare_sample(trainer._dummy_batch)
            is_dummy_batch = True
        else:
            is_dummy_batch = False

        def maybe_no_sync():
            """
            Whenever *samples* contains more than one mini-batch, we
            want to accumulate gradients locally and only call
            all-reduce in the last backwards pass.
            """
            if (
                args.distributed_world_size > 1
                and hasattr(trainer.model, "no_sync")
                and i < len(samples) - 1
            ):
                return trainer.model.no_sync()
            else:
                return contextlib.ExitStack()  # dummy contextmanager

        try:
            with maybe_no_sync():
            
                # forward and backward
                loss, sample_size_i, logging_output = train_step_task(args, 
                                                                      trainer.task, 
                                                                      sample, 
                                                                      trainer.model, 
                                                                      trainer.criterion, 
                                                                      trainer.optimizer, 
                                                                      discriminator,
                                                                      ignore_grad = is_dummy_batch)
            
                del loss

            logging_outputs.append(logging_output)
            if not is_dummy_batch:
                sample_size += sample_size_i

            # emptying the CUDA cache after the first step can
            # reduce the chance of OOM
            if trainer.cuda and trainer.get_num_updates() == 0:
                torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                trainer._log_oom(e)
                if raise_oom:
                    raise e
                logger.warning(
                    "attempting to recover from OOM in forward/backward pass"
                )
                ooms += 1
                trainer.zero_grad()
            else:
                raise e
    # gather logging outputs from all replicas
    if trainer._sync_stats():
        logging_outputs, (sample_size, ooms) = trainer._aggregate_logging_outputs(
            logging_outputs, sample_size, ooms, ignore=is_dummy_batch,
        )
    metrics.log_scalar("oom", ooms, len(samples), priority=600, round=3)
    if ooms == trainer.args.distributed_world_size * len(samples):
        logger.warning("OOM in all workers, skipping update")
        trainer.zero_grad()
        return None

    try:
        # normalize grads by sample size
        if sample_size > 0:
            if trainer._sync_stats():
                # multiply gradients by (# GPUs / sample_size) since DDP
                # already normalizes by the number of GPUs. Thus we get
                # (sum_of_gradients / sample_size).
                trainer.optimizer.multiply_grads(trainer.args.distributed_world_size / sample_size)
            else:
                trainer.optimizer.multiply_grads(1 / sample_size)

        # clip grads
        grad_norm = trainer.optimizer.clip_grad_norm(trainer.args.clip_norm)

        if not trainer.args.use_bmuf:
            trainer._check_grad_norms(grad_norm)

        # take an optimization step
        trainer.optimizer.step()
        trainer.set_num_updates(trainer.get_num_updates() + 1)

        # task specific update per step
        #為何這個有bug...
        #Lev沒有這個選項, 只有semi的才有
        #trainer.task.update_step(trainer.get_num_updates())

        # log stats
        logging_output = trainer._reduce_and_log_stats(logging_outputs, sample_size)
        metrics.log_speed("ups", 1., ignore_first=10, priority=100, round=2)
        metrics.log_scalar("gnorm", utils.item(grad_norm), priority=400, round=3)
        metrics.log_scalar(
            "clip",
            100 if grad_norm > trainer.args.clip_norm > 0 else 0,
            priority=500,
            round=1,
        )

        # clear CUDA cache to reduce memory fragmentation
        if (
            trainer.args.empty_cache_freq > 0
            and (
                (trainer.get_num_updates() + trainer.args.empty_cache_freq - 1)
                % trainer.args.empty_cache_freq
            ) == 0
            and torch.cuda.is_available()
            and not trainer.args.cpu
        ):
            torch.cuda.empty_cache()
    
    except OverflowError as e:
        logger.info("NOTE: overflow detected, " + str(e))
        trainer.zero_grad()
        logging_output = None
    except RuntimeError as e:
        if "out of memory" in str(e):
            trainer._log_oom(e)
            logger.error("OOM during optimization, irrecoverable")
        raise e

    if trainer.args.fp16:
        metrics.log_scalar("loss_scale", trainer.optimizer.scaler.loss_scale, priority=700, round=0)

    metrics.log_stop_time("train_wall")

    return logging_output
    

          
##train_step_task
#translation_lev/ train_step
#train.train -> train_step_GAN -> train_step_task
def train_step_task(args,
               task,
                sample,
                model,
                criterion,
                optimizer,
                discriminator,
                ignore_grad=False):

    model.train()
    sample['prev_target'] = task.inject_noise(sample['target'])
    #loss, sample_size, logging_output = criterion(model, sample)
    loss, sample_size, logging_output = criterion_GAN(args, model, discriminator,sample, task)
    if ignore_grad:
        loss *= 0
    optimizer.backward(loss)
    return loss, sample_size, logging_output