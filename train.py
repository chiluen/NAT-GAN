import logging
import math
import os
import random
import sys
from argparse import ArgumentParser
import fairseq
import numpy as np
import torch

from fairseq import (
    checkpoint_utils, distributed_utils, metrics, options, progress_bar, tasks, utils
)
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import StopwatchMeter

##從其他地方去import
#from NAT_GAN.discriminator import RNN_Discriminator, LSTM_Discriminator
#from NAT_GAN.train_process import train_step_GAN
from discriminator import RNN_Discriminator, LSTM_Discriminator
from train_process import train_step_GAN


#for validation loss
def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""

    if args.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(args.fixed_validation_seed)

    valid_losses = []
    for subset in subsets:
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens_valid,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.progress_bar(
            itr,
            log_format=args.log_format,
            log_interval=args.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                args.tensorboard_logdir if distributed_utils.is_master(args) else None
            ),
            default_log_format=('tqdm' if not args.no_progress_bar else 'simple'),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for sample in progress:
                trainer.valid_step(sample)

        # log validation stats
        stats = get_valid_stats(args, trainer)
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[args.best_checkpoint_metric])
    return valid_losses
  
def get_valid_stats(args, trainer):
    stats = metrics.get_smoothed_values('valid')
    if 'valid_nll_loss' in stats and 'ppl' not in stats:
        stats['valid_ppl'] = utils.get_perplexity(stats['nll_loss'])
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, 'best'):
        key = 'best_{0}'.format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[args.best_checkpoint_metric],
        )
    return stats

#for stop_early

def should_stop_early(args, valid_loss):
    if args.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if args.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, 'best', None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        return should_stop_early.num_runs > args.patience



####################
###     main     ###
####################
def main(args, init_distributed = False):
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        stream=sys.stdout,
    )
    logger = logging.getLogger('fairseq_cli.train')

    #utils.import_user_module(args)  這一行不需要
    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'


    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if init_distributed:
        args.distributed_rank = distributed_utils.distributed_init(args)

    #這不需要, 多GPU的
    #if distributed_utils.is_master(args):
    #    checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    logger.info(args)
    task = tasks.setup_task(args) #要看translation_lev那邊的  #這就只是在把data處理好

    for valid_sub_split in args.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=0)

    model = task.build_model(args)  #先傳到tasks/translation
    criterion = task.build_criterion(args) #這邊的build_criterion的位置在fairseq/fairseq/tasks/fairseq_task.py/
    
    logger.info('model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    logger.info('num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    trainer = Trainer(args, task, model, criterion) #用來把model, task, criterion綁起來, 準備一起Train
    logger.info('training on {} GPUs'.format(args.distributed_world_size))
    logger.info('max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)  

    max_epoch = args.max_epoch or math.inf  #如果沒有特別設定max_epoch（dafault = 0）, 那就會變成inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()  #要把lr_scheduler改一改   #為何這邊初始的lr會變成 0  ->  要用warmup_init_lr去調整
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_subsets = args.valid_subset.split(',')

    # construct LSTM discriminator
    input_size = 1 #len(task.target_dictionary)
    output_size = 1
    hidden_size = 50
    input_vocab = task.target_dictionary
    Discriminator = LSTM_Discriminator(input_size, output_size, hidden_size, input_vocab)
    if args.discriminator_path != '':
        Discriminator.model.load_state_dict(torch.load(args.discriminator_path))

    #紀錄每一個epoch的loss
    loss_per_epoch = []
    valid_loss_per_epoch = []


    #進入到Training地方
    while (
        lr > args.min_lr
        and (
            #這個epoch_itr是從checkpoint來的
            epoch_itr.epoch < max_epoch
            # allow resuming training from the final checkpoint
            or epoch_itr._next_epoch_itr is not None
        )
        and trainer.get_num_updates() < max_update
    ):
        ###Train for one epoch###

        # Initialize data iterator
        # Initialize for discriminator
        itr = epoch_itr.next_epoch_itr(
            fix_batches_to_gpus=args.fix_batches_to_gpus,
            shuffle=(epoch_itr.epoch >= args.curriculum),
        )
        update_freq = (
            args.update_freq[epoch_itr.epoch - 1]
            if epoch_itr.epoch <= len(args.update_freq)
            else args.update_freq[-1]
        )
        itr = iterators.GroupedIterator(itr, update_freq)
        progress = progress_bar.build_progress_bar( 
            args, itr, epoch_itr.epoch, no_progress_bar='simple',
        )

        args.current_epoch = epoch_itr.epoch 
        torch.multiprocessing.set_sharing_strategy('file_system') ##fix "received 0 items of ancdata" bug

        #Progress只要使用就會消失
        #新增一個list讓progress在每個epoch都保存
        progress_list = [] #新建與清空內存
        progress_counter = 0 #只用前一萬個train discriminator, 以防overfitting
        for samples in progress:
            progress_list.append(samples)
            progress_counter += 1
            if progress_counter == 10:
                break
        logger.info("Complete the progress_list")

        ##Construct progress for generator
        itr = epoch_itr.next_epoch_itr(
            fix_batches_to_gpus=args.fix_batches_to_gpus,
            shuffle=(epoch_itr.epoch >= args.curriculum),
        )
        update_freq = (
            args.update_freq[epoch_itr.epoch - 1]
            if epoch_itr.epoch <= len(args.update_freq)
            else args.update_freq[-1]
        )
        itr = iterators.GroupedIterator(itr, update_freq)
        progress = progress_bar.build_progress_bar( 
            args, itr, epoch_itr.epoch, no_progress_bar='simple',
        )

        #pretrain for discriminator
        if args.pretrain_D_times > 0:
            logger.info('Pretrain ' + str(args.pretrain_D_times) + ' for discriminator')
            for i in range(args.pretrain_D_times):
                train_D(args, trainer, task, epoch_itr, Discriminator, progress_list)
                logger.info('Pretrain on Discriminator for ' + str(i) + " times")

            args.pretrain_D_times = 0 #Ending pretrain


        #Training generator for one epoch
        stats = train_G(args, trainer, task, epoch_itr, Discriminator, progress)  #train for generator
        
        #train discriminator n_times for one epoch
        if args.only_G == False:
            logger.info('Training on Discriminator for ' + str(args.train_D_times_per_epoch) + " times")
            for i in range(args.train_D_times_per_epoch):
                train_D(args, trainer, task, epoch_itr, Discriminator, progress_list)  #train for discriminator
            torch.save(Discriminator.model.state_dict(), os.path.join('./NAT-GAN/discriminator_model',"param_"+str(epoch_itr.epoch)+".pkl"))

        loss_per_epoch.append(stats['loss']) # record loss

        #用validate_interval去控制多少epoch後要去算validate
        if not args.disable_validation and epoch_itr.epoch % args.validate_interval == 0:
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
        else:
            valid_losses = [None]
        valid_loss_per_epoch.append(valid_losses[0])

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # save checkpoint, which is average loss and best model
        if epoch_itr.epoch % args.save_interval == 0:
            logger.info('Enter check point')
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        # early stop
        if should_stop_early(args, valid_losses[0]):
            logger.info('early stop since valid performance hasn\'t improved for last {} runs'.format(args.patience))
            break

        # for next epoch
        epoch_itr = trainer.get_train_iterator(
            epoch_itr.epoch,
            # sharded data: get train iterator for next epoch
            load_dataset=(os.pathsep in getattr(args, 'data', '')),
        )



    #把loss與valid loss save到local端
    import csv
    with open(args.save_loss_dir + "/loss.txt","w") as f:
        for i in loss_per_epoch:
            f.write(str(i))
            f.write(',')
    with open(args.save_loss_dir + "/valid_loss.txt", "w") as f:
        for i in valid_loss_per_epoch:
            f.write(str(i))
            f.write(',')

    train_meter.stop()
    logger.info('done training in {:.1f} seconds'.format(train_meter.sum))


#train.train_D
def train_D(args, trainer, task, epoch_itr, discriminator, progress):
    """Train the discriminator for one epoch"""

    def change_output2target(outputs, batch_size): 
        outputs = outputs['word_ins']['out']
        outputs_argmax = torch.max(outputs[0], dim = 1)[1].reshape(1,-1)
        for i in range(1, batch_size):
            indice = torch.max(outputs[i], dim = 1)[1].reshape(1,-1)
            outputs_argmax = torch.cat((outputs_argmax, indice))
        return outputs_argmax

    generator = trainer.model
    
    for samples in progress:
        for i, sample in enumerate(samples):
            sample = trainer._prepare_sample(sample) 
            sample['prev_target'] = task.inject_noise(sample['target']) 

            #把sample處理成可以被Generator吃的
            src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
            )
            tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]
            outputs = generator(src_tokens, src_lengths, prev_output_tokens, tgt_tokens) 

            #把outputs換成可以被discriminator吃的東西
            outputs_argmax = change_output2target(outputs, src_tokens.shape[0])

            #把tgt與outputs吃進去, 做training
            discriminator.train(tgt_tokens, outputs_argmax)
            torch.cuda.empty_cache()


#train.train_G
def get_training_stats(stats_key):
    stats = metrics.get_smoothed_values(stats_key)
    if 'nll_loss' in stats and 'ppl' not in stats:
        stats['ppl'] = utils.get_perplexity(stats['nll_loss'])
    stats['wall'] = round(metrics.get_meter('default', 'wall').elapsed_time, 0)
    return stats

@metrics.aggregate('train')
def train_G(args, trainer, task, epoch_itr, discriminator, progress):
    """Train the model for one epoch."""

    task.begin_epoch(epoch_itr.epoch, trainer.get_model())

    valid_subsets = args.valid_subset.split(',')
    max_update = args.max_update or math.inf
    for samples in progress:
        with metrics.aggregate('train_inner'):
            log_output = train_step_GAN(args, trainer, samples, discriminator)    #已改完, 並且test可行
            num_updates = trainer.get_num_updates()
            if log_output is None:
                continue

            # log mid-epoch stats
            stats = get_training_stats('train_inner')
            progress.log(stats, tag='train', step=num_updates)

            if (
                not args.disable_validation
                and args.save_interval_updates > 0
                and num_updates % args.save_interval_updates == 0
                and num_updates > 0
            ):
                valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
                checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

            if num_updates >= max_update:
                break
    
    # log end-of-epoch stats
    stats = get_training_stats('train')
    progress.print(stats, tag='train', step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters('train')

    return stats


if __name__ == '__main__':
    #dealing with argparse
    
    parser = ArgumentParser()
    parser.add_argument('--max-sentences', type = int)
    parser.add_argument('--device-id', type=int, default = 0)
    parser.add_argument('--seed', type=int, default = 1)
    parser.add_argument('--cpu', action='store_true')

    parser.add_argument('--task', type = str, default = 'translation_lev')
    parser.add_argument('--left-pad-source', action='store_false')
    parser.add_argument('--left-pad-target', action='store_true')
    parser.add_argument('--source-lang', type=str, default = 'de')
    parser.add_argument('--target-lang', type=str, default = 'en')
    parser.add_argument('--valid-subset', type = str, default = 'valid')
    parser.add_argument('--dataset-impl', type=str)
    parser.add_argument('--upsample-primary', type = int, default = 1)
    parser.add_argument('--max-source-positions', type = int, default = 1024)
    parser.add_argument('--max-target-positions', type = int, default = 1024)

    #build model
    parser.add_argument('--arch', type = str, default = 'nonautoregressive_transformer')
    parser.add_argument('--encoder-layers-to-keep', action = 'store_true')
    parser.add_argument('--decoder-layers-to-keep', action = 'store_true')
    parser.add_argument('--share-all-embeddings', action = 'store_false')
    parser.add_argument('--apply-bert-init', action = 'store_false')
    parser.add_argument('--encoder-layerdrop', type=int, default = 0)
    parser.add_argument('--decoder_layerdrop', type=int, default = 0)
    parser.add_argument('--pred-length-offset', action = 'store_false')
    parser.add_argument('--length-loss-factor', type=float, default = 0.1)    

    #build Trainer
    parser.add_argument('--criterion', type = str, default = 'nat_loss')
    parser.add_argument('--fp16', action = 'store_true')
    parser.add_argument('--distributed-world-size', type = int, default = 1)
    parser.add_argument('--use-bmuf', action = 'store_true')
    parser.add_argument('--optimizer', type = str, default = 'adam')
    parser.add_argument('--lr', action = 'append', default = [0.0005])
    parser.add_argument('--adam-betas', type = str, default = '(0.9,0.98)')
    parser.add_argument('--weight-decay', type = float, default = 0.01)
    parser.add_argument('--lr-scheduler', type = str, default = 'inverse_sqrt')
    parser.add_argument('--min-lr', type = float, default = 1e-09)
    parser.add_argument('--warmup-init-lr', type = float, default = 1e-07)

    #builde checkpoint arguments
    parser.add_argument('--distributed-rank', type = int, default = 0)
    parser.add_argument('--save-dir', type = str, default = 'checkpoints')
    parser.add_argument('--restore-file', type = str, default = 'checkpoint_last.pt')
    parser.add_argument('--reset-optimizer', action = 'store_true')
    parser.add_argument('--reset-lr-scheduler', action = 'store_true')
    parser.add_argument('--optimizer-overrides', type = str, default = '{}')
    parser.add_argument('--reset-meters', action = 'store_true')
    parser.add_argument('--train-subset', type = str, default = 'train')
    parser.add_argument('--num-workers', type = int, default = 1)
    parser.add_argument('--reset-dataloader', action = 'store_true')

    parser.add_argument('--max-epoch', type = int, default = 0)
    parser.add_argument('--max-update', type = int, default = 0)
    parser.add_argument('--no-epoch-checkpoints', action = 'store_true')

    #train function所用到的args
    parser.add_argument('--fix-batches-to-gpus', action = 'store_true')
    parser.add_argument('--curriculum', type = int, default = 0)
    parser.add_argument('--update-freq', action = 'append', default = [1])
    parser.add_argument('--log-format', type = str, default = 'simple')
    parser.add_argument('--log-interval', type = int, default = 100)
    parser.add_argument('--noise', type = str, default = 'full_mask')
    parser.add_argument('--tensorboard-logdir', type = str, default = '')
    parser.add_argument('--clip_norm', type = float, default = 0.0)
    parser.add_argument('--eval-bleu', action = 'store_true')
    parser.add_argument('--empty-cache-freq', type = int, default = 0)
    parser.add_argument('--disable-validation', action = 'store_true')
    parser.add_argument('--save-interval-updates', type = int, default = 0)

    #用在結束one epoch之後的
    parser.add_argument('--validate-interval', type = int, default = 1)
    parser.add_argument('--fixed-validation-seed', type = int, default = 7)
    parser.add_argument('--max-tokens-valid', type = int, default = 4096)
    parser.add_argument('--max-sentences-valid', type = int)
    parser.add_argument('--skip-invalid-size-inputs-valid-test', action = 'store_true')
    parser.add_argument('--required-batch-size-multiple', type = int, default = 8)
    parser.add_argument('--no-progress-bar', action = 'store_true')
    parser.add_argument('--best-checkpoint-metric', type = str, default = 'loss')
    parser.add_argument('--save-interval', type = int, default = 2)
    parser.add_argument('--maximize-best-checkpoint-metric', action = 'store_true')
    parser.add_argument('--no-save', action = 'store_true')
    parser.add_argument('--keep-best-checkpoints', type = int, default = 0)
    parser.add_argument('--no-last-checkpoints', action = 'store_true')
    parser.add_argument('--no-save-optimizer-state', action = 'store_true')
    parser.add_argument('--keep-interval-updates', type = int, default = -1)
    parser.add_argument('--keep-last-epochs', type = int, default = -1)
    parser.add_argument('--patience', type = int, default = 2)

    #save for loss and valid loss
    # & other options
    parser.add_argument('--data', type = str) 
    parser.add_argument('--max-tokens', type=int, default = 4000) 
    parser.add_argument('--save-loss-dir', type = str)  #要放絕對路徑
    parser.add_argument('--only-G', action = 'store_true') #用來只train Generator
    parser.add_argument('--pretrain-D-times', type = int, default = 0)
    parser.add_argument('--train-D-times-per-epoch', type = int, default = 1)
    parser.add_argument('--discriminator-path', type = str, default = '') #loading trained discriminator
    parser.add_argument('--current-epoch', type = int, default=0) #for convenient
    parser.add_argument('--apply-reward-scalar', action = 'store_true')


    args = parser.parse_args()
    main(args)