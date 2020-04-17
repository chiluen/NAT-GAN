##criterion_GAN
#fairseq/criterions/nat_loss.py 
#train.train -> train_step_GAN -> train_step_task -> criterion_GAN
import math
import torch.nn.functional as F
import torch
from torch import Tensor

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


def _compute_loss(args, outputs, targets, reward, mode='word_ins', masks=None, label_smoothing=0.0, name="loss", factor=1.0):
    """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len
        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
    """
    def mean_ds(x: Tensor, dim=None) -> Tensor:
        return (
            x.float().mean().type_as(x)
            if dim is None
            else x.float().mean(dim).type_as(x)
        )

    def reward_losses(args, losses, reward, batch_size, length):
        """
        Policy gradient for generator and discriminator
        losses per sentence have to add reward to themselves
        4/4 adding function let reward effect from 0 to 1
        """

        head_loc = 0
        losses_list = []
        #reward_scalar = 1 - ((args.max_epoch - args.current_epoch) / args.max_epoch)

        for i in range(batch_size):
            losses_sentence = 0

            for j in range(length):
                try: #sometimes losses.shape not equal to batch*length
                    losses_sentence += losses[head_loc + j]
                except:
                    losses_sentence += 0
            head_loc += length

            if args.apply_reward_scalar == True:
                losses_sentence = -1 * reward_scalar * reward[i] * (losses_sentence / length)
            else:
                losses_sentence = -1 * reward[i] * (losses_sentence / length)
                
            losses_list.append(losses_sentence) 

        nll_loss = mean_ds(torch.tensor(losses_list, device=losses.device, requires_grad=True)) #change to mean
        return nll_loss

    batch_size = outputs.shape[0] 
    length = outputs.shape[1]

    if masks is not None:
        """
        only word_ins will enter there
        outputs.shape = (bsz, len, vocab_size)
        outputs[masks].shape = (bsz*len, vocab_size)
        targets[masks].shape = (bsz*len)
        """

        outputs, targets = outputs[masks], targets[masks] 

    if masks is not None and not masks.any():
        nll_loss = torch.tensor(0)
        loss = nll_loss

    else:
        """
        1. count losses
        2. calculate mean losses or reward losses
        3. label smoothing
        """
        logits = F.log_softmax(outputs, dim=-1)
        if targets.dim() == 1:
            losses = F.nll_loss(logits, targets.to(logits.device), reduction='none')
        else:  # soft-labels
            losses = F.kl_div(logits, targets.to(logits.device), reduction='none')
            losses = losses.sum(-1)

        if args.only_G == False:
            ## adding mode for discriminator rewards
            if mode == 'word_ins': #for word
                nll_loss = reward_losses(args, losses, reward, batch_size, length)
                #nll_loss = mean_ds(losses)
            else: # for length
                nll_loss = mean_ds(losses)
        else:
            nll_loss = mean_ds(losses)

        if label_smoothing > 0:
            loss = nll_loss * (
                1 - label_smoothing) - mean_ds(logits) * label_smoothing
        else:
            loss = nll_loss

    loss = loss * factor
    return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}

def _custom_loss(loss, name="loss", factor=1.0):
    return {"name": name, "loss": loss, "factor": factor}

#用來把output轉成跟target token一樣的function
#最後return的是選擇的index
def change_output2target(outputs, batch_size): 
    outputs = outputs['word_ins']['out']
    outputs_argmax = torch.max(outputs[0], dim = 1)[1].reshape(1,-1)
    for i in range(1, batch_size):
        indice = torch.max(outputs[i], dim = 1)[1].reshape(1,-1)
        outputs_argmax = torch.cat((outputs_argmax, indice))
    return outputs_argmax

def criterion_GAN(args, model, discriminator, sample, task,reduce = True):

    nsentences, ntokens = sample["nsentences"], sample["ntokens"]
    src_tokens, src_lengths = (
        sample["net_input"]["src_tokens"],
        sample["net_input"]["src_lengths"],
    )
    tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]
    outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens) #Generator
    losses, nll_loss = [], []

    batch_size, length = src_tokens.size()
    outputs_argmax = change_output2target(outputs, batch_size) #把outputs轉成跟target token一樣的形式
    reward = discriminator.model(outputs_argmax) #len of reward = batch_size
    reward = 2*(reward - 0.5) #change to (-1,1)

    for obj in outputs: #word_ins/ length
        if outputs[obj].get("loss", None) is None:  #其實兩個都是None
            _losses = _compute_loss(
                args,
                outputs[obj].get("out"),
                outputs[obj].get("tgt"),
                reward,
                obj,
                outputs[obj].get("mask", None),
                outputs[obj].get("ls", 0.0),
                name=obj + '-loss',
                factor=outputs[obj].get("factor", 1.0)
            )
        else:
            _losses = _custom_loss(
                outputs[obj].get("loss"),
                name=obj + '-loss',
                factor=outputs[obj].get("factor", 1.0)
            )

        losses += [_losses]
        if outputs[obj].get("nll_loss", False):
            nll_loss += [_losses.get("nll_loss", 0.0)]

    loss  = losses[0]['loss']  #length losses
    loss += losses[1]['loss']  #word_ins losses

    nll_loss = sum(l for l in nll_loss) if len(nll_loss) > 0 \
        else loss.new_tensor(0)

    # NOTE:
    # we don't need to use sample_size as denominator for the gradient
    # here sample_size is just used for logging
    sample_size = 1
    logging_output = {
        "loss": loss.data,
        "nll_loss": nll_loss.data,
        "ntokens": ntokens,
        "nsentences": nsentences,
        "sample_size": sample_size,
    }

    for l in losses:
        logging_output[l["name"]] = (
            utils.item(l["loss"].data / l["factor"])
            if reduce
            else l[["loss"]].data / l["factor"]
        )
    return loss, sample_size, logging_output