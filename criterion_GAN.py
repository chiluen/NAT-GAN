##criterion_GAN
#fairseq/criterions/nat_loss.py 
#train.train -> train_step_GAN -> train_step_task -> criterion_GAN
import math
import torch.nn.functional as F
import torch
from torch import Tensor

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


def _compute_loss(outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0):
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
    if masks is not None:
        outputs, targets = outputs[masks], targets[masks]

    if masks is not None and not masks.any():
        nll_loss = torch.tensor(0)
        loss = nll_loss

    else:
        logits = F.log_softmax(outputs, dim=-1)
        if targets.dim() == 1:
            losses = F.nll_loss(logits, targets.to(logits.device), reduction='none')

        else:  # soft-labels
            losses = F.kl_div(logits, targets.to(logits.device), reduction='none')
            losses = losses.sum(-1)
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
def change_output2target(outputs): 
    outputs_argmax = outputs['word_ins']['out']
    outputs_argmax = torch.max(outputs_argmax[0], dim = 1)[1].reshape(1,-1)
    for i in range(1,outputs_argmax.shape[0]):
        indice = torch.max(outputs_argmax[i], dim = 1)[1].reshape(1,-1)
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

    #把outputs轉成跟target token一樣的形式
    outputs_argmax = change_output2target(outputs)

    #input_size, output_size, hidden_size, input_vocab
    input_size = len(task.target_dictionary)
    output_size = 1
    hidden_size = 10
    input_vocab = task.target_dictionary
    #RNN_Discriminator = FairseqRNNClassifier(input_size, output_size, hidden_size, input_vocab)
    #True_prob = RNN_Discriminator(outputs_argmax) #len of ans: batch_size, 代表其有為每一個sentence 做分類
    True_prob = discriminator.model(outputs_argmax) #len of True_prob = batch_size


    for obj in outputs:
        if outputs[obj].get("loss", None) is None:
            _losses = _compute_loss(
                outputs[obj].get("out"),
                outputs[obj].get("tgt"),
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
    #第一個是length 的 loss, 第二個是output的loss
    #要去testing length與output的loss的scalar, 再discriminator
    if args.only_G == False:
        loss = losses[0]['loss']
        loss += True_prob.mean() * losses[1]['loss'] * -1 #之所以乘上-1, 是因為如果Discriminator覺得好的話, 我們要減少generator的loss

    else:
        loss = losses[0]['loss'] + losses[1]['loss'] ##只有Generator時

    #print("This loss is length loss", losses[0]['loss'])
    #print("This loss is target loss", losses[1]['loss'])
    #loss = sum(l["loss"] for l in losses)  #這個是已經把batch的東西處理好了, 只剩下word以及length的losses
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