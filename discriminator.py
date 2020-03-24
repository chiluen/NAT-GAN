import fairseq
import torch
import torch.nn as nn
from fairseq.models import BaseFairseqModel

#discriminator setting
#Discriminator
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

class FairseqRNNClassifier(BaseFairseqModel):

    def __init__(self, input_size, output_size, hidden_size, input_vocab):
        super().__init__()

        self.rnn = RNN(input_size=input_size,  #要把model放到cuda裡面
                       hidden_size=hidden_size,
                       output_size=output_size,
                       ).cuda()
        self.input_vocab = input_vocab #用來做one hot encoder的東西, 用target的dictionary來做
        self.register_buffer('one_hot_inputs', torch.eye(len(input_vocab)))


    def forward(self, src_tokens):

        #src_tokens.shape = (batch, src_len)
        #src_lengths.shape = (batch)
        bsz, max_src_len = src_tokens.size()

        #initialize hidden state
        hidden = self.rnn.initHidden()
        hidden = hidden.repeat(bsz, 1)  # expand for batched inputs
        hidden = hidden.to(src_tokens.device)  # move to GPU

        for i in range(max_src_len):
            input = self.one_hot_inputs[src_tokens[:, i].long()]# One-hot encode a batch of input characters.
            input = input.to(src_tokens.device)
            output, hidden = self.rnn(input, hidden)# Feed the input to our RNN.

        return output

class LSTM(BaseFairseqModel):
    def __init__(self, input_size, output_size, hidden_size, input_vocab):
        super().__init__()
        self.input_vocab = input_vocab
        self.input_size = input_size  #在seq2seq NLP, 這就是len(input_vocab)
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, 1)  #LSTM(數據向量, 隱藏元向量, number of layer)
        self.linear = nn.Linear(hidden_size, output_size)
        #self.one_hot_inputs = torch.eye(len(input_vocab))
        self.register_buffer('one_hot_inputs', torch.eye(len(input_vocab)))
    def init_hidden(self, bsz):
        h0 = torch.randn(1, bsz, self.hidden_size) #1是指number of layer
        c0 = torch.randn(1, bsz, self.hidden_size)
        return h0, c0

    def forward(self, src_tokens):
        bsz, _ = src_tokens.size()
        h0, c0 = self.init_hidden(bsz)
        input = self.one_hot_inputs[src_tokens.long()]# (bsz, seq-len, vocab-len)
        input = input.permute(1,0,2) #(seq-len, bsz, vocab-len)
        input = input.to(src_tokens.device)

        output, _ = self.lstm(input, (h0,c0))
        output = self.linear(output[-1], self.output_size) #這邊output[-1]其實就是hn
        return output

class Discriminator():
    def __init__(self, input_size, output_size, hidden_size, input_vocab):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.input_vocab = input_vocab
        self.model = FairseqRNNClassifier(input_size, output_size, hidden_size, input_vocab)

  #在train時, 直接吃target sentence
    def train(self, target_Real, target_G):  
        optimizer = torch.optim.Adam(self.model.parameters())
        loss_function = nn.BCEWithLogitsLoss(reduction='mean') #這個先暫時用
        self.model.train()

    #先進行target_real
        optimizer.zero_grad()
        output_D = self.model(target_Real)
        output_D_target = torch.full([output_D.shape[0],1],1).cuda()
        loss = loss_function(output_D, output_D_target)
        loss.backward()
        optimizer.step()

        #再進行target_G
        optimizer.zero_grad()
        output_D = self.model(target_G)
        #這邊錯誤的標籤用-1去做, 不要用0, 解決資訊問題
        output_D_target = torch.full([output_D.shape[0],1],-1).cuda() 
        loss = loss_function(output_D, output_D_target)
        loss.backward()
        optimizer.step()

