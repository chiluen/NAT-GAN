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


class RNN_Discriminator():
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

##LSTM discriminator

class LSTM(BaseFairseqModel):
    def __init__(self, input_size, output_size, hidden_size, input_vocab):
        super().__init__()
        """
        在seq2seq NLP:
        input_size = 1 #只有一維, 代表純粹用數字去代表
        output_size = 1 
        hidden_size = 10 #h的維度
        input_vocab = vocabulary
        """
        self.input_size = input_size  
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.input_vocab = input_vocab

        self.lstm = nn.LSTM(input_size, hidden_size, 1)  #LSTM(數據向量, 隱藏元向量, number of layer)
        self.linear_1 = nn.Linear(hidden_size, 5)
        self.linear_2 = nn.Linear(5, output_size)
        self.layernorm = nn.LayerNorm(normalized_shape = 5)
        self.sigmoid = nn.Sigmoid()

    def init_hidden(self, bsz, device):
        h0 = torch.randn(1, bsz, self.hidden_size, device=device) #1是指number of layer
        c0 = torch.randn(1, bsz, self.hidden_size, device=device)
        return h0, c0

    def forward(self, src_tokens):
        bsz, max_src_len = src_tokens.size()
        h0, c0 = self.init_hidden(bsz, src_tokens.device)
        input = src_tokens.float().permute(1,0) #(seq-len, bsz)
        input = input.unsqueeze(-1)
        input = input.to(src_tokens.device)

        output, _ = self.lstm(input, (h0,c0)) # output dimension: (seq-len, bsz, hidden_size)
        hn = output[-1] # bsz,hidden_size
        output = []
        for i in range(bsz): #根據batch去output
            hn_temp = self.linear_1(hn[i])
            hn_temp = self.layernorm(hn_temp)
            hn_temp = self.linear_2(hn_temp)
            hn_temp = self.sigmoid(hn_temp) #轉到(0,1)
            hn_temp = 2*(hn_temp-0.5) #轉到(-1,1)

            output.append(hn_temp)
        output = torch.tensor(output, device=src_tokens.device, requires_grad=True)
        
        return output

class LSTM_Discriminator():
    def __init__(self, input_size, output_size, hidden_size, input_vocab):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.input_vocab = input_vocab
        self.model = LSTM(input_size, output_size, hidden_size, input_vocab).cuda()

  #在train時, 直接吃target sentence
    def train(self, target_Real, target_G):  
        optimizer = torch.optim.Adam(self.model.parameters())
        loss_function = nn.BCEWithLogitsLoss(reduction='mean') #這個先暫時用
        self.model.train()

    #先進行target_real
        optimizer.zero_grad()
        output_D = self.model(target_Real)
        output_D_target = torch.full([output_D.shape[0]],1).cuda()
        loss = loss_function(output_D, output_D_target)
        loss.backward()
        optimizer.step()

        #再進行target_G
        optimizer.zero_grad()
        output_D = self.model(target_G)
        #這邊錯誤的標籤用-1去做, 不要用0, 解決資訊問題
        output_D_target = torch.full([output_D.shape[0]],-1).cuda() 
        loss = loss_function(output_D, output_D_target)
        loss.backward()
        optimizer.step()

