import fairseq
import torch
import torch.nn as nn
from fairseq.models import BaseFairseqModel

#discriminator setting
#Discriminator
class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, input_vocab):
        super().__init__()
        """
        在seq2seq NLP:
        input_size = embed_dim #也就代表embedding的dimension, 預設是256
        output_size = 1 
        hidden_size = 10 #h的維度
        input_vocab = vocabulary
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.input_vocab = input_vocab
        self.n_layers = 3

        self.embed = nn.Embedding(len(input_vocab), input_size)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=self.n_layers)  #LSTM(數據向量, 隱藏元向量, number of layer)
        self.linear = nn.Linear(hidden_size, output_size)
        self.layernorm = nn.LayerNorm(normalized_shape = 5)
        self.sigmoid = nn.Sigmoid()

    def init_hidden(self, bsz, device):
        h0 = torch.randn(self.n_layers, bsz, self.hidden_size, device=device) 
        return h0

        #(seq_len, batch_size, input_size)
    def forward(self, src_tokens):
        bsz, max_src_len = src_tokens.size()
        h0 = self.init_hidden(bsz, src_tokens.device)

        src_embed = self.embed(src_tokens) #(bsz, seq-len) -> (bsz, seq-len, embed_dim)
        src_embed = src_embed.float().permute(1,0,2) #(seq_len, bsz, hidden_size)
        src_embed = src_embed.to(src_tokens.device)

        output, _ = self.rnn(src_embed, h0) # output dimension: (seq-len, bsz, hidden_size)

        hn = output[-1] # bsz,hidden_size
        output = self.linear(hn)
        output = self.sigmoid(output)
        output = output.squeeze() #把維度為1的去除
        
        return output
        
class RNN_Discriminator():
    def __init__(self, input_size, output_size, hidden_size, input_vocab):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.input_vocab = input_vocab
        self.model = RNN(input_size, output_size, hidden_size, input_vocab).cuda()

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
        output_D_target = torch.full([output_D.shape[0]],0).cuda() 
        loss = loss_function(output_D, output_D_target)
        loss.backward()
        optimizer.step()

##LSTM discriminator

class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, input_vocab):
        super().__init__()
        """
        在seq2seq NLP:
        input_size = embed_dim #也就代表embedding的dimension, 預設是200
        output_size = 1 
        hidden_size = 10 #h的維度
        input_vocab = vocabulary
        """
        self.input_size = input_size  
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.input_vocab = input_vocab

        self.embed = nn.Embedding(len(input_vocab), input_size)
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

        src_embed = self.embed(src_tokens) #(bsz, seq-len) -> (bsz, seq-len, embed_dim)
        src_embed = src_embed.float().permute(1,0,2)
        src_embed = src_embed.to(src_tokens.device)

        output, _ = self.lstm(src_embed, (h0,c0)) # output dimension: (seq-len, bsz, hidden_size)
        hn = output[-1] # bsz,hidden_size

        output = self.linear_1(hn)
        output = self.layernorm(output)
        output = self.linear_2(output)
        output = self.sigmoid(output)
        output = output.squeeze() #把維度為1的去除
        
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
        output_D_target = torch.full([output_D.shape[0]],0).cuda() 
        loss = loss_function(output_D, output_D_target)
        loss.backward()
        optimizer.step()

