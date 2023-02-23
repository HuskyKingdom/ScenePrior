import math
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l


class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                DecoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size) # need to change

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)


            # visulization
            # self-attention weights of decoder
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # Encoder-Decoder self-attention weights
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights

        return self.dense(X), state

   
    def attention_weights(self):
        return self._attention_weights

class DecoderBlock(nn.Module):

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,
                                   num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):

        enc_outputs, enc_valid_lens = state[0], state[1]
        # 训练阶段，输出序列的所有词元都在同一时间处理，
        # 因此state[2][self.i]初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training: # from super()
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens的开头:(batch_size,num_steps),
            # 其中每一行是[1,2,...,num_steps]
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        # self-attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # encoder-decoder attention
        # enc_outputs的开头:(batch_size,num_steps,num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state




class EncoderBlock(nn.Module): # individul encoder block

    def __init__(self,key_size,query_size,value_size,num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout, use_bias=False,**kwargs):
        super(EncoderBlock,self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size,query_size,value_size,num_hiddens,num_heads,dropout,use_bias)
        self.addnorm1 = AddNorm(norm_shape,dropout)
        self.ffn = PositionWiseFFN(ffn_num_input,ffn_num_hiddens,num_hiddens)
        self.addnorm2 = AddNorm(norm_shape,dropout)

    def forward(self,X,valid_lends):
        Y = self.addnorm1(X,self.attention(X,X,X,valid_lends)) # self-attention
        return self.addnorm2(Y,self.ffn(Y))


class TransformerEncoder(d2l.Encoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):

        # Embedding is scaled by sqrt of embedding dimensions to match the position encoding (-1,1)
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X


class AddNorm(nn.Module):

    def __init__(self,normalized_shape,dropout,**kwargs):
        super(AddNorm,self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self,X,Y):
        return self.ln(self.dropout(Y) + X)


class PositionWiseFFN(nn.Module):

    def __init__(self,ffn_num_input,ffn_num_hiddens,ffn_num_output,**kwargs):
        super(PositionWiseFFN,self).__init__(**kwargs)

        self.dense1 = nn.Linear(ffn_num_input,ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens,ffn_num_output)

    def forward(self,X):
        return self.dense2(self.relu(self.dense1(X)))


def transpose_qkv(X,num_heads): # input (b,num_QKV,len_QKV) -> (b,num_QKV,num_hidden)

    X = X.reshape(X.shape[0],X.shape[1],num_heads,-1) # (b,num_QKV,num_heads,num_hidden/num_heads) 

    X = X.permute(0,2,1,3) # (b,num_heads,num_QKV,num_hidden/num_heads) 

    return X.reshape(-1,X.shape[2],X.shape[3]) # (b * num_heads,num_QKV,num_hidden/num_heads) 


def transpose_output(X,num_heads):

    # inversing transpose_qkv
    X = X.reshape(-1,num_heads,X.shape[1],X.shape[2])

    X = X.permute(0,2,1,3)

    return X.reshape(X.shape[0],X.shape[1],-1)



class MultiHeadAttention(nn.Module):

    def __init__(self,key_size,query_size,value_size,num_hiddens,num_heads,dropout,bias=False,**kwargs):

        super(MultiHeadAttention,self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)

        self.W_q = nn.Linear(query_size,num_hiddens,bias=bias) # mapping 
        self.W_k = nn.Linear(key_size,num_hiddens,bias=bias)
        self.W_v = nn.Linear(value_size,num_hiddens,bias=bias)
        self.W_o = nn.Linear(num_hiddens,num_hiddens,bias=bias)

    

    def forward(self,queries,keys,values,valid_lens):

        queries = transpose_qkv(self.W_q(queries),self.num_heads) # parallel multi-head computing
        keys = transpose_qkv(self.W_k(keys),self.num_heads)
        values = transpose_qkv(self.W_v(values),self.num_heads)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens,repeats=self.num_heads,dim=0)

        output = self.attention(queries,keys,values,valid_lens)

        output_concat = transpose_output(output,self.num_heads)
        return self.W_o(output_concat)
    