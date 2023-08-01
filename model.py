import math

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import  GraphConv, HeteroGraphConv, RelGraphConv
from sklearn.metrics import precision_score
from torch.nn import init


class HINormer(nn.Module):
    def __init__(self, 
                 g, 
                 dataset, 
                 num_class, 
                 input_dimensions, 
                 embeddings_dimension=64, 
                 ffn_dim=64, 
                 cut=4, 
                 num_layers=8, 
                 num_gnns=2, 
                 nheads=2, 
                 dropout=0, 
                 attn_dropout=0,  
                 temper=1.0, 
                 num_type=4,  
                 device=0):

        super(HINormer, self).__init__()

        self.g = g
        self.device = device
        self.embeddings_dimension = embeddings_dimension
        self.num_layers = num_layers
        self.num_class = num_class
        self.num_gnns = num_gnns
        if dataset =='IMDB':
            self.num_rels = 6
        if dataset == 'DBLP' :
            self.num_rels = 6 # DBLP IMDB
        if dataset == 'Freebase':
            self.num_rels = 36 # Freebase
        if dataset == 'ACM':
            self.num_rels = 8 # ACM
        # self.num_bases =  
        self.hop = num_gnns
        self.seq_len = self.hop + 1
        self.temper = temper
        self.nheads = nheads
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, embeddings_dimension) for in_dim in input_dimensions])
        self.ffn_dim = ffn_dim
        self.cut_dim = embeddings_dimension // cut
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.GCNLayers = torch.nn.ModuleList()
        self.RGCNLayers = torch.nn.ModuleList()
        self.GTLayers = torch.nn.ModuleList()

        for layer in range(self.num_gnns):
            # self.GCNLayers.append(GraphConv(self.embeddings_dimension, 
            #                                 self.embeddings_dimension, 
            #                                 activation=F.relu,
            #                                 allow_zero_in_degree=True))
            self.RGCNLayers.append(RelGraphConv(self.embeddings_dimension, 
                                                self.embeddings_dimension, 
                                                self.num_rels, 
                                                regularizer='basis', 
                                                activation=F.relu, # 激活函数也可以换一换试试 LeakyReLU
                                                layer_norm=False, 
                                                self_loop=False))  # 调参数时记得把self_loop设置为False再试一试
        
        for layer in range(self.num_layers):
            self.GTLayers.append(EncoderLayer(self.cut_dim, 
                                              self.ffn_dim, 
                                              self.dropout,
                                              self.attn_dropout, 
                                              self.nheads)
            )
        
        self.final_ln = nn.LayerNorm(self.cut_dim)
        self.attn_layer = nn.Linear(2 * self.embeddings_dimension, 1)
        self.Drop = nn.Dropout(self.dropout)
        self.leaky = nn.LeakyReLU(0.01)
        # self.out_proj = nn.Linear(self.cut_dim, int(self.cut_dim/2))
        self.Prediction2 = nn.Linear(self.embeddings_dimension, self.num_class)
        # self.Linear1 = nn.Linear(self.cut_dim, self.num_class)

    def forward(self, features_list, idx, type_emb, node_type, etype, norm=False):
        batch_size = len(idx)
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        gh = torch.cat(h, 0) # IMDB:gh shape: torch.Size([21420, 256])  DBLP:gh shape: torch.Size([26128, 256])
        
        input_seq = torch.zeros((self.num_gnns+1, batch_size, self.embeddings_dimension)).to(self.device)
        input_seq[0] = gh[idx]

        for layer in range(self.num_gnns):
            # gh = self.GCNLayers[layer](self.g, gh)
            gh = self.RGCNLayers[layer](self.g, gh, etype)
            gh = self.Drop(gh)
            input_seq[layer+1] = gh[idx]

        h = input_seq.transpose(0,1)
        h = h.reshape(batch_size, -1, self.cut_dim) # 解耦dim cut_dim = hidden_dim // cut
        #============================
        tensor = h
        for enc_layer in self.GTLayers:
            tensor = enc_layer(tensor)
        
        output = self.final_ln(tensor)
        output = output.reshape(batch_size, -1, self.embeddings_dimension)
       
        target = output[:,0,:].unsqueeze(1).repeat(1, output.shape[1]-1, 1)
        # print('target shape:', target.shape)
        split_tensor = torch.split(output, [1, output.shape[1]-1], dim=1)
        node_tensor = split_tensor[0]
        neighbor_tensor = split_tensor[1]

        layer_atten = self.attn_layer(torch.cat((target, neighbor_tensor), dim=2))
        layer_atten = layer_atten / self.temper
        layer_atten = F.softmax(layer_atten, dim=1)
        
        neighbor_tensor = neighbor_tensor * layer_atten
    
        neighbor_tensor = torch.sum(neighbor_tensor, dim=1, keepdim=True)
        # Z = Z0 + sum(attn * neighbor_tensor)
        output = (node_tensor + neighbor_tensor).squeeze() # [1097, 256]        

        output = self.Prediction2(torch.relu(output))
        # if norm:
        #     output = output / (torch.norm(output, dim=1, keepdim=True)+1e-12)
        return output


# python run.py --dataset DBLP --len-seq 50 --dropout 0.5 --beta 0.1 --temperature 2
# python run_multi.py --dataset IMDB --len-seq 20 --beta 0.1 --temperature 0.1

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3) # [1097, 2, 10, 10]
    
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn] [1097, 2, 10, 128]
        
        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn] [1097, 10, 2, 128]
        x = x.view(batch_size, -1, self.num_heads * d_v) # [1097, 10, 256]
        
        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x) # 过LN
        y = self.self_attention(y, y, y, attn_bias) #MSA
        y = self.self_attention_dropout(y)
        x = x + y # residual connection

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x