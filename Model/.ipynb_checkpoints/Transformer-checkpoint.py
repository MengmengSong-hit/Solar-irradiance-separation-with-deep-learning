import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model,ffn_hidden, n_head):#d_model是输入的特征维度；n_head是头的个数
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        # 1. dot product with weight matrices
        q, k, v = F.sigmoid(self.w_q(q)),F.sigmoid(self.w_k(k)),F.sigmoid(self.w_v(v))
        #q, k, v = F.relu(self.w_q(q)),F.relu(self.w_k(k)),F.relu(self.w_v(v))
        #q, k, v = F.tanh(self.w_q(q)),F.tanh(self.w_k(k)),F.tanh(self.w_v(v))
        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)
        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v)
        
        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out) 

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out #[bat,向量的个数,d_model]

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)#得到的tensor [bat,注意力个数，向量的个数，分割后特征向量的维度]
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose k_t：[bat,注意力个数，分割后特征向量的维度,向量的个数]
        #print(q.shape)
        #print(k_t.shape)
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product
        #score = (q.mm(k_t)) #/ torch.sqrt(d_tensor)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)#[bat,注意力个数，向量的个数,向量的个数]

        # 4. multiply with Value
        v = score@v # [bat,注意力个数，向量的个数,分割后特征向量的维度]

        return v, score

#LayerNorm
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension. 

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out
    
class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        return x
    
#一层EncoderLayer
class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, ffn_hidden, n_head):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model,ffn_hidden=ffn_hidden, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden)
        self.norm2 = LayerNorm(d_model=ffn_hidden)

    def forward(self, x):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x)
        # 2. add and norm
        x = self.norm1(x + _x)
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        # 4. add and norm
        x = self.norm2(x + _x)
        return x
    
class Transformer(nn.Module):
    
    def __init__(self, d_model, ffn_hidden, n_head, n_layers, input_size,device):#d_model是输入的特征维度；ffn_hidden是FFN中dense的维度；n_head是头的个数， n_layers是encod额layer的层数
        super().__init__()
        self.layer1 =EncoderLayer(d_model=d_model,ffn_hidden=ffn_hidden,n_head=n_head)                   
        self.dense1=nn.Linear(ffn_hidden*15,2)
        #self.activation=nn.ReLU()
        #self.dense2=nn.Linear(input_size,2)
        
    def forward(self, x):
        x = self.layer1(x) 
        x=x.flatten(1)
        forecast=self.dense1(x)
        #feature_vector=self.dense1(x).squeeze()
        #feature_vector=self.activation(feature_vector)
        #forecast = self.dense2(feature_vector)
        
        summing=forecast [:,0].add(forecast[:,1])
        kd_ratio=forecast[:,0].div(summing)
        kb_ratio=forecast[:,1].div(summing)
        kd_ratio=kd_ratio.unsqueeze(-1)
        kb_ratio=kb_ratio.unsqueeze(-1)
        return kd_ratio,kb_ratio,x