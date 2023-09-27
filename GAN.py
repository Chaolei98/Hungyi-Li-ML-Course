#reference
#https://blog.csdn.net/qq_36693723/article/details/130332573?ops_request_misc=&request_id=&biz_id=102&utm_term=gan%20pytorch&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-2-130332573.142^v94^insert_down28v1&spm=1018.2226.3001.4187

import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, dim_input, dim_q, dim_k, dim_v):
        super(SelfAttention, self).__init__()

        self.dim_q, self.dim_k, self.dim_v = dim_q, dim_k, dim_v    #!! dim_q == dim_k

        self.transform_q = nn.Linear(dim_input, dim_q)
        self.transform_k = nn.Linear(dim_input, dim_k)
        self.transform_v = nn.Linear(dim_input, dim_v)

    def forward(self, x):   #shape of x:(batch_size, num_seq, dim_input)
        Q = self.transform_q(x) #shape of Q:(batch_size, num_seq, dim_q)
        K = self.transform_k(x) #shape of K:(batch_size, num_seq, dim_k)
        V = self.transform_v(x) #shape of V:(batch_size, num_seq, dim_v)

        attention_matrix = nn.Softmax(dim=-1)(torch.matmul(Q, K.permute(0, 2, 1)))  #shape of attention_matrix:(batch_size, num_seq, num_seq)
        attention_matrix /= math.sqrt(self.dim_q)
        out = torch.matmul(attention_matrix, V) #shape of out:(batch_size, num_seq, dim_v)

        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, dim_input, dim_q, dim_k, dim_v, num_head):
        super(MultiHeadAttention, self).__init__()

        self.dim_q, self.dim_k, self.dim_v = dim_q, dim_k, dim_v    #!! dim_q == dim_k
        self.head = num_head

        assert self.dim_k % self.head == 0, f'dim_k:{self.dim_k}不是num_head:{self.head}的整数倍'
        assert self.dim_v % self.head == 0, f'dim_k:{self.dim_v}不是num_head:{self.head}的整数倍'

        self.transform_q = nn.Linear(dim_input, dim_q, bias=False)
        self.transform_k = nn.Linear(dim_input, dim_k, bias=False)
        self.transform_v = nn.Linear(dim_input, dim_v, bias=False)
        self.transform_out = nn.Linear(dim_v, dim_v, bias=False)

    def forward(self, x):   #shape of x:(batch_size, num_seq, dim_input)
        Q = self.transform_q(x) #shape of Q:(batch_size, num_seq, dim_q)
        K = self.transform_k(x) #shape of K:(batch_size, num_seq, dim_k)
        V = self.transform_v(x) #shape of V:(batch_size, num_seq, dim_v)

        multi_Q = Q.reshape(x.shape[0], x.shape[1], self.head, self.dim_q//self.head).permute(0, 2, 1, 3) #(batch_size, num_head, num_seq, dim_q//head)
        multi_K = K.reshape(x.shape[0], x.shape[1], self.head, self.dim_k//self.head).permute(0, 2, 1, 3) #(batch_size, num_head, num_seq, dim_k//head)
        multi_V = V.reshape(x.shape[0], x.shape[1], self.head, self.dim_v//self.head).permute(0, 2, 1, 3) #(batch_size, num_head, num_seq, dim_v//head)

        attention_matrix = nn.Softmax(dim=-1)(torch.matmul(multi_Q, multi_K.permute(0, 1, 3, 2)))  #shape of attention_matrix:(batch_size, num_head, num_seq, num_seq)
        attention_matrix /= math.sqrt(self.dim_q)
        out = torch.matmul(attention_matrix, multi_V) #shape of out:(batch_size, num_head, num_seq, dim_v//head)

        out = out.permute(0, 2, 1, 3) #shape of out:(batch_size, num_seq, num_head, dim_v//head)
        out = out.reshape(x.shape[0], x.shape[1], -1) #shape of out:(batch_size, num_seq, dim_v)
        out = self.transform_out(out) #shape of out:(batch_size, num_seq, dim_v)

        return out


if __name__ == '__main__':
    num_head = 2
    batch_size = 16
    num_seq = 9
    dim_input = 3
    x = torch.rand(batch_size, num_seq, dim_input)
    print(f'x shape is:{x.shape}')

    self_attention = SelfAttention(dim_input, dim_q=8, dim_k=8, dim_v=12)
    output = self_attention(x)
    print(f'output shape of SelfAttention is:{output.shape}')

    multi_head_attention = MultiHeadAttention(dim_input, dim_q=8, dim_k=8, dim_v=12, num_head=num_head)
    output2 = multi_head_attention(x)
    print(f'output shape of MultiHeadSelfAttention is:{output2.shape}')
