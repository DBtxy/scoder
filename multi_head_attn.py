import torch
from torch import nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, hidden_size, dropout=0.1):
        assert hidden_size % num_heads == 0

        self.head_size = hidden_size // num_heads
        self.num_heads = num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        # 这个注意是合并之后过的一个整体的线性层
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, dropout=None, mask=None):
        """
        :param q: shape[batch_size, seq_length, hidden_size]
        :param k: shape[batch_size, seq_length, hidden_size]
        :param v: shape[batch_size, seq_length, hidden_size]
        :return:
        """

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        (batch_size, seq_length, hidden_size) = q.shape

        # 要把q k v 变成 [batch_size, seq_length, nums_heads, head_size]，需要一个torch的reshape的函数
        q = q.view(batch_size, -1, self.num_heads, self.head_size)
        k = k.view(batch_size, -1, self.num_heads, self.head_size)
        v = v.view(batch_size, -1, self.num_heads, self.head_size)
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_length, head_size]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
        attn_score = F.softmax(attn_score, dim=-1)
        # 需要加dropout在attn哦
        if dropout:
            attn_score = self.dropout(attn_score)

        if mask:
            attn_score = attn_score.masked_fill_(mask == 0, -1e9)

        o1 = torch.matmul(attn_score, v)
        o1 = o1.transpose(1, 2).view(batch_size, -1, hidden_size)
        o2 = self.o_proj(o1)
        return o2
