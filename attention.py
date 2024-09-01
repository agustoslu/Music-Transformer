# Code is adapted from: 
# https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch
# https://jaketae.github.io/study/relative-positional-encoding/

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RelativeMultiHeadedAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len=1024, dropout=0.1, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(RelativeMultiHeadedAttention, self).__init__()

        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_qk = d_model // num_heads
        self.device = device

        self.max_len = max_len
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.Er = nn.Parameter(torch.randn(max_len, self.d_qk, device=device))
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_len, max_len, device=device))
        )
        self._reset_parameters()
        self.to(device)

    def _reset_parameters(self):
        # Weight initialization: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_qk).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_qk = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def skew(self, QEr):
        padded = F.pad(QEr, (1, 0))
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        Srel = reshaped[:, :, 1:, :]
        return Srel
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        if seq_len > self.max_len:
            raise ValueError("Sequence length exceeds model capacity!")

        Q = self.split_heads(self.W_q(x))  # (batch_size, num_heads, seq_len, d_qk)
        K = self.split_heads(self.W_k(x))  # (batch_size, num_heads, seq_len, d_qk)
        V = self.split_heads(self.W_v(x))  # (batch_size, num_heads, seq_len, d_qk)

        start = self.max_len - seq_len
        Er = self.Er[start:, :]  # (seq_len, d_qk)
        QEr = torch.einsum('bhld, ld -> bhl', Q, Er)  # (batch_size, num_heads, seq_len)
        QEr = QEr.unsqueeze(-1)  # (batch_size, num_heads, seq_len, 1)
        Srel = self.skew(QEr)  # (batch_size, num_heads, seq_len, seq_len)
        K = K.permute(0, 1, 3, 2)  # (batch_size, num_heads, d_qk, seq_len)
        QK_t = torch.einsum('bhld, bhdm -> bhlm', Q, K) # (batch_size, num_heads, seq_len, seq_len)
        
        attn_scores = (QK_t + Srel) / math.sqrt(self.d_qk) # (batch_size, num_heads, seq_len, seq_len)

        if mask is not None:
            mask = self.mask[:seq_len, :seq_len].unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attn_output = torch.matmul(attn_probs, V)  # (batch_size, num_heads, seq_len, d_qk)

        attn_output = self.combine_heads(attn_output)  # (batch_size, seq_len, d_model)
        output = self.W_o(attn_output)  # (batch_size, seq_len, d_model)

        return self.dropout(output)