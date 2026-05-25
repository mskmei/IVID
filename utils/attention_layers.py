import torch
from torch import nn


def _key_padding_mask(mask):
    mask = mask.to(dtype=torch.bool)
    empty_rows = ~mask.any(dim=1)
    if empty_rows.any():
        mask = mask.clone()
        mask[empty_rows, 0] = True
    return ~mask


class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_size=1024, num_heads=16, dropout=0.3):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm_attn = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.norm_ffn = nn.LayerNorm(hidden_size)

    def forward(self, inputs, mask):
        padding_mask = _key_padding_mask(mask)
        attended, _ = self.attention(
            inputs,
            inputs,
            inputs,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        hidden = self.norm_attn(inputs + self.dropout(attended))
        output = self.norm_ffn(hidden + self.dropout(self.ffn(hidden)))
        return output


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size=1024, num_heads=16, dropout=0.3):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm_cross = nn.LayerNorm(hidden_size)
        self.norm_self = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.norm_ffn = nn.LayerNorm(hidden_size)

    def forward(self, query_states, query_mask, guide_states, guide_mask):
        guide_padding_mask = _key_padding_mask(guide_mask)
        query_padding_mask = _key_padding_mask(query_mask)

        cross_attended, _ = self.cross_attention(
            query_states,
            guide_states,
            guide_states,
            key_padding_mask=guide_padding_mask,
            need_weights=False,
        )
        hidden = self.norm_cross(query_states + self.dropout(cross_attended))

        self_attended, _ = self.self_attention(
            hidden,
            hidden,
            hidden,
            key_padding_mask=query_padding_mask,
            need_weights=False,
        )
        hidden = self.norm_self(hidden + self.dropout(self_attended))
        output = self.norm_ffn(hidden + self.dropout(self.ffn(hidden)))
        return output
