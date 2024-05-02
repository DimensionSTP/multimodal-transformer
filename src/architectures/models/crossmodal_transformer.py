from typing import Optional
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList

from fairseq.modules import SinusoidalPositionalEmbedding


class TransformerBlock(nn.Module):
    def __init__(
        self,
        model_dims: int,
        num_heads: int,
        attn_dropout: float,
        res_dropout: float,
    ) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(model_dims)
        self.attn = nn.MultiheadAttention(
            embed_dim=model_dims,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(res_dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        attn_mask: bool,
    ) -> torch.Tensor:
        query, key, value = [self.layer_norm(x) for x in (query, key, value)]
        mask = (
            self.get_future_mask(
                query=query,
                key=key,
            )
            if attn_mask
            else None
        )
        x = self.attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            attn_mask=mask,
        )[0]
        return query + self.dropout(x)

    def get_future_mask(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
    ) -> torch.Tensor:
        seq_len_query = query.shape[1]
        seq_len_key = seq_len_query if key is None else key.shape[1]

        future_mask = torch.ones(
            seq_len_query,
            seq_len_key,
        )
        future_mask = torch.triu(
            future_mask,
            diagonal=1,
        ).float()
        future_mask = future_mask.masked_fill(
            future_mask == float(1),
            float("-inf"),
        )
        return future_mask


class FeedForwardBlock(nn.Module):
    def __init__(
        self,
        model_dims: int,
        feed_forward_dims: int,
        relu_dropout: float,
        res_dropout: float,
    ) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(model_dims)
        self.feed_forward1 = nn.Linear(
            model_dims,
            feed_forward_dims,
        )
        self.dropout1 = nn.Dropout(relu_dropout)
        self.feed_forward2 = nn.Linear(
            feed_forward_dims,
            model_dims,
        )
        self.dropout2 = nn.Dropout(res_dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.layer_norm(x)
        forwarded = self.feed_forward2(
            self.dropout1(F.relu(self.feed_forward1(normalized)))
        )
        residual = normalized + self.dropout2(forwarded)
        return residual


class EncoderBlock(nn.Module):
    def __init__(
        self,
        model_dims: int,
        num_heads: int,
        feed_forward_dims: int,
        attn_dropout: float,
        relu_dropout: float,
        res_dropout: float,
    ) -> None:
        super().__init__()
        self.transformer = TransformerBlock(
            model_dims=model_dims,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            res_dropout=res_dropout,
        )
        self.feed_forward = FeedForwardBlock(
            model_dims=model_dims,
            feed_forward_dims=feed_forward_dims,
            res_dropout=res_dropout,
            relu_dropout=relu_dropout,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        attn_mask: bool,
    ) -> torch.Tensor:
        if key is not None:
            x = self.transformer(
                query=query,
                key=key,
                value=key,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        else:
            x = self.transformer(
                query=query,
                key=query,
                value=query,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        x = self.feed_forward(x)
        return x


class CrossModalTransformer(nn.Module):
    def __init__(
        self,
        model_dims: int,
        num_heads: int,
        num_layers: int,
        text_max_length: int,
        attn_dropout: float,
        relu_dropout: float,
        res_dropout: float,
        emb_dropout: float,
        attn_mask: bool,
        scale_embedding: bool,
    ) -> None:
        super().__init__()
        self.attn_mask = attn_mask
        self.emb_scale = math.sqrt(model_dims) if scale_embedding else 1.0
        self.pos_emb = SinusoidalPositionalEmbedding(
            embedding_dim=model_dims,
            padding_idx=0,
            init_size=text_max_length,
        )
        self.dropout = nn.Dropout(emb_dropout)

        layer = EncoderBlock(
            model_dims=model_dims,
            num_heads=num_heads,
            feed_forward_dims=model_dims * 4,
            attn_dropout=attn_dropout,
            relu_dropout=relu_dropout,
            res_dropout=res_dropout,
        )
        self.layers = self.get_clone(
            module=layer,
            iteration=num_layers,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # query settings
        pos_query = self.pos_emb(query[:, :, 0])
        query = self.emb_scale * query + pos_query
        query = self.dropout(query)

        # key settings
        if key is not None:
            pos_key = self.pos_emb(key[:, :, 0])
            key = self.emb_scale * key + pos_key
            key = self.dropout(key)

        for layer in self.layers:
            query = layer(
                query=query,
                key=key,
                key_padding_mask=key_padding_mask,
                attn_mask=self.attn_mask,
            )
        return query

    @staticmethod
    def get_clone(
        module: nn.Module,
        iteration: int,
    ) -> ModuleList:
        return ModuleList([copy.deepcopy(module) for _ in range(iteration)])
