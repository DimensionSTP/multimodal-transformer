import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList

from fairseq.modules import SinusoidalPositionalEmbedding


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, attn_dropout: float, res_dropout: float
    ):
        super(TransformerBlock, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=attn_dropout, batch_first=True
        )
        self.dropout = nn.Dropout(res_dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask=None,
        attn_mask: bool = True,
    ):
        query, key, value = [self.layer_norm(x) for x in (query, key, value)]
        mask = self.get_future_mask(query, key) if attn_mask else None
        x = self.attn(
            query, key, value, key_padding_mask=key_padding_mask, attn_mask=mask
        )[0]
        return query + self.dropout(x)

    def get_future_mask(
        self, query: torch.Tensor, key: torch.Tensor = None
    ) -> torch.Tensor:
        seq_len_query = query.shape[1]
        seq_len_key = seq_len_query if key is None else key.shape[1]

        future_mask = torch.ones(seq_len_query, seq_len_key, device=query.device)
        future_mask = torch.triu(future_mask, diagonal=1).float()
        future_mask = future_mask.masked_fill(future_mask == float(1), float("-inf"))
        return future_mask


class FeedForwardBlock(nn.Module):
    def __init__(
        self, d_model: int, d_feed_forward: int, relu_dropout: float, res_dropout: float
    ):
        super(FeedForwardBlock, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.feed_forward1 = nn.Linear(d_model, d_feed_forward)
        self.dropout1 = nn.Dropout(relu_dropout)
        self.feed_forward2 = nn.Linear(d_feed_forward, d_model)
        self.dropout2 = nn.Dropout(res_dropout)

    def forward(self, x):
        normalized = self.layer_norm(x)
        forwarded = self.feed_forward2(
            self.dropout1(F.relu(self.feed_forward1(normalized)))
        )
        residual = normalized + self.dropout2(forwarded)
        return residual


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_feed_forward: int,
        attn_dropout: float,
        relu_dropout: float,
        res_dropout: float,
    ):
        super(EncoderBlock, self).__init__()
        self.transformer = TransformerBlock(d_model, n_heads, attn_dropout, res_dropout)
        self.feed_forward = FeedForwardBlock(
            d_model, d_feed_forward, res_dropout, relu_dropout
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        key_padding_mask=None,
        attn_mask: bool = True,
    ):
        if key is not None:
            x = self.transformer(
                query, key, key, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
        else:
            x = self.transformer(
                query,
                query,
                query,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        x = self.feed_forward(x)
        return x


class CrossModalTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        attn_dropout: float,
        relu_dropout: float,
        res_dropout: float,
        emb_dropout: float,
        attn_mask: bool,
        scale_embedding: bool,
    ):
        super(CrossModalTransformer, self).__init__()
        self.attn_mask = attn_mask
        self.emb_scale = math.sqrt(d_model) if scale_embedding else 1.0
        self.pos_emb = SinusoidalPositionalEmbedding(d_model, 0, init_size=249)
        self.dropout = nn.Dropout(emb_dropout)

        layer = EncoderBlock(
            d_model=d_model,
            n_heads=n_heads,
            d_feed_forward=d_model * 4,
            attn_dropout=attn_dropout,
            relu_dropout=relu_dropout,
            res_dropout=res_dropout,
        )
        self.layers = self.get_clone(layer, n_layers)

    def forward(self, query: torch.Tensor, key: torch.Tensor, key_padding_mask=None):
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
                query, key, key_padding_mask=key_padding_mask, attn_mask=self.attn_mask
            )
        return query

    @staticmethod
    def get_clone(module: nn.Module, iteration: int) -> ModuleList:
        return ModuleList([copy.deepcopy(module) for _ in range(iteration)])


class MultiModalTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 8,
        n_layers: int = 4,
        d_audio: int = 1024,
        d_text: int = 1024,
        n_classes: int = 7,
        attn_dropout: float = 0.25,
        relu_dropout: float = 0.0,
        res_dropout: float = 0.0,
        emb_dropout: float = 0.3,
        out_dropout: float = 0.1,
        attn_mask: bool = True,
        scale_embedding: bool = True,
    ):
        super(MultiModalTransformer, self).__init__()
        combined_dim = d_model * 2

        self.audio_encoder = nn.Conv1d(d_audio, d_model, 3, padding=1, bias=True)
        self.text_encoder = nn.Conv1d(d_text, d_model, 3, padding=1, bias=True)

        kwargs = {
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "attn_dropout": attn_dropout,
            "relu_dropout": relu_dropout,
            "res_dropout": res_dropout,
            "emb_dropout": emb_dropout,
            "attn_mask": attn_mask,
            "scale_embedding": scale_embedding,
        }

        self.audio_text = self.get_network(**kwargs)
        self.text_audio = self.get_network(**kwargs)
        self.audio_self = self.get_network(**kwargs)
        self.text_self = self.get_network(**kwargs)

        self.fc1 = nn.Linear(combined_dim, combined_dim)
        self.fc2 = nn.Linear(combined_dim, combined_dim)
        self.dropout = nn.Dropout(out_dropout)
        self.out_layer = nn.Linear(combined_dim, n_classes)

    def forward(
        self,
        audio: torch.Tensor,
        audio_mask: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
    ):
        audio = self.audio_encoder(audio.transpose(1, 2)).transpose(1, 2)
        text = self.text_encoder(text.transpose(1, 2)).transpose(1, 2)
        audio = self.audio_text(query=audio, key=text, key_padding_mask=text_mask)
        text = self.text_audio(query=text, key=audio, key_padding_mask=audio_mask)
        audio = self.audio_self(query=audio, key=audio, key_padding_mask=audio_mask)
        text = self.text_self(query=text, key=text, key_padding_mask=text_mask)
        features = torch.cat([audio, text], dim=2)
        pooler_output = features[:, 0, :].squeeze()
        out = pooler_output + self.fc2(self.dropout(F.relu(self.fc1(pooler_output))))

        return self.out_layer(out)

    @staticmethod
    def get_network(**kwargs) -> nn.Module:
        return CrossModalTransformer(
            d_model=kwargs["d_model"],
            n_heads=kwargs["n_heads"],
            n_layers=kwargs["n_layers"],
            attn_dropout=kwargs["attn_dropout"],
            relu_dropout=kwargs["relu_dropout"],
            res_dropout=kwargs["res_dropout"],
            emb_dropout=kwargs["emb_dropout"],
            attn_mask=kwargs["attn_mask"],
            scale_embedding=["scale_embedding"],
        )
