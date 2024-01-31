import torch
import torch.nn as nn
import torch.nn.functional as F

from .crossmodal_transformer import CrossModalTransformer


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
    ) -> None:
        super().__init__()
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
    ) -> torch.Tensor:
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
