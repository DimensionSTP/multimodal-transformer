import torch
import torch.nn as nn
import torch.nn.functional as F

from .crossmodal_transformer import CrossModalTransformer


class MultiModalTransformer(nn.Module):
    def __init__(
        self,
        model_dims: int,
        num_heads: int,
        num_layers: int,
        audio_dims: int,
        text_dims: int,
        text_max_length: int,
        num_labels: int,
        attn_dropout: float,
        relu_dropout: float,
        res_dropout: float,
        emb_dropout: float,
        out_dropout: float,
        attn_mask: bool,
        scale_embedding: bool,
    ) -> None:
        super().__init__()
        combined_dim = model_dims * 2

        self.audio_encoder = nn.Conv1d(
            audio_dims,
            model_dims,
            3,
            padding=1,
            bias=True,
        )
        self.text_encoder = nn.Conv1d(
            text_dims,
            model_dims,
            3,
            padding=1,
            bias=True,
        )

        kwargs = {
            "model_dims": model_dims,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "text_max_length": text_max_length,
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

        self.fc1 = nn.Linear(
            combined_dim,
            combined_dim,
        )
        self.fc2 = nn.Linear(
            combined_dim,
            combined_dim,
        )
        self.dropout = nn.Dropout(out_dropout)
        self.out_layer = nn.Linear(
            combined_dim,
            num_labels,
        )

    def forward(
        self,
        audio: torch.Tensor,
        audio_mask: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        if audio.dim() <= 2:
            audio = audio.unsqueeze(0)
        if text.dim() <= 2:
            text = text.unsqueeze(0)
        audio = self.audio_encoder(audio.transpose(1, 2)).transpose(1, 2)
        text = self.text_encoder(text.transpose(1, 2)).transpose(1, 2)
        audio = self.audio_text(
            query=audio,
            key=text,
            key_padding_mask=text_mask,
        )
        text = self.text_audio(
            query=text,
            key=audio,
            key_padding_mask=audio_mask,
        )
        audio = self.audio_self(
            query=audio,
            key=audio,
            key_padding_mask=audio_mask,
        )
        text = self.text_self(
            query=text,
            key=text,
            key_padding_mask=text_mask,
        )
        features = torch.cat(
            [
                audio,
                text,
            ],
            dim=2,
        )
        pooler_output = features[:, 0, :].squeeze()
        out = pooler_output + self.fc2(self.dropout(F.relu(self.fc1(pooler_output))))
        if out.dim() == 1:
            out = out.unsqueeze(0)
        return self.out_layer(out)

    @staticmethod
    def get_network(**kwargs) -> nn.Module:
        return CrossModalTransformer(
            model_dims=kwargs["model_dims"],
            num_heads=kwargs["num_heads"],
            num_layers=kwargs["num_layers"],
            text_max_length=kwargs["text_max_length"],
            attn_dropout=kwargs["attn_dropout"],
            relu_dropout=kwargs["relu_dropout"],
            res_dropout=kwargs["res_dropout"],
            emb_dropout=kwargs["emb_dropout"],
            attn_mask=kwargs["attn_mask"],
            scale_embedding=["scale_embedding"],
        )
