
import torch
import torch.nn as nn
import torch.nn.functional as F

class FTTransformer(nn.Module):
    def __init__(self, categories, num_continuous, dim=128, depth=6, heads=8, attn_dropout=0.1, ff_dropout=0.1):
        super().__init__()
        self.num_categories = sum(categories)
        self.num_continuous = num_continuous
        self.dim = dim

        self.cat_embed = nn.ModuleList([
            nn.Embedding(num_cat, dim) for num_cat in categories
        ])

        self.cont_embed = nn.Parameter(torch.randn(num_continuous, dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=heads,
        dropout=attn_dropout,
        dim_feedforward=dim*4,
        activation='gelu',
        batch_first=True
    ),
    num_layers=depth
)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, x_cat, x_cont):
        B = x_cat.size(0)

        x_cat = torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embed)], dim=1)
        x_cont = (x_cont.unsqueeze(-1) * self.cont_embed).to(x_cat.dtype)

        tokens = torch.cat([x_cat, x_cont], dim=1)
        cls_tokens = self.cls_token.expand(B, 1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)

        x = self.transformer(tokens)
        x = self.norm(x[:, 0])
        return nn.Identity()(self.head(x))
