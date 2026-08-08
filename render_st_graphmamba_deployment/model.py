import math
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    history_len: int = 24
    horizon: int = 12
    step_minutes: int = 5
    d_model: int = 64
    d_state: int = 16
    n_layers: int = 3
    n_heads: int = 4
    dropout: float = 0.1


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, n_time_feats: int):
        super().__init__()
        self.value_proj = nn.Linear(1, d_model)
        self.time_proj = nn.Linear(n_time_feats, d_model)

    def forward(self, x, time_feats):
        B, T, N = x.shape
        v = self.value_proj(x.unsqueeze(-1))
        t = self.time_proj(time_feats).unsqueeze(2).expand(-1, -1, N, -1)
        return v + t


class GraphTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.adj_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, adj_bias):
        Bs, N, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(
            Bs, N, 3, self.n_heads, self.d_head
        ).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = attn + self.adj_scale * adj_bias.unsqueeze(0).unsqueeze(0)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(Bs, N, D)
        x = x + self.out_proj(out)
        x = x + self.ffn(self.norm2(x))
        return x


class SelectiveSSMBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        conv_kernel: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=d_model,
        )
        self.x_proj = nn.Linear(d_model, d_state * 2 + 1)
        self.A_log = nn.Parameter(
            torch.log(torch.rand(d_model, d_state) * 0.5 + 0.5)
        )
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        Bs, T, D = x.shape
        h = self.norm(x)
        xz = self.in_proj(h)
        x_in, gate = xz.chunk(2, dim=-1)

        x_conv = self.conv(x_in.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)

        proj = self.x_proj(x_conv)
        B_t, C_t, delta_t = torch.split(
            proj, [self.d_state, self.d_state, 1], dim=-1
        )
        delta_t = F.softplus(delta_t)

        A = -torch.exp(self.A_log)
        state = torch.zeros(
            Bs, D, self.d_state,
            device=x.device,
            dtype=x.dtype,
        )

        ys = []
        for t in range(T):
            dt = delta_t[:, t]
            dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
            dB = dt.unsqueeze(-1) * B_t[:, t].unsqueeze(1)
            state = dA * state + dB * x_conv[:, t].unsqueeze(-1)
            y_t = (state * C_t[:, t].unsqueeze(1)).sum(-1)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)
        y = y + self.D * x_conv
        y = y * F.silu(gate)
        y = self.out_proj(y)

        return x + self.dropout(y)


class STGraphMamba(nn.Module):
    def __init__(self, n_sensors: int, cfg: Config, n_time_feats: int = 4):
        super().__init__()
        self.cfg = cfg

        self.embed = InputEmbedding(cfg.d_model, n_time_feats)

        self.spatial_layers = nn.ModuleList([
            GraphTransformerLayer(
                cfg.d_model, cfg.n_heads, cfg.dropout
            )
            for _ in range(cfg.n_layers)
        ])

        self.temporal_layers = nn.ModuleList([
            SelectiveSSMBlock(
                cfg.d_model, cfg.d_state, dropout=cfg.dropout
            )
            for _ in range(cfg.n_layers)
        ])

        self.fuse_gate = nn.ModuleList([
            nn.Linear(cfg.d_model * 2, cfg.d_model)
            for _ in range(cfg.n_layers)
        ])

        self.reg_head = nn.Linear(cfg.d_model, cfg.horizon)
        self.cls_head = nn.Linear(
            cfg.d_model, cfg.horizon * 4
        )

    def forward(self, x, time_feats, adj_bias):
        B, T, N = x.shape
        h = self.embed(x, time_feats)

        for spatial, temporal, gate in zip(
            self.spatial_layers,
            self.temporal_layers,
            self.fuse_gate,
        ):
            h_sp = h.reshape(B * T, N, -1)
            h_sp = spatial(h_sp, adj_bias)
            h_sp = h_sp.reshape(B, T, N, -1)

            h_tp = h.permute(0, 2, 1, 3).reshape(B * N, T, -1)
            h_tp = temporal(h_tp)
            h_tp = h_tp.reshape(B, N, T, -1).permute(0, 2, 1, 3)

            fused = gate(torch.cat([h_sp, h_tp], dim=-1))
            h = h + torch.tanh(fused)

        h_last = h[:, -1]
        reg_out = self.reg_head(h_last)
        cls_out = self.cls_head(h_last).reshape(
            B, N, self.cfg.horizon, 4
        )

        return reg_out, cls_out


def load_model(checkpoint_path: str, n_sensors: int = 207):
    cfg = Config()

    model = STGraphMamba(
        n_sensors=n_sensors,
        cfg=cfg,
        n_time_feats=4,
    )

    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state, strict=True)
    model.eval()

    return model
