"""
Neural network model architectures for terrain friction estimation.
To modify architecture hyperparameters, edit the corresponding *_cfg.yaml.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== Baseline models =====

class FCNNBaseline(nn.Module):
    """Fully-connected neural network that flattens the entire horizon."""

    def __init__(self, horizon_size: int, hidden: tuple, dropout: float):
        super().__init__()
        H = int(horizon_size)
        input_dim = H * (6 + 12 * 3)  # H * 42
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h, elementwise_affine=False),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, base_seq, joint_seq):
        B, H, _ = base_seq.shape
        x = torch.cat([base_seq.reshape(B, H * 6),
                        joint_seq.reshape(B, H * 36)], dim=-1)
        return F.softplus(self.net(x).squeeze(-1))


class GRUBaseline(nn.Module):
    """GRU encoder with an MLP regression head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        head_hidden: int,
        dropout: float,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.LayerNorm(head_hidden, elementwise_affine=False),
            nn.LeakyReLU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, base_seq, joint_seq):
        B, H, _ = base_seq.shape
        x = torch.cat([base_seq, joint_seq.reshape(B, H, 36)], dim=-1)  # (B,H,42)
        _, hN = self.gru(x)       # hN: (num_layers, B, hidden_dim)
        h_last = hN[-1]           # (B, hidden_dim)
        return F.softplus(self.head(h_last).squeeze(-1))


class TransformerEstimator(nn.Module):
    """Transformer encoder with learnable positional embedding."""

    def __init__(
        self,
        horizon_size: int,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.H = int(horizon_size)

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, self.H, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.LayerNorm(dim_ff, elementwise_affine=False),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim_ff, 1),
        )

    def forward(self, base_seq: torch.Tensor, joint_seq: torch.Tensor) -> torch.Tensor:
        B, H, _ = base_seq.shape
        x = torch.cat([base_seq, joint_seq.reshape(B, H, 36)], dim=-1)  # (B,H,42)
        x = self.input_proj(x) + self.pos_emb[:, :H, :]                 # (B,H,d_model)
        x = self.encoder(x)                                             # (B,H,d_model)
        h = x[:, -1, :]                                                 # (B,d_model)
        return F.softplus(self.head(h).squeeze(-1))


# ===== Graph network model =====

def _make_leg_seq(joint_seq: torch.Tensor) -> torch.Tensor:
    """Reshape (B,H,12,3) joint tensor into (B,H,4,9) per-leg tensor."""
    B, H, J, C = joint_seq.shape
    assert J == 12 and C == 3
    return joint_seq.view(B, H, 4, 3, 3).reshape(B, H, 4, 9)


class RecGNN(nn.Module):
    """Recurrent Graph Neural Network with quadruped-like-topology message passing.
    Graph topology: 1 base node (center) connected to 4 leg nodes.
    """

    def __init__(
        self,
        horizon_size: int,
        emb_dim: int,
        msg_dim: int,
        hidden_mlp: int,
        mp_steps_per_t: int,
        dropout: float,
        gru_hidden: int,
        gru_layers: int,
    ):
        super().__init__()
        self.H = int(horizon_size)
        self.N = 5   # graph nodes: 1 base + 4 legs
        self.D = int(emb_dim)
        self.msg_dim = int(msg_dim)
        self.mp_steps = int(mp_steps_per_t)
        self.state_dim = self.N * self.D  # concat pooling

        # node encoders
        self.base_enc = nn.Sequential(
            nn.Linear(6, hidden_mlp), nn.ReLU(), nn.Linear(hidden_mlp, self.D),
        )
        self.leg_enc = nn.Sequential(
            nn.Linear(9, hidden_mlp), nn.ReLU(), nn.Linear(hidden_mlp, self.D),
        )

        # message passing
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * self.D, hidden_mlp), nn.ReLU(),
            nn.Linear(hidden_mlp, self.msg_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(self.D + self.msg_dim, hidden_mlp),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_mlp, self.D),
        )

        # temporal encoder
        gh = int(gru_hidden)
        self.gru = nn.GRU(
            input_size=self.state_dim,
            hidden_size=gh,
            num_layers=int(gru_layers),
            batch_first=True,
        )

        # regression head
        self.head = nn.Sequential(
            nn.Linear(gh, hidden_mlp),
            nn.LayerNorm(hidden_mlp, elementwise_affine=True),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_mlp, 1),
        )

    def _encode_nodes(self, base_seq: torch.Tensor, joint_seq: torch.Tensor) -> torch.Tensor:
        """Encode raw inputs into per-node embeddings."""
        leg_seq = _make_leg_seq(joint_seq)                                 # (B,H,4,9)
        base_e = self.base_enc(base_seq)                                   # (B,H,D)
        leg_e = self.leg_enc(leg_seq)                                      # (B,H,4,D)
        return torch.cat([base_e.unsqueeze(2), leg_e], dim=2)              # (B,H,5,D)

    def _mp(self, x: torch.Tensor) -> torch.Tensor:
        """One round of star-topology message passing."""
        B_, N, D = x.shape
        assert N == 5 and D == self.D

        x0 = x[:, 0:1, :]                                                 # (B_,1,D)
        xl = x[:, 1:5, :]                                                 # (B_,4,D)
        x0e = x0.expand(-1, 4, -1)                                        # (B_,4,D)

        inp_0i = torch.cat([x0e, xl], dim=-1)                             # base -> leg
        inp_i0 = torch.cat([xl, x0e], dim=-1)                             # leg -> base
        inp = torch.cat([inp_0i, inp_i0], dim=1)                          # (B_,8,2D)

        m = self.edge_mlp(inp.reshape(B_ * 8, 2 * D)).view(B_, 8, self.msg_dim)
        m_0i, m_i0 = m[:, :4, :], m[:, 4:, :]

        agg = x.new_zeros((B_, 5, self.msg_dim))
        agg[:, 0, :] = m_i0.sum(dim=1)
        agg[:, 1:5, :] = m_0i

        return self.node_mlp(torch.cat([x, agg], dim=-1))

    def forward(self, base_seq: torch.Tensor, joint_seq: torch.Tensor) -> torch.Tensor:
        B, H, _ = base_seq.shape
        assert H == self.H

        x_seq = self._encode_nodes(base_seq, joint_seq)                    # (B,H,5,D)
        x = x_seq.reshape(B * H, 5, self.D)

        for _ in range(self.mp_steps):
            x = self._mp(x)

        s = x.reshape(x.shape[0], self.N * self.D)                        # concat pool
        s_seq = s.reshape(B, H, self.state_dim)                           # (B,H,state_dim)

        _, hN = self.gru(s_seq)
        h_last = hN[-1]                                                   # (B, gh)

        return F.softplus(self.head(h_last).squeeze(-1))
