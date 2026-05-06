"""Policy / value network for the DRL agent.

Architecture (faithful to the position paper's framework, with the
simplifications documented in CHANGES.md):

    node_features  --(MLP)-->  node_embed (N, d)
                                 |
                                 |   self-attention (1 round, multi-head)
                                 v
                              ctx_node_embed (N, d)

    service_features --(MLP)--> svc_embed (M, d)

    For each microservice j (autoregressive over a fixed order):
        query = MLP_q( svc_embed[j] ⊕ load_summary )    -- includes the
                                                          running cpu/mem
                                                          tally so the policy
                                                          is capacity-aware
        scores_i = (query · ctx_node_embed_i) / sqrt(d)
        scores -= 1e9 * (1 - feasibility_mask)
        prob = softmax(scores)                          (over candidate nodes)
        a_j ~ prob

    value = MLP_v( mean(ctx_node_embed) ⊕ mean(svc_embed) )

The autoregressive sampling reduces the effective action space from
|N|^|M| to |N|·|M|, which is the key tractability move described in the
position paper. Each service's sampling step also receives an updated
load-summary feature reflecting the placements made so far in this round, so
the policy learns to spread load.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeEncoder(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class ServiceEncoder(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class NodeSelfAttention(nn.Module):
    """One multi-head self-attention block over nodes. With a residual + layer
    norm this acts as a 1-layer transformer encoder that contextualises each
    node's representation with the other nodes' state. This is the message-
    passing component."""

    def __init__(self, embed_dim: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm(x + h)
        x = self.norm2(x + self.ff(x))
        return x


class PlacementPolicy(nn.Module):
    """Joint policy/value network for the placement MDP."""

    def __init__(
        self,
        node_in_dim: int,
        svc_in_dim: int,
        embed_dim: int = 64,
        n_heads: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.node_encoder = NodeEncoder(node_in_dim + 4, embed_dim)
        self.svc_encoder = ServiceEncoder(svc_in_dim, embed_dim)
        self.node_attn = NodeSelfAttention(embed_dim, n_heads)
        # Query MLP combines the service embedding and a running load summary
        # (cpu_used, mem_used, num_placed) so the policy can adapt to its own
        # prior placements within the same round.
        self.query_mlp = nn.Sequential(
            nn.Linear(embed_dim + 3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def encode(self, node_features: torch.Tensor, svc_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Augment node features with one-hot of the layer index already in
        # column 0 of node_features. Encoder MLP expects node_in_dim+4.
        layer_idx = node_features[..., 0].long().clamp(min=0, max=2)
        layer_oh = F.one_hot(layer_idx, num_classes=4).float()
        nfeat = torch.cat([node_features, layer_oh], dim=-1)
        node_embed = self.node_encoder(nfeat)
        ctx = self.node_attn(node_embed)
        svc_embed = self.svc_encoder(svc_features)
        return ctx, svc_embed

    def value(self, ctx_node: torch.Tensor, svc_embed: torch.Tensor) -> torch.Tensor:
        node_summary = ctx_node.mean(dim=-2)
        svc_summary = svc_embed.mean(dim=-2)
        return self.value_head(torch.cat([node_summary, svc_summary], dim=-1)).squeeze(-1)

    def score(
        self,
        svc_embed_j: torch.Tensor,        # (B, d)
        ctx_node: torch.Tensor,           # (B, N, d)
        load_summary: torch.Tensor,       # (B, 3)
        feas_mask: torch.Tensor,          # (B, N) bool / float
    ) -> torch.Tensor:
        """Compute placement logits over nodes for one service."""
        q = self.query_mlp(torch.cat([svc_embed_j, load_summary], dim=-1))
        # Scaled dot-product attention scores
        scores = (q.unsqueeze(1) * ctx_node).sum(dim=-1) / (self.embed_dim ** 0.5)
        # Mask out infeasible nodes
        scores = scores.masked_fill(~feas_mask.bool(), float("-inf"))
        return scores
