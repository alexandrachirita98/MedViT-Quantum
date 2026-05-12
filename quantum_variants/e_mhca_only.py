"""Quantum-MHCA variant: replaces MHCA's core attention score with a quantum kernel."""

from functools import partial

import torch
from torch import nn

from MedViT import NORM_EPS, ECB, LTB
from QMedViT import QMedViT, _build_qkernel


class Q_MHCA(nn.Module):
    """MHCA whose post-conv per-token feature is gated by a quantum kernel score.

    Mirrors MedViT.MHCA layer-for-layer (group_conv3x3 -> norm -> act ->
    projection). Between act and projection we treat each (head, spatial-token)
    feature of size head_dim as the input to a quantum kernel: two classical
    adapters map it to n_qubits angles for the "query" and "key" registers,
    the kernel returns a scalar per token, and we use sigmoid(score) as a
    multiplicative gate over that head's channels.
    """

    def __init__(
        self,
        out_channels,
        head_dim,
        n_qubits=4,
        q_depth=2,
        qdevice="default.qubit",
        qbackend=None,
    ):
        super().__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        self.out_channels = out_channels
        self.head_dim = head_dim
        self.num_heads = out_channels // head_dim
        self.n_qubits = n_qubits

        self.group_conv3x3 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1,
            groups=self.num_heads, bias=False,
        )
        self.norm = norm_layer(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.projection = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

        self.to_angles_q = nn.Linear(head_dim, n_qubits)
        self.to_angles_k = nn.Linear(head_dim, n_qubits)
        self.qkernel, weight_shape = _build_qkernel(n_qubits, q_depth, qdevice, qbackend)
        self.qweights = nn.Parameter(0.1 * torch.randn(*weight_shape))

    def _quantum_gate(self, feats):
        B, Hh, N, _ = feats.shape
        q_ang = self.to_angles_q(feats)
        k_ang = self.to_angles_k(feats)
        flat_q = q_ang.reshape(-1, self.n_qubits)
        flat_k = k_ang.reshape(-1, self.n_qubits)
        out = torch.empty(flat_q.shape[0], device=feats.device, dtype=feats.dtype)
        for t in range(flat_q.shape[0]):
            out[t] = self.qkernel(flat_q[t], flat_k[t], self.qweights)
        return torch.sigmoid(out.reshape(B, Hh, N))

    def forward(self, x):
        out = self.group_conv3x3(x)
        out = self.norm(out)
        out = self.act(out)

        B, C, H, W = out.shape
        feats = out.reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        gate = self._quantum_gate(feats)
        gate = gate.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        feats = feats * gate
        out = feats.permute(0, 1, 3, 2).reshape(B, C, H, W)

        out = self.projection(out)
        return out


class QMedViT_E_MHCA_Only(QMedViT):
    """MHCA is patched in-place inside each ECB/LTB rather than replacing the
    whole block. Both block types own a `.mhca` submodule of the same MHCA
    class, so a single hook handles both with no duplication and the
    surrounding LTB.e_mhsa stays classical."""

    def _should_quantize_block(self, block, stage_id, block_idx) -> bool:
        return isinstance(block, (ECB, LTB)) and stage_id in self.quantum_stages

    def _build_quantum_block(self, block, stage_id, block_idx, **ctx) -> nn.Module:
        old = block.mhca
        block.mhca = Q_MHCA(
            out_channels=old.group_conv3x3.out_channels,
            head_dim=self.head_dim,
            n_qubits=self.n_qubits,
            q_depth=self.q_depth,
            qdevice=self.qdevice,
            qbackend=self.qbackend,
        )
        return block
