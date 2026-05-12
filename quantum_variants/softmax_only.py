"""Quantum-softmax variant: replaces `attn.softmax(dim=-1)` inside E_MHSA with
the Born-rule probability of an amplitude-encoded score row passed through a
trainable orthogonal RBS-pyramid (Cherrat et al., Quantum 8, 1265, §5).

Classical Q, K, V projections and the (attn @ V) weighted sum are untouched.
"""

from functools import partial

import pennylane as qml
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from MedViT import (
    NORM_EPS,
    LTB,
    MHCA,
    PatchEmbed,
    LocalityFeedForward,
    _make_divisible,
)
from QMedViT import QMedViT
from utils import merge_pre_bn


def _build_qsoftmax(n_qubits: int, q_depth: int, qdevice: str, qbackend: str | None):
    dev_kwargs = {"wires": n_qubits}
    if qbackend is not None:
        dev_kwargs["backend"] = qbackend
    dev = qml.device(qdevice, **dev_kwargs)

    # Brick-wall nearest-neighbor RBS pyramid (Cherrat et al., §5): per layer,
    # an even sublayer of pairs (0,1),(2,3),... followed by an odd sublayer
    # (1,2),(3,4),... — fewer gates than all-to-all at equal expressivity.
    n_even = n_qubits // 2
    n_odd = (n_qubits - 1) // 2
    n_pairs = n_even + n_odd
    weight_shape = (q_depth, n_pairs)

    @qml.qnode(dev, interface="torch", diff_method="backprop", cache=True)
    def circuit(amps, weights):
        qml.AmplitudeEmbedding(amps, wires=range(n_qubits), normalize=False)
        for d in range(q_depth):
            a = 0
            for i in range(0, n_qubits - 1, 2):
                qml.SingleExcitation(weights[d, a], wires=[i, i + 1])
                a += 1
            for i in range(1, n_qubits - 1, 2):
                qml.SingleExcitation(weights[d, a], wires=[i, i + 1])
                a += 1
        # Return the full state so the caller can extract the unitary U(theta)
        # by feeding the D basis vectors once per forward, then apply U
        # classically to all (B,H,N) samples (mathematically identical to
        # running the circuit per sample, but avoids per-sample Pennylane
        # overhead — the unitary depends only on `weights`, not on `amps`).
        return qml.state()

    return circuit, weight_shape


def _build_qsoftmax_qpu(n_qubits: int, q_depth: int, qdevice: str, qbackend: str | None, shots: int):
    """QPU-style QNode: per-sample execution, finite shots, returns sampled
    probabilities (not the statevector). Mirrors what running on real quantum
    hardware looks like — no statevector access, stochastic estimates of
    |U(theta) * amps|^2 from `shots` measurements per call. Intended for
    inference only (gradients via parameter-shift would be expensive and noisy).
    """
    dev_kwargs = {"wires": n_qubits, "shots": shots}
    if qbackend is not None:
        dev_kwargs["backend"] = qbackend
    dev = qml.device(qdevice, **dev_kwargs)

    @qml.qnode(dev, interface="torch")
    def circuit(amps, weights):
        # normalize=True: amps come in as float32 (~1e-7 precision); the device
        # renormalizes in complex128, eliminating the drift that otherwise
        # trips PennyLane's strict sum-to-1 check during finite-shot sampling.
        qml.AmplitudeEmbedding(amps, wires=range(n_qubits), normalize=True)
        for d in range(q_depth):
            a = 0
            for i in range(0, n_qubits - 1, 2):
                qml.SingleExcitation(weights[d, a], wires=[i, i + 1])
                a += 1
            for i in range(1, n_qubits - 1, 2):
                qml.SingleExcitation(weights[d, a], wires=[i, i + 1])
                a += 1
        return qml.probs(wires=range(n_qubits))

    return circuit


class Q_E_MHSA(nn.Module):
    def __init__(
        self,
        dim,
        out_dim=None,
        head_dim=32,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0,
        proj_drop=0.0,
        sr_ratio=1,
        n_qubits=4,
        q_depth=3,
        qdevice="default.qubit",
        qbackend=None,
        qpu_mode=False,
        qpu_shots=5000,
    ):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.num_heads = self.dim // head_dim
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.v = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.proj = nn.Linear(self.dim, self.out_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        self.N_ratio = sr_ratio ** 2
        if sr_ratio > 1:
            self.sr = nn.AvgPool1d(kernel_size=self.N_ratio, stride=self.N_ratio)
            self.norm = nn.BatchNorm1d(dim, eps=NORM_EPS)
        self.is_bn_merged = False

        self.n_qubits = n_qubits
        self.softmax_dim = 1 << n_qubits  # 2 ** n_qubits
        self.qpu_mode = qpu_mode
        self.qpu_shots = qpu_shots
        # Simulator path: analytic, broadcast-friendly, returns state for
        # one-shot U(theta) extraction. Always built — used for training and
        # whenever qpu_mode is False.
        self.qsoftmax, weight_shape = _build_qsoftmax(n_qubits, q_depth, qdevice, qbackend)
        self.softmax_weights = nn.Parameter(0.5 * torch.randn(*weight_shape))
        # QPU path: per-sample, finite shots, returns probs. Mirrors real
        # hardware execution (no statevector access). Built lazily only when
        # qpu_mode is requested.
        if qpu_mode:
            self.qsoftmax_qpu = _build_qsoftmax_qpu(
                n_qubits, q_depth, qdevice, qbackend, qpu_shots
            )

    def merge_bn(self, pre_bn):
        merge_pre_bn(self.q, pre_bn)
        if self.sr_ratio > 1:
            merge_pre_bn(self.k, pre_bn, self.norm)
            merge_pre_bn(self.v, pre_bn, self.norm)
        else:
            merge_pre_bn(self.k, pre_bn)
            merge_pre_bn(self.v, pre_bn)
        self.is_bn_merged = True

    def _quantum_softmax(self, attn):
        # attn: (B, H, N, M) classical scores. Replace softmax(dim=-1) with
        # alpha_ij = |<j| U(theta) | s_i / ||s_i||_2 >|^2  (Cherrat et al.).
        B, H, N, M = attn.shape
        D = self.softmax_dim
        top_idx = None
        if M < D:
            x = F.pad(attn, (0, D - M))
        elif M > D:
            # Top-D selection: keep the D highest scores per query so the
            # quantum block attends to the most relevant keys, not the first D.
            x, top_idx = attn.topk(D, dim=-1)
        else:
            x = attn

        norm = x.norm(dim=-1, keepdim=True)
        # For rows with effectively-zero score (norm < eps), fall back to a
        # uniform amplitude (1/sqrt(D)) — semantically the same as softmax of
        # all-equal logits, and keeps |amps|^2 summing to 1 (required by
        # AmplitudeEmbedding on the QPU path).
        zero_row = norm < 1e-9
        safe_norm = torch.where(zero_row, torch.ones_like(norm), norm)
        amps = x / safe_norm
        if zero_row.any():
            uniform = torch.full_like(amps, 1.0 / (D ** 0.5))
            amps = torch.where(zero_row.expand_as(amps), uniform, amps)

        if self.qpu_mode:
            # QPU-style execution: no statevector shortcut, results are sampled
            # from `qpu_shots` measurements per input — what real hardware does.
            # We feed all (B*H*N) samples as a single broadcasted call. On
            # qiskit.remote this becomes ONE batched job submission (Sampler
            # primitive evaluates the same parametric circuit at T binding
            # values) instead of T separate jobs — orders-of-magnitude faster
            # on real backends. Intended for inference only.
            # Cast to float64 and renormalize in higher precision before
            # handing off to PennyLane. The device runs in complex128, and
            # its finite-shot sampler checks |state|^2 sums to 1 with a
            # tolerance of ~100*eps(float64) ≈ 2e-14 — float32 amplitudes
            # carry ~1e-7 drift, which fails that check.
            amps_flat = amps.reshape(-1, D).to(torch.float64)
            amps_flat = amps_flat / amps_flat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            probs_flat = self.qsoftmax_qpu(amps_flat, self.softmax_weights)
            probs = probs_flat.to(amps.dtype).reshape(B, H, N, D)
        else:
            # Simulator path: extract U(theta) once by feeding the D basis
            # vectors through the QNode (one broadcasted call), then apply U
            # classically to all samples. Mathematically identical to the
            # per-sample call but ~1000x faster on default.qubit.
            basis = torch.eye(D, dtype=amps.dtype, device=amps.device)
            UT = self.qsoftmax(basis, self.softmax_weights)  # (D, D), complex
            UT = UT.real.to(amps.dtype)                       # RBS pyramid is real
            transformed = amps @ UT          # (B, H, N, D), equivalent to U @ amps
            probs = transformed ** 2          # Born rule

        if M < D:
            probs = probs[..., :M]
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        elif M > D:
            # Scatter the D probabilities back to the original key positions.
            full = torch.zeros(B, H, N, M, device=attn.device, dtype=probs.dtype)
            full.scatter_(-1, top_idx, probs)
            probs = full
        return probs

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x)
        q = q.reshape(B, N, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2)
            x_ = self.sr(x_)
            if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
                x_ = self.norm(x_)
            x_ = x_.transpose(1, 2)
            k = self.k(x_)
            k = k.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 3, 1)
            v = self.v(x_)
            v = v.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)
        else:
            k = self.k(x)
            k = k.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 3, 1)
            v = self.v(x)
            v = v.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)

        attn = (q @ k) * self.scale

        # ---- THE ONLY CHANGE vs. classical E_MHSA --------------------------
        attn = self._quantum_softmax(attn)
        # --------------------------------------------------------------------

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class QLTB(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        path_dropout,
        stride=1,
        sr_ratio=1,
        mlp_ratio=2,
        head_dim=32,
        mix_block_ratio=0.75,
        attn_drop=0,
        drop=0,
        n_qubits=4,
        q_depth=3,
        qdevice="default.qubit",
        qbackend=None,
        qpu_mode=False,
        qpu_shots=5000,
    ):
        super().__init__()
        from timm.models.layers import DropPath

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mix_block_ratio = mix_block_ratio
        norm_func = partial(nn.BatchNorm2d, eps=NORM_EPS)

        self.mhsa_out_channels = _make_divisible(int(out_channels * mix_block_ratio), 32)
        self.mhca_out_channels = out_channels - self.mhsa_out_channels

        self.patch_embed = PatchEmbed(in_channels, self.mhsa_out_channels, stride)
        self.norm1 = norm_func(self.mhsa_out_channels)
        self.e_mhsa = Q_E_MHSA(
            self.mhsa_out_channels,
            head_dim=head_dim,
            sr_ratio=sr_ratio,
            attn_drop=attn_drop,
            proj_drop=drop,
            n_qubits=n_qubits,
            q_depth=q_depth,
            qdevice=qdevice,
            qbackend=qbackend,
            qpu_mode=qpu_mode,
            qpu_shots=qpu_shots,
        )
        self.mhsa_path_dropout = DropPath(path_dropout * mix_block_ratio)

        self.projection = PatchEmbed(self.mhsa_out_channels, self.mhca_out_channels, stride=1)
        self.mhca = MHCA(self.mhca_out_channels, head_dim=head_dim)
        self.mhca_path_dropout = DropPath(path_dropout * (1 - mix_block_ratio))

        self.norm2 = norm_func(out_channels)
        self.conv = LocalityFeedForward(out_channels, out_channels, 1, mlp_ratio, reduction=out_channels)
        self.is_bn_merged = False

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        out = self.norm1(x) if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged else x
        out = rearrange(out, "b c h w -> b (h w) c")
        out = self.mhsa_path_dropout(self.e_mhsa(out))
        x = x + rearrange(out, "b (h w) c -> b c h w", h=H)

        out = self.projection(x)
        out = out + self.mhca_path_dropout(self.mhca(out))
        x = torch.cat([x, out], dim=1)

        out = self.norm2(x) if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged else x
        x = x + self.conv(out)
        return x


class QMedViT_Softmax_Only(QMedViT):
    def _should_quantize_block(self, block, stage_id, block_idx) -> bool:
        if not (isinstance(block, LTB) and stage_id in self.quantum_stages):
            return False
        if self.quantum_block_indices is None:
            return True
        within = block_idx - sum(self.depths[:stage_id])
        return within in self.quantum_block_indices

    def _build_quantum_block(self, block, stage_id, block_idx, **ctx) -> nn.Module:
        return QLTB(
            in_channels=block.in_channels,
            out_channels=block.out_channels,
            path_dropout=ctx["dpr"],
            stride=1,
            sr_ratio=ctx["sr_ratio"],
            mlp_ratio=2,
            head_dim=self.head_dim,
            mix_block_ratio=self.mix_block_ratio,
            attn_drop=self.attn_drop,
            drop=self.drop,
            n_qubits=self.n_qubits,
            q_depth=self.q_depth,
            qdevice=self.qdevice,
            qbackend=self.qbackend,
            qpu_mode=self.qpu_mode,
            qpu_shots=self.qpu_shots,
        )
