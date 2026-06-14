"""QPA-scoring variant: replaces the dot-product attention SCORING FUNCTION
`(q @ k) * scale` inside E_MHSA with the Quantum Parametric Attention (QPA)
circuit from QPSAN (arXiv:2605.25365; ref impl github.com/Comets9224/QPSAN,
models/quantum_attention.py).

Each (query, key) scalar pair is scored by a 2-qubit PQC
    encode  RY(phi0) (x) RY(phi1)
    entangle CNOT(0->1) ; RY(alpha*(q+k)) on q1 ; CNOT(1->0)
    mixer   RX(2*beta) on both qubits
    measure mu = P(|00>) + P(|11>)  in [0, 1]
and per-dimension scores are summed over the first D dims to form A_ij
(paper eq.11). Softmax, Q/K/V projections and the (attn @ V) sum are untouched;
`self.scale` is unused since mu is already bounded.

The 2-qubit circuit is simulated analytically (closed-form statevector) with
pure torch ops so it stays fully differentiable and runs batched on GPU over
all (B, heads, N, N, D) pairs at once — a per-pair simulator call would be far
too slow given MedViT's token counts.
"""

import math
from functools import partial

import torch
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


def qpa_mu(q, k, theta_s, gamma_d, gamma_s, alpha, beta):
    """Per-dimension QPA score mu(q, k) = P(|00>) + P(|11>) in [0, 1].

    `q` and `k` are broadcastable tensors of any shape; the 5 params are
    scalar/[1] nn.Parameters shared across all pairs (QPSAN: per attention
    layer). Basis index: |q0 q1> -> 2*q0 + q1, so 0=|00>, 1=|01>, 2=|10>, 3=|11>.
    """
    lam1 = theta_s + gamma_d + gamma_s          # self-coefficient (paper Lemma 1)
    lam2 = gamma_s - gamma_d                     # cross-coefficient
    phi0 = math.pi / 4 + lam1 * q + lam2 * k     # RY angle, qubit 0 (eq.5)
    phi1 = math.pi / 4 + lam2 * q + lam1 * k     # RY angle, qubit 1 (eq.6)

    a0, b0 = torch.cos(phi0 / 2), torch.sin(phi0 / 2)
    a1, b1 = torch.cos(phi1 / 2), torch.sin(phi1 / 2)
    # product state after encoding (real amplitudes)
    c0, c1, c2, c3 = a0 * a1, a0 * b1, b0 * a1, b0 * b1

    # CNOT(0->1): flip qubit1 when qubit0=1 -> swap |10>,|11>
    c2, c3 = c3, c2

    # RY(alpha*(q+k)) on qubit1
    eta = alpha * (q + k)
    ce, se = torch.cos(eta / 2), torch.sin(eta / 2)
    c0, c1 = ce * c0 - se * c1, se * c0 + ce * c1   # block q0=0 (idx 0,1)
    c2, c3 = ce * c2 - se * c3, se * c2 + ce * c3   # block q0=1 (idx 2,3)

    # CNOT(1->0): flip qubit0 when qubit1=1 -> swap |01>,|11>
    c1, c3 = c3, c1

    # RX(2*beta) on qubit0 then qubit1 (theta/2 = beta); amplitudes go complex.
    cb, sb = torch.cos(beta), torch.sin(beta)
    j = 1j
    # RX on qubit0: pairs (0,2),(1,3)
    c0, c2 = cb * c0 - j * sb * c2, -j * sb * c0 + cb * c2
    c1, c3 = cb * c1 - j * sb * c3, -j * sb * c1 + cb * c3
    # RX on qubit1: pairs (0,1),(2,3)
    c0, c1 = cb * c0 - j * sb * c1, -j * sb * c0 + cb * c1
    c2, c3 = cb * c2 - j * sb * c3, -j * sb * c2 + cb * c3

    return c0.abs() ** 2 + c3.abs() ** 2


class Q_E_MHSA_Scoring(nn.Module):
    """E_MHSA with the QK^T scoring function replaced by the QPA circuit."""

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
        agg_dim=16,
    ):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.num_heads = self.dim // head_dim
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5  # unused (mu is bounded), kept for parity

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

        # QPA: 5 trainable params per attention layer, init as in QPSAN's QAOACircuit.
        self.agg_dim = min(agg_dim, head_dim)  # D=16 in paper, head_dim=32 here
        self.theta_s = nn.Parameter(torch.tensor(0.5))      # enc_scale
        self.gamma_d = nn.Parameter(torch.randn(1) * 0.1)   # gamma_diff
        self.gamma_s = nn.Parameter(torch.randn(1) * 0.1)   # gamma_sum
        self.alpha = nn.Parameter(torch.randn(1) * 0.1)     # entanglement
        self.beta = nn.Parameter(torch.randn(1) * 0.1)      # mixer

    def merge_bn(self, pre_bn):
        merge_pre_bn(self.q, pre_bn)
        if self.sr_ratio > 1:
            merge_pre_bn(self.k, pre_bn, self.norm)
            merge_pre_bn(self.v, pre_bn, self.norm)
        else:
            merge_pre_bn(self.k, pre_bn)
            merge_pre_bn(self.v, pre_bn)
        self.is_bn_merged = True

    def _qpa_scores(self, q, k):
        # q: (B, H, Nq, head_dim);  k: (B, H, head_dim, Nk) -> align to (B,H,Nk,hd)
        k = k.transpose(-2, -1)
        D = self.agg_dim
        # NOTE: mu is (B, H, Nq, Nk, D). N is kept small by sr_ratio>1 stages;
        # apply QPA only where the feature map (hence N) is reduced.
        qd = q[..., :D].unsqueeze(3)   # (B, H, Nq, 1,  D)
        kd = k[..., :D].unsqueeze(2)   # (B, H, 1,  Nk, D)
        mu = qpa_mu(qd, kd, self.theta_s, self.gamma_d, self.gamma_s, self.alpha, self.beta)
        return mu.sum(dim=-1)          # A_ij = sum_d mu_d  (paper eq.11)

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

        # ---- THE ONLY CHANGE vs. classical E_MHSA -------------------------
        # classical: attn = (q @ k) * self.scale
        attn = self._qpa_scores(q, k)
        # -------------------------------------------------------------------

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class QLTB_Scoring(nn.Module):
    """LTB with its E_MHSA swapped for the QPA-scoring attention."""

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
        agg_dim=16,
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
        self.e_mhsa = Q_E_MHSA_Scoring(
            self.mhsa_out_channels,
            head_dim=head_dim,
            sr_ratio=sr_ratio,
            attn_drop=attn_drop,
            proj_drop=drop,
            agg_dim=agg_dim,
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


class QMedViT_QPA_Scoring(QMedViT):
    def _should_quantize_block(self, block, stage_id, block_idx) -> bool:
        if not (isinstance(block, LTB) and stage_id in self.quantum_stages):
            return False
        if self.quantum_block_indices is None:
            return True
        within = block_idx - sum(self.depths[:stage_id])
        return within in self.quantum_block_indices

    def _build_quantum_block(self, block, stage_id, block_idx, **ctx) -> nn.Module:
        return QLTB_Scoring(
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
            agg_dim=getattr(self, "qpa_agg_dim", 16),
        )


if __name__ == "__main__":
    # Validation: the analytic qpa_mu must reproduce an independent simulator
    # (PennyLane, already a repo dependency) running the same QPSAN circuit.
    import pennylane as qml

    torch.manual_seed(0)
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, interface="torch")
    def ref(q, k, theta_s, gamma_d, gamma_s, alpha, beta):
        lam1 = theta_s + gamma_d + gamma_s
        lam2 = gamma_s - gamma_d
        phi0 = math.pi / 4 + lam1 * q + lam2 * k
        phi1 = math.pi / 4 + lam2 * q + lam1 * k
        qml.RY(phi0, wires=0)
        qml.RY(phi1, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(alpha * (q + k), wires=1)
        qml.CNOT(wires=[1, 0])
        qml.RX(2 * beta, wires=0)
        qml.RX(2 * beta, wires=1)
        return qml.probs(wires=[0, 1])

    n = 50
    qs = torch.randn(n)
    ks = torch.randn(n)
    # random circuit params shared across pairs
    ts = torch.tensor(0.5)
    gd, gs, al, be = (torch.randn(()) * 0.5 for _ in range(4))

    mine = qpa_mu(qs, ks, ts, gd, gs, al, be)
    max_err = 0.0
    for i in range(n):
        p = ref(qs[i], ks[i], ts, gd, gs, al, be)
        ref_mu = (p[0] + p[3]).item()
        max_err = max(max_err, abs(ref_mu - mine[i].item()))
    print(f"[validation] max |qpa_mu - pennylane_ref| over {n} pairs = {max_err:.2e}")
    assert max_err < 1e-5, "qpa_mu does not match the reference circuit!"
    print("[validation] PASS: analytic circuit matches reference within 1e-5")

    # boundedness sanity + gradient flow
    assert (mine >= -1e-6).all() and (mine <= 1 + 1e-6).all(), "mu out of [0,1]"
    g = qpa_mu(qs.requires_grad_(), ks, ts, gd.requires_grad_(), gs, al, be).sum()
    g.backward()
    assert gd.grad is not None and qs.grad is not None
    print("[validation] PASS: mu in [0,1] and gradients flow to inputs and params")

    # end-to-end smoke test of the full model variant
    from quantum_variants import VARIANTS

    # minimal valid MedViT config (stage 2 depth must be a multiple of 5); the
    # single LTB in stage 3 is the one quantized by default (quantum_stages=(3,)).
    model = VARIANTS["qpa"](
        stem_chs=[64, 32, 64], depths=[1, 1, 5, 1], path_dropout=0.0,
        num_classes=3, quantum_stages=(3,),
    )
    model.eval()
    y = model(torch.randn(2, 3, 64, 64))
    print(f"[validation] PASS: forward ok, logits shape = {tuple(y.shape)}")
