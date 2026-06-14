"""QPA-tuned variant: the QPSAN/QPA scoring attention (quantum_variants/qpa_scoring.py)
plus two fixes that address its underfitting vs. the classical baseline.

Fix 1 — learnable temperature on the attention logits.
    The QPA score matrix A_ij = sum_d mu_d is bounded and low-contrast, so the
    softmax is nearly uniform at init and the attention barely selects tokens.
    A learnable scalar temperature is applied to A *before* softmax to restore
    contrast: attn = (A * tau).softmax(-1). (Softmax is shift-invariant, so only
    this multiplicative scale matters.) Motivation: scaled dot-product attention
    scales the logits to keep softmax out of low-gradient regions (Vaswani et
    al., NeurIPS 2017); learnable attention temperature has direct precedent in
    Query-Key Normalization (Henry et al., Findings of EMNLP 2020) and the
    learnable capped temperature of scaled cosine attention (Liu et al., Swin
    Transformer V2, CVPR 2022).

Fix 2 — full-dimension score aggregation.
    The base variant truncates aggregation to agg_dim = min(16, head_dim) = 16,
    discarding half the q/k channels. Here we default to ALL head dimensions
    (agg_dim = head_dim). Full-dim aggregation is the standard scaled
    dot-product formulation (Vaswani et al. 2017); QPSAN's D=16
    (arXiv:2605.25365, §III-B) was an efficiency trade-off we re-tune for
    MedViT's larger head dimension.

The 2-qubit QPA circuit itself is REUSED unchanged from qpa_scoring.qpa_mu, so
this variant differs from "qpa" only in Fixes 1 and 2.
"""

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
from quantum_variants.qpa_scoring import qpa_mu  # REUSE the validated circuit


class Q_E_MHSA_Tuned(nn.Module):
    """QPA-scoring attention with a learnable temperature (Fix 1) and full-dim
    aggregation (Fix 2)."""

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
        agg_dim=None,
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

        # Fix 2: aggregate over ALL head dims by default (vs. min(16, head_dim)).
        self.agg_dim = head_dim if agg_dim is None else min(agg_dim, head_dim)

        # QPA: 5 trainable params per attention layer, init as in QPSAN's QAOACircuit.
        self.theta_s = nn.Parameter(torch.tensor(0.5))      # enc_scale
        self.gamma_d = nn.Parameter(torch.randn(1) * 0.1)   # gamma_diff
        self.gamma_s = nn.Parameter(torch.randn(1) * 0.1)   # gamma_sum
        self.alpha = nn.Parameter(torch.randn(1) * 0.1)     # entanglement
        self.beta = nn.Parameter(torch.randn(1) * 0.1)      # mixer

        # Fix 1: learnable temperature applied to A before softmax (restores
        # attention contrast lost when the bounded QPA score replaces q@k).
        self.tau = nn.Parameter(torch.tensor(1.0))

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
        # Fix 1: learnable temperature tau scales the QPA logits before softmax.
        attn = self._qpa_scores(q, k) * self.tau
        # -------------------------------------------------------------------

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class QLTB_Tuned(nn.Module):
    """LTB with its E_MHSA swapped for the tuned QPA-scoring attention."""

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
        agg_dim=None,
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
        self.e_mhsa = Q_E_MHSA_Tuned(
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


class QMedViT_QPA_Tuned(QMedViT):
    def _should_quantize_block(self, block, stage_id, block_idx) -> bool:
        if not (isinstance(block, LTB) and stage_id in self.quantum_stages):
            return False
        if self.quantum_block_indices is None:
            return True
        within = block_idx - sum(self.depths[:stage_id])
        return within in self.quantum_block_indices

    def _build_quantum_block(self, block, stage_id, block_idx, **ctx) -> nn.Module:
        return QLTB_Tuned(
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
            agg_dim=getattr(self, "qpa_agg_dim", None),  # None -> full head_dim (Fix 2)
        )


if __name__ == "__main__":
    import torch
    from quantum_variants import VARIANTS

    # Fix 1/2 sanity at the attention-module level.
    attn = Q_E_MHSA_Tuned(dim=96, head_dim=32, sr_ratio=2)
    assert isinstance(attn.tau, nn.Parameter) and attn.tau.requires_grad, "tau not trainable"
    assert abs(attn.tau.item() - 1.0) < 1e-9, "tau should init to 1.0"
    assert attn.agg_dim == attn.head_dim == 32, f"agg_dim should be head_dim, got {attn.agg_dim}"
    print(f"[validation] PASS: tau trainable (init {attn.tau.item():.1f}), "
          f"agg_dim={attn.agg_dim} == head_dim (Fix 2)")

    # End-to-end build + forward via the registered variant.
    # Minimal valid MedViT config: stage-2 depth MUST be a multiple of 5; the
    # single LTB in stage 3 is the one quantized by default (quantum_stages=(3,)).
    model = VARIANTS["qpa_tuned"](
        stem_chs=[64, 32, 64], depths=[1, 1, 5, 1], path_dropout=0.0,
        num_classes=3, quantum_stages=(3,),
    )
    model.eval()
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    assert y.shape == (2, 3), f"unexpected logits shape {tuple(y.shape)}"
    print(f"[validation] PASS: forward ok, logits shape = {tuple(y.shape)}")

    # Gradient must reach tau. (Select by attribute, not isinstance: running via
    # `python -m` makes __main__.Q_E_MHSA_Tuned distinct from the package class.)
    model.train()
    tau_param = next(
        m.tau for m in model.modules()
        if isinstance(getattr(m, "tau", None), nn.Parameter)
    )
    model(x).sum().backward()
    assert tau_param.grad is not None and torch.isfinite(tau_param.grad).all(), "no grad on tau"
    print(f"[validation] PASS: gradient reaches tau (grad={tau_param.grad.item():.3e})")
