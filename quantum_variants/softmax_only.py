"""Quantum-softmax variant: replaces `attn.softmax(dim=-1)` inside E_MHSA with
the Born-rule probability of an amplitude-encoded score row passed through a
trainable orthogonal layer of RBS gates (`qml.SingleExcitation`).

Classical Q, K, V projections and the (attn @ V) weighted sum are untouched.

Circuit layout (`layout=`)
--------------------------
The orthogonal layer is a configurable arrangement of RBS gates. The RBS gate
itself is Cherrat et al. (Quantum 8, 1265, §2, Eq. 1) / Landman et al.
(Quantum 6, 881); the four layouts are:

  * "brick_wall" (DEFAULT) — alternating even/odd nearest-neighbor sublayers:
    per block, pairs (0,1),(2,3),... then (1,2),(3,4),...  This is the original
    layout; its gate order and parameter indexing are preserved byte-for-byte
    so existing checkpoints/results stay valid. `N-1` gates per block. This is
    a hardware-efficient layered ansatz (our design choice), NOT a named circuit
    from the paper.
  * "pyramid" — the Kerenidis/Landman triangular pyramid (Cherrat et al.
    Fig. 7): a Givens-rotation cascade of `N(N-1)/2` nearest-neighbor RBS gates
    per block, realizing the full orthogonal group SO(N).
  * "butterfly" — the Cooley-Tukey butterfly (Fig. 8): RBS gates on qubit pairs
    at distances 2^k, `(N/2)*log2(N)` gates per block, all-to-all connectivity.
    Requires `n_qubits` to be a power of 2 (raises ValueError otherwise).
  * "x" — the X circuit (Fig. 9): two crossing nearest-neighbor diagonals,
    `2N-3` gates per block.

q_depth semantics: `q_depth` is the number of times the chosen layout BLOCK is
stacked, uniformly for all four layouts. The trainable angle tensor has shape
`(q_depth, n_pairs)` where `n_pairs` is the per-block gate count above. For
"brick_wall" this reproduces the original `(q_depth, N-1)` tensor exactly.

NOTE on encoding: vectors are loaded with dense `AmplitudeEmbedding` (n qubits
-> 2^n amplitudes), not the paper's unary encoding. Since RBS gates preserve
Hamming weight, the realized orthogonal matrix is block-diagonal across weight
sectors rather than a single full 2^n x 2^n orthogonal mixing.
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


LAYOUTS = ("brick_wall", "pyramid", "butterfly", "x")


def _layout_pairs(n_qubits: int, layout: str):
    """Ordered list of (control, target) wire pairs for ONE block of `layout`.

    `q_depth` stacks this block `q_depth` times. Gate counts (the correctness
    check, per Cherrat et al. §2.2 / Landman et al.):
      * brick_wall: N-1            (even sublayer then odd sublayer)
      * pyramid:    N(N-1)/2       (triangular Givens cascade, SO(N)-complete)
      * x:          2N-3           (two crossing nearest-neighbor diagonals)
      * butterfly:  (N/2)*log2(N)  (pairs at distance 2^k; N must be 2^m)
    """
    if layout == "brick_wall":
        even = [(i, i + 1) for i in range(0, n_qubits - 1, 2)]
        odd = [(i, i + 1) for i in range(1, n_qubits - 1, 2)]
        return even + odd
    if layout == "pyramid":
        # Triangular Givens cascade: any N-dim orthogonal matrix decomposes into
        # N(N-1)/2 adjacent (Givens) rotations in this triangular pattern, so the
        # block spans the full SO(N).
        pairs = []
        for i in range(n_qubits - 1):
            for j in range(n_qubits - 1 - i):
                pairs.append((j, j + 1))
        return pairs
    if layout == "x":
        # Descending diagonal (0,1)..(N-2,N-1) then ascending back, sharing the
        # apex gate -> 2(N-1)-1 = 2N-3 nearest-neighbor gates.
        pairs = [(i, i + 1) for i in range(n_qubits - 1)]
        pairs += [(i, i + 1) for i in range(n_qubits - 3, -1, -1)]
        return pairs
    if layout == "butterfly":
        if n_qubits < 2 or (n_qubits & (n_qubits - 1)) != 0:
            raise ValueError(
                f"butterfly layout requires n_qubits to be a power of 2, "
                f"got n_qubits={n_qubits}"
            )
        n_stages = n_qubits.bit_length() - 1  # log2(n_qubits)
        pairs = []
        for s in range(n_stages):
            dist = 1 << s
            for i in range(n_qubits):
                # pair i with i+dist when bit s of i is 0 -> N/2 pairs per stage
                if (i // dist) % 2 == 0 and (i + dist) < n_qubits:
                    pairs.append((i, i + dist))
        return pairs
    raise ValueError(
        f"unknown layout {layout!r}; expected one of {LAYOUTS}"
    )


def _apply_rbs_layout(weights, q_depth: int, pairs):
    """Apply `q_depth` stacked blocks of the RBS layout described by `pairs`.

    Shared by the simulator and QPU circuit builders so the two paths can never
    diverge. `weights` has shape `(q_depth, len(pairs))`; the RBS gate is
    `qml.SingleExcitation` (the PennyLane realization of the paper's RBS gate).
    """
    for d in range(q_depth):
        for a, (i, j) in enumerate(pairs):
            qml.SingleExcitation(weights[d, a], wires=[i, j])


def _build_qsoftmax(
    n_qubits: int,
    q_depth: int,
    qdevice: str,
    qbackend: str | None,
    layout: str = "brick_wall",
):
    dev_kwargs = {"wires": n_qubits}
    if qbackend is not None:
        dev_kwargs["backend"] = qbackend
    dev = qml.device(qdevice, **dev_kwargs)

    pairs = _layout_pairs(n_qubits, layout)
    weight_shape = (q_depth, len(pairs))

    @qml.qnode(dev, interface="torch", diff_method="backprop", cache=True)
    def circuit(amps, weights):
        qml.AmplitudeEmbedding(amps, wires=range(n_qubits), normalize=False)
        _apply_rbs_layout(weights, q_depth, pairs)
        # Return the full state so the caller can extract the unitary U(theta)
        # by feeding the D basis vectors once per forward, then apply U
        # classically to all (B,H,N) samples (mathematically identical to
        # running the circuit per sample, but avoids per-sample Pennylane
        # overhead — the unitary depends only on `weights`, not on `amps`).
        return qml.state()

    return circuit, weight_shape


def _build_qsoftmax_qpu(
    n_qubits: int,
    q_depth: int,
    qdevice: str,
    qbackend: str | None,
    shots: int,
    layout: str = "brick_wall",
):
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

    pairs = _layout_pairs(n_qubits, layout)

    @qml.qnode(dev, interface="torch", diff_method=None)
    def circuit(amps, weights):
        # normalize=True: amps come in as float32 (~1e-7 precision); the device
        # renormalizes in complex128, eliminating the drift that otherwise
        # trips PennyLane's strict sum-to-1 check during finite-shot sampling.
        qml.AmplitudeEmbedding(amps, wires=range(n_qubits), normalize=True)
        _apply_rbs_layout(weights, q_depth, pairs)
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
        layout="brick_wall",
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
        self.layout = layout
        # Simulator path: analytic, broadcast-friendly, returns state for
        # one-shot U(theta) extraction. Always built — used for training and
        # whenever qpu_mode is False.
        self.qsoftmax, weight_shape = _build_qsoftmax(
            n_qubits, q_depth, qdevice, qbackend, layout
        )
        self.softmax_weights = nn.Parameter(0.5 * torch.randn(*weight_shape))
        # QPU path: per-sample, finite shots, returns probs. Mirrors real
        # hardware execution (no statevector access). Built lazily only when
        # qpu_mode is requested.
        if qpu_mode:
            self.qsoftmax_qpu = _build_qsoftmax_qpu(
                n_qubits, q_depth, qdevice, qbackend, qpu_shots, layout
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
            # Hard-cut the gradient at the QPU boundary. PennyLane 0.45's
            # `diff_method=None` is not honored when the device has shots —
            # the torch interface still registers a backward hook that falls
            # back to parameter-shift, which then errors on the broadcasted
            # (B*H*N, D) input (#4462). Detaching inputs+weights and wrapping
            # in no_grad bypasses that path entirely. This is consistent with
            # the design intent: the QPU path is inference-only.
            amps_flat = amps.detach().reshape(-1, D).to(torch.float64)
            amps_flat = amps_flat / amps_flat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            with torch.no_grad():
                probs_flat = self.qsoftmax_qpu(amps_flat, self.softmax_weights.detach())
            probs = probs_flat.to(device=amps.device, dtype=amps.dtype).reshape(B, H, N, D)
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
        layout="brick_wall",
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
            layout=layout,
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
    def __init__(self, *args, softmax_layout="brick_wall", **kwargs):
        # Stash the layout BEFORE super().__init__() because the base
        # constructor runs the block-replacement loop (which calls
        # `_build_quantum_block` -> reads `self.softmax_layout`). Plain-string
        # assignment before nn.Module.__init__ is safe.
        if softmax_layout not in LAYOUTS:
            raise ValueError(
                f"unknown softmax_layout {softmax_layout!r}; expected one of {LAYOUTS}"
            )
        self.softmax_layout = softmax_layout
        super().__init__(*args, **kwargs)

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
            layout=getattr(self, "softmax_layout", "brick_wall"),
        )


if __name__ == "__main__":
    import math

    # 1) Gate count per layout matches the canonical formula (correctness check).
    def _expected_count(n, layout):
        if layout == "brick_wall":
            return n - 1
        if layout == "pyramid":
            return n * (n - 1) // 2
        if layout == "x":
            return 2 * n - 3
        if layout == "butterfly":
            return (n // 2) * int(math.log2(n))
        raise AssertionError(layout)

    for n in (4, 8):
        for layout in LAYOUTS:
            got = len(_layout_pairs(n, layout))
            exp = _expected_count(n, layout)
            assert got == exp, f"{layout} N={n}: {got} gates, expected {exp}"
            # all pairs reference valid, distinct wires
            for i, j in _layout_pairs(n, layout):
                assert 0 <= i < n and 0 <= j < n and i != j, f"bad pair ({i},{j})"
    print(f"[validation] PASS: gate counts match formulas for layouts {LAYOUTS}")

    # 2) brick_wall is byte-for-byte the original (even sublayer then odd).
    for n in (4, 5, 8):
        even = [(i, i + 1) for i in range(0, n - 1, 2)]
        odd = [(i, i + 1) for i in range(1, n - 1, 2)]
        assert _layout_pairs(n, "brick_wall") == even + odd, "brick_wall order changed"
    print("[validation] PASS: brick_wall gate order/indexing preserved")

    # 3) butterfly rejects non-power-of-2 qubit counts.
    for bad_n in (5, 6, 12):
        try:
            _layout_pairs(bad_n, "butterfly")
        except ValueError:
            pass
        else:
            raise AssertionError(f"butterfly should reject n_qubits={bad_n}")
    print("[validation] PASS: butterfly rejects non-power-of-2 n_qubits")

    # 4) Per-layout: attention module builds, the weight tensor is sized per
    #    layout, a default.qubit forward runs, and gradients reach softmax_weights.
    torch.manual_seed(0)
    for layout in LAYOUTS:
        attn = Q_E_MHSA(dim=96, head_dim=32, sr_ratio=2, n_qubits=4, q_depth=2, layout=layout)
        exp_pairs = _expected_count(4, layout)
        assert tuple(attn.softmax_weights.shape) == (2, exp_pairs), (
            f"{layout}: weight shape {tuple(attn.softmax_weights.shape)} != (2, {exp_pairs})"
        )
        x = torch.randn(2, 49, 96)
        y = attn(x)
        assert y.shape == (2, 49, 96), f"{layout}: bad output shape {tuple(y.shape)}"
        y.sum().backward()
        g = attn.softmax_weights.grad
        assert g is not None and torch.isfinite(g).all(), f"{layout}: no grad on softmax_weights"
    print("[validation] PASS: all layouts build, forward, and backprop to softmax_weights")

    # 5) End-to-end model build + forward for each layout via the registry.
    from quantum_variants import VARIANTS

    for layout in LAYOUTS:
        model = VARIANTS["softmax"](
            stem_chs=[64, 32, 64], depths=[1, 1, 5, 1], path_dropout=0.0,
            num_classes=3, quantum_stages=(3,), n_qubits=4, q_depth=2,
            softmax_layout=layout,
        )
        model.eval()
        out = model(torch.randn(2, 3, 64, 64))
        assert out.shape == (2, 3), f"{layout}: logits {tuple(out.shape)}"
    print("[validation] PASS: end-to-end model forward ok for every layout")

    # 6) Unknown layout is rejected early.
    try:
        QMedViT_Softmax_Only(
            stem_chs=[64, 32, 64], depths=[1, 1, 5, 1], path_dropout=0.0,
            num_classes=3, softmax_layout="not_a_layout",
        )
    except ValueError:
        print("[validation] PASS: unknown softmax_layout raises ValueError")
    else:
        raise AssertionError("unknown softmax_layout should raise")
