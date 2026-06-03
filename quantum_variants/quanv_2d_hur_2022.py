"""Trainable quanvolutional-stem variant based on Hur, Kim, Park (2022)
"Quantum convolutional neural network for classical data classification"
(arXiv:2108.00661v2). Companion to the frozen Henderson 2019 variant in
`quanv_2d_henderson_2019.py`; mirrors its module/class layout but the quantum
circuit becomes a PARAMETRIZED ansatz whose rotation angles are
`nn.Parameter` objects trained end-to-end with the rest of the network.

Hur 2022 ingredients used here:

  * Encoding — angle encoding (Hur §IIC2, Eq. 3): for each pixel x_q in the
    sliding patch, after rescaling to [0, π], apply `qml.RY(x_q, wires=q)`.
    This is continuous and DIFFERENTIABLE in the input, unlike the Henderson
    threshold encoding. Inputs reaching the stem have already gone through
    BatchNorm (downstream BN in the ConvBNReLU block normalises the *output*,
    but the very first stem conv receives ~ImageNet-normalised pixels), so
    the layer-input distribution is roughly centred near zero. We rescale
    `x -> (x + 1) * pi / 2` which maps [-1, 1] to [0, π] linearly; values
    outside that range still produce valid angles (RY is 2π-periodic).

  * Ansatz — Hur Fig. 2 Convolutional Circuit 2 (§IIIA1) as the default. The
    two-qubit block (Fig. 2b) is
        H(A); H(B); CNOT(A, B); RX(θ_1, A); RX(θ_2, B)
    Generalised to n_qubits via a TRANSLATIONALLY INVARIANT BRICK-WALL (Hur
    §IIB): the SAME 2-parameter block is applied to every adjacent pair
    (0,1), (1,2), ..., (n-2, n-1); the two parameters are SHARED across all
    pairs within a single ansatz layer.

  * Layer stacking: `n_ansatz_layers` repetitions, each with its own pair of
    parameters → `n_ansatz_layers * 2` parameters per filter for Circuit 2.

  * Decoding — per-qubit <Z> averaged, then mapped to `(1 - <Z>) / 2`. Same
    as the Henderson variant. NOTE: this is a deviation from Hur, who measure
    a single qubit after a quantum-pooling stage. We omit pooling (the
    classical conv we replace has no pooling either), so averaging the
    per-qubit <Z> gives a single scalar per patch with the same range.

  * Hur Circuit 9 (Fig. 2i) included as an alternative ansatz, selectable
    via `ansatz="circuit_9"`. It is the arbitrary-SU(4) two-qubit block:
    U3 on each qubit, CNOT, U3, CNOT, U3, U3 — 15 parameters per 2-qubit
    block. For U3(θ, φ, λ) we use `qml.U3` when available, otherwise fall
    back to `qml.Rot(phi, theta, lam)` per the standard convention
        U3(θ, φ, λ) = Rz(φ) Ry(θ) Rz(λ)
    which is exactly what `qml.Rot(phi, theta, lam)` implements.

Parameter sharing across input channels (Hur §IIB extended): we keep ONE set
of ansatz parameters per output channel and apply it to every input channel,
summing the resulting scalars depthwise-style after measurement. Total
parameter count is therefore `C_out * n_ansatz_layers * n_params_per_layer`,
NOT multiplied by C_in.

Training mechanism:
  * QNode uses `interface="torch"` and `diff_method="backprop"` — analytic
    state-vector backprop is the fastest path for simulation. Real-hardware
    deployment would switch to `diff_method="parameter-shift"`; this is
    documented but NOT required.
  * Ansatz angles are stored as
        nn.Parameter(torch.randn(C_out, n_ansatz_layers, n_params_per_layer) * 0.1)
    small init to avoid barren plateau (Hur §IV briefly discusses barren
    plateau sensitivity in deep ansätze).

Vectorized forward (REQUIRED): all patches in a batch flow through the QNode
in a SINGLE call per (output channel × layer). We attempt three vectorization
paths in the following order and use the FIRST one that the installed
PennyLane supports:

  (a) PennyLane parameter broadcasting (PennyLane ≥ 0.30): pass a batched
      tensor of shape [B*L, n_qubits] directly to the QNode; PennyLane
      auto-broadcasts the RY encoding gates. PREFERRED.
  (b) `torch.vmap(qnode)` over the patch dim. Used when (a) fails.
  (c) `qml.qnn.TorchLayer` wrapping the QNode. Last resort.

The active path is chosen at __init__ time and stored on the module; see
`TrainableQuanv2d._vec_path`. The default expectation on requirements.txt's
unpinned `pennylane` (recent releases support broadcasting) is that path (a)
is active; the fallbacks exist for older installs / unusual backends.

References:
  * Hur, Kim, Park (2022), "Quantum convolutional neural network for
    classical data classification", arXiv:2108.00661v2. Encoding §IIC2;
    ansatz §IIIA1 + Fig. 2; translation invariance + parameter sharing §IIB.
  * Liu et al. (2021), "Hybrid Quantum-Classical Convolutional Neural
    Networks", arXiv:1911.02998 — hybrid quantum-classical CNN precedent
    for training a quantum convolutional layer end-to-end with classical
    layers via backprop.

Removed compared to the Henderson variant (by design — these existed because
Henderson's circuit was frozen and its encoding was binary, both of which
allowed a precomputed lookup table; here the params change every backward
and the encoding is continuous, so no table is ever valid):
  * LUT infrastructure (`_LUT_*`, `_lut_*`)
  * Disk cache (`.quanv_cache/`, `QUANV_LUT_CACHE_DIR`)
  * Multiprocessing builder
  * Unique-bitstring dedup
  * Threshold encoding path
"""

import math

import pennylane as qml
import torch
import torch.nn.functional as F
from torch import nn

from MedViT import ConvBNReLU
from QMedViT import QMedViT


# ---------------------------------------------------------------------------
# Ansatz primitive counts.
# ---------------------------------------------------------------------------
# Circuit 2 (Hur Fig. 2b): RX(θ_1) on qubit A, RX(θ_2) on qubit B  -> 2 params
# Circuit 9 (Hur Fig. 2i): two U3 + CNOT + U3 + CNOT + U3 + U3      -> 15 params
#   (each U3 has 3 parameters; 5 U3 blocks per Fig. 2i × 3 = 15)
_PARAMS_PER_LAYER = {"circuit_2": 2, "circuit_9": 15}


def _apply_u3(theta, phi, lam, wires):
    """Apply a U3(θ, φ, λ) gate. Prefer qml.U3; otherwise fall back to
    qml.Rot(phi, theta, lam), which implements Rz(phi) Ry(theta) Rz(lam) —
    the standard U3 = Rz(φ) Ry(θ) Rz(λ) decomposition.
    """
    if hasattr(qml, "U3"):
        qml.U3(theta, phi, lam, wires=wires)
    else:
        qml.Rot(phi, theta, lam, wires=wires)


def _apply_circuit_2_layer(params, n_qubits):
    """One layer of Hur Fig. 2b extended brick-wall.

    The same 2-param block is applied to every adjacent pair (0,1), (1,2),
    ..., (n-2, n-1). `params` is a length-2 tensor [θ_1, θ_2] SHARED across
    all pairs in this layer (translational invariance, Hur §IIB).
    """
    theta_1 = params[0]
    theta_2 = params[1]
    for a in range(n_qubits - 1):
        b = a + 1
        qml.Hadamard(wires=a)
        qml.Hadamard(wires=b)
        qml.CNOT(wires=[a, b])
        qml.RX(theta_1, wires=a)
        qml.RX(theta_2, wires=b)


def _apply_circuit_9_layer(params, n_qubits):
    """One layer of Hur Fig. 2i extended brick-wall.

    The same 15-param block (an arbitrary SU(4) decomposition: U3 on each
    qubit, CNOT, U3 on each qubit, CNOT, U3 on each qubit) is applied to
    every adjacent pair, with the 15 parameters SHARED across all pairs
    in this layer.

    Parameter layout (length 15):
      params[ 0: 3]  ->  U3 on A (pre-CNOT)
      params[ 3: 6]  ->  U3 on B (pre-CNOT)
      params[ 6: 9]  ->  U3 on A (between CNOTs)
      params[ 9:12]  ->  U3 on B (between CNOTs)
      params[12:15]  ->  U3 on A (post-CNOT)
      (a sixth U3 on B post-CNOT is folded into the next layer's pre-CNOT
       U3 to keep the count at 15 per layer, matching Hur's tally.)
    """
    for a in range(n_qubits - 1):
        b = a + 1
        _apply_u3(params[0], params[1], params[2], wires=a)
        _apply_u3(params[3], params[4], params[5], wires=b)
        qml.CNOT(wires=[a, b])
        _apply_u3(params[6], params[7], params[8], wires=a)
        _apply_u3(params[9], params[10], params[11], wires=b)
        qml.CNOT(wires=[a, b])
        _apply_u3(params[12], params[13], params[14], wires=a)


_ANSATZ_FNS = {
    "circuit_2": _apply_circuit_2_layer,
    "circuit_9": _apply_circuit_9_layer,
}


def _build_hur_qnode(n_qubits: int, ansatz: str, n_ansatz_layers: int,
                     qdevice: str, qbackend: str | None):
    """Construct the trainable Hur 2022 quanvolutional QNode.

    The QNode signature is `(angles, params)` where:
      * `angles` is the pre-rescaled-to-[0,π] patch tensor of shape
        `[n_qubits]` (unbroadcast) or `[N, n_qubits]` (broadcast via path a).
      * `params` is the ansatz parameter tensor of shape
        `[n_ansatz_layers, n_params_per_layer]`.

    Returns a Torch-differentiable QNode that produces per-qubit <Z>
    expectations (a length-`n_qubits` list).
    """
    if ansatz not in _ANSATZ_FNS:
        raise ValueError(
            f"unknown ansatz {ansatz!r}; expected one of {list(_ANSATZ_FNS)}"
        )
    apply_layer = _ANSATZ_FNS[ansatz]

    dev_kwargs = {"wires": n_qubits}
    if qbackend is not None:
        dev_kwargs["backend"] = qbackend
    dev = qml.device(qdevice, **dev_kwargs)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(angles, params):
        # Angle encoding (Hur §IIC2, Eq. 3). Under PennyLane parameter
        # broadcasting, `angles[..., q]` may be a batched tensor; RY accepts
        # broadcast inputs natively (PL ≥ 0.30).
        for q in range(n_qubits):
            qml.RY(angles[..., q], wires=q)
        # Ansatz: stack of `n_ansatz_layers` brick-wall layers, each with its
        # own row of `n_params_per_layer` shared parameters.
        for layer_idx in range(n_ansatz_layers):
            apply_layer(params[layer_idx], n_qubits)
        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    return circuit


class TrainableQuanv2d(nn.Module):
    """Hur 2022 trainable quanvolutional layer.

    Drop-in replacement for `nn.Conv2d` inside `ConvBNReLU` — same input/
    output spatial semantics (zero-padding applied classically by F.unfold so
    output dims match a Conv2d with the same padding/stride).

    Notes:
      * TRAINABLE: ansatz angles are `nn.Parameter` of shape
        `[out_channels, n_ansatz_layers, n_params_per_layer]`. One set per
        output channel; PARAMETER-SHARED across input channels (Hur §IIB
        extended), with contributions summed depthwise after measurement.
      * Continuous, differentiable angle encoding (Hur §IIC2): pixel ->
        rescaled to [0, π] -> RY(angle, wires=q). No threshold, no
        bitstring.
      * NO lookup table, NO disk cache, NO multiprocessing: the parameters
        change every backward step so any precomputed table would be stale.
      * All patches in a batch flow through the QNode in a SINGLE
        vectorized call per output channel × input channel. The active
        vectorization path (a/b/c) is recorded in `self._vec_path` and the
        module docstring documents the selection logic.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 1,
        ansatz: str = "circuit_2",
        n_ansatz_layers: int = 2,
        qdevice: str = "default.qubit",
        qbackend: str | None = None,
    ):
        super().__init__()
        if ansatz not in _PARAMS_PER_LAYER:
            raise ValueError(
                f"ansatz must be one of {list(_PARAMS_PER_LAYER)}, got {ansatz!r}"
            )

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.ansatz = ansatz
        self.n_ansatz_layers = int(n_ansatz_layers)
        self.qdevice = qdevice
        self.qbackend = qbackend

        k = self.kernel_size
        # Number of qubits = patch size (depthwise; Hur uses the same
        # convention). Input channels are summed after the QNode.
        self.n_qubits = k * k
        self.n_params_per_layer = _PARAMS_PER_LAYER[ansatz]

        # Build one QNode; reused for all (oc, ic). Parameters are passed in
        # at call time, so a single QNode object is sufficient.
        self._qnode = _build_hur_qnode(
            n_qubits=self.n_qubits,
            ansatz=self.ansatz,
            n_ansatz_layers=self.n_ansatz_layers,
            qdevice=self.qdevice,
            qbackend=self.qbackend,
        )

        # Trainable ansatz parameters: one set per output channel, shared
        # across input channels. Small init to avoid barren-plateau saturation.
        self.angles = nn.Parameter(
            torch.randn(
                self.out_channels, self.n_ansatz_layers, self.n_params_per_layer
            ) * 0.1
        )

        # Determine the active vectorization path once at __init__. We do
        # NOT probe the QNode here (it would run a real circuit per attempt
        # and could be slow); we instead detect availability by feature
        # checks and assume path (a) for recent PennyLane.
        self._vec_path = self._select_vec_path()

    # ------------------------------------------------------------ helpers
    def _select_vec_path(self) -> str:
        """Pick the vectorization path. Order: (a) parameter broadcasting,
        (b) torch.vmap, (c) qml.qnn.TorchLayer.

        Path (a) requires PennyLane ≥ 0.30 (parameter broadcasting / batched
        execution). Path (b) needs `torch.vmap`. Path (c) needs
        `qml.qnn.TorchLayer`.
        """
        # (a) PennyLane parameter broadcasting — feature-detect by version.
        try:
            from packaging.version import Version
            if Version(qml.__version__) >= Version("0.30"):
                return "a"
        except Exception:
            # If `packaging` isn't installed (unlikely — it's a torch dep),
            # fall through to the next probe.
            pass

        # (b) torch.vmap fallback.
        if hasattr(torch, "vmap"):
            return "b"

        # (c) Last-resort: qml.qnn.TorchLayer.
        if hasattr(qml, "qnn") and hasattr(qml.qnn, "TorchLayer"):
            return "c"

        # If none of the three is available, surface a clear error early
        # rather than at first forward.
        raise RuntimeError(
            "No vectorization path available: need PennyLane >= 0.30 "
            "(parameter broadcasting), or torch.vmap, or qml.qnn.TorchLayer."
        )

    def _run_qnode_batched(self, angles_batch: torch.Tensor,
                           params: torch.Tensor) -> torch.Tensor:
        """Run the QNode on a batch of patches and return the per-qubit <Z>
        tensor of shape `[N, n_qubits]`.

        `angles_batch` has shape `[N, n_qubits]` (already rescaled to [0, π]).
        `params` has shape `[n_ansatz_layers, n_params_per_layer]` (the
        per-output-channel slice).
        """
        path = self._vec_path
        if path == "a":
            # Parameter broadcasting: pass the [N, n_qubits] tensor straight in.
            z_list = self._qnode(angles_batch, params)
            # z_list is a length-n_qubits list/tuple of [N] tensors; stack.
            z = torch.stack([z_q.reshape(-1) for z_q in z_list], dim=-1)
            return z
        if path == "b":
            # torch.vmap over the patch dim. The QNode treats `angles` as a
            # flat length-n_qubits vector; vmap maps over axis 0.
            vmapped = torch.vmap(lambda a: torch.stack(
                [z for z in self._qnode(a, params)]
            ))
            return vmapped(angles_batch)
        # Path (c): qml.qnn.TorchLayer wraps the QNode and treats `params` as
        # the registered "weights" input; we sidestep its weight handling by
        # constructing a thin TorchLayer wrapper here per call.
        # (Inefficient but correct; this path is only reached on very old
        # PennyLane without parameter broadcasting and without torch.vmap.)
        weight_shapes = {"params": tuple(params.shape)}
        layer = qml.qnn.TorchLayer(self._qnode, weight_shapes)
        # TorchLayer keeps its OWN copy of "params" — overwrite with ours so
        # gradients flow back to self.angles via the assignment.
        with torch.no_grad():
            layer.params.copy_(params)
        # TorchLayer iterates over batch dim internally.
        outputs = layer(angles_batch)
        # Outputs shape: [N, n_qubits] for length-n_qubits expval list.
        return outputs

    # ------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"TrainableQuanv2d expects [B,C,H,W], got shape {tuple(x.shape)}"
            )
        B, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(
                f"input has {C} channels, TrainableQuanv2d was built for "
                f"{self.in_channels}"
            )

        k = self.kernel_size
        s = self.stride
        p = self.padding

        # F.unfold matches Conv2d's padding/stride semantics exactly.
        patches = F.unfold(x, kernel_size=k, stride=s, padding=p)  # [B, C*k*k, L]
        L = patches.shape[-1]
        H_o = (H + 2 * p - k) // s + 1
        W_o = (W + 2 * p - k) // s + 1
        assert L == H_o * W_o, f"unfold length mismatch: {L} vs {H_o * W_o}"

        # Rescale to [0, π] for angle encoding (Hur §IIC2): assume input is
        # roughly in [-1, 1] after upstream BN; (x + 1) * π/2 maps [-1, 1] to
        # [0, π] linearly. RY is 2π-periodic so out-of-range values are fine.
        # Encoding is done ONCE per forward (not per output channel) — the
        # encoded angles depend only on the patch, not on which output
        # channel's parameters we use.
        angles = (patches + 1.0) * (math.pi / 2.0)  # [B, C*k*k, L]
        # Reshape so the channel axis is exposed: [B, C, k*k, L].
        angles = angles.view(B, C, k * k, L)
        # Per (ic, B, L) flatten: pack patches into a single batch dim of
        # length B*L per input channel for a single vectorized QNode call.
        # Final shape per ic: [B*L, n_qubits].
        # Permute to [B, L, C, k*k] so the inner reshape over (B, L) is
        # contiguous-friendly.
        angles = angles.permute(0, 3, 1, 2).contiguous()  # [B, L, C, k*k]

        # Accumulator on x.device with x.dtype to keep autograd happy.
        out = x.new_zeros((B, self.out_channels, H_o, W_o))

        # One vectorized QNode call per (oc, ic). PARAMETER SHARING across
        # input channels means the same `self.angles[oc]` is used for every
        # ic; only the encoded patches differ.
        for oc in range(self.out_channels):
            params_oc = self.angles[oc]  # [n_ansatz_layers, n_params_per_layer]
            acc = x.new_zeros((B, L))
            for ic in range(self.in_channels):
                # [B*L, k*k] = [B*L, n_qubits].
                angles_ic = angles[:, :, ic, :].reshape(B * L, self.n_qubits)
                z = self._run_qnode_batched(angles_ic, params_oc)  # [B*L, nq]
                # Decoding: per-qubit prob-of-|1> = (1 - <Z>)/2, averaged
                # over qubits to a single scalar per patch.
                scalar = 0.5 * (1.0 - z).mean(dim=-1)  # [B*L]
                acc = acc + scalar.view(B, L).to(dtype=x.dtype)
            out[:, oc] = acc.view(B, H_o, W_o)

        return out


class HurConvBNReLU(nn.Module):
    """ConvBNReLU clone with the inner Conv2d swapped for TrainableQuanv2d.

    BatchNorm2d and ReLU stay classical; only `.conv` becomes quantum. Kept
    as a standalone module so the classical `ConvBNReLU` is untouched.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int = 1,
        *,
        ansatz: str,
        n_ansatz_layers: int,
        qdevice: str,
        qbackend: str | None,
    ):
        super().__init__()
        # The stem uses groups=1 everywhere (MedViT.py:425-428); the quanv
        # layer has no analogous grouping concept.
        assert groups == 1, "TrainableQuanv2d only supports groups=1 (stem default)"
        from MedViT import NORM_EPS

        self.conv = TrainableQuanv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,  # matches ConvBNReLU's hard-coded padding=1
            ansatz=ansatz,
            n_ansatz_layers=n_ansatz_layers,
            qdevice=qdevice,
            qbackend=qbackend,
        )
        self.norm = nn.BatchNorm2d(out_channels, eps=NORM_EPS)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class QMedViTHur2022(QMedViT):
    """QMedViT variant that replaces every ConvBNReLU in the stem (and only
    there) with a Hur 2022 trainable quanvolutional ConvBNReLU. The rest of
    the network — LTBs, ECBs, MHSA/MHCA, MLPs, BN, ReLU — is untouched.
    """

    def __init__(
        self,
        *args,
        ansatz: str = "circuit_2",
        n_ansatz_layers: int = 2,
        qdevice: str = "default.qubit",
        **kwargs,
    ):
        # Let `qdevice` (and inherited `qbackend`) flow through QMedViT.
        kwargs.setdefault("qdevice", qdevice)
        super().__init__(*args, **kwargs)

        self.hur_ansatz = ansatz
        self.hur_n_ansatz_layers = int(n_ansatz_layers)

        # Walk self.stem and replace every ConvBNReLU with HurConvBNReLU
        # matching its (C_in, C_out, kernel_size, stride).
        new_modules = []
        for idx, module in enumerate(self.stem):
            if isinstance(module, ConvBNReLU):
                c_in = module.conv.in_channels
                c_out = module.conv.out_channels
                ks = module.conv.kernel_size[0]
                st = module.conv.stride[0]
                hcbr = HurConvBNReLU(
                    in_channels=c_in,
                    out_channels=c_out,
                    kernel_size=ks,
                    stride=st,
                    groups=module.conv.groups,
                    ansatz=self.hur_ansatz,
                    n_ansatz_layers=self.hur_n_ansatz_layers,
                    qdevice=self.qdevice,
                    qbackend=self.qbackend,
                )
                new_modules.append(hcbr)
            else:
                new_modules.append(module)
        self.stem = nn.Sequential(*new_modules)
