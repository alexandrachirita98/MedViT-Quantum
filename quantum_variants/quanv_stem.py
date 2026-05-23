"""Quanvolutional Stem variant: replaces the first ConvBNReLU of MedViT's
stem with a Quanvolutional layer (Henderson, Shakya, Pradhan, Cook, 2019,
"Quanvolutional Neural Networks: Powering Image Recognition with Quantum
Circuits", arXiv:1904.04767, §2.3 and §3.4–3.5).

Each output channel is one random quantum circuit sampled once at __init__
and frozen — there are no trainable quantum parameters. Encoding is the
paper's threshold scheme (§3.5); decoding is sum_i <Z_i> per patch.

Because threshold encoding has a finite input space (2^(k*k) bitstrings per
patch), all input->output mappings are precomputed into a lookup table at
init (paper §3.5). forward() never executes a live circuit — it indexes
the table — which is the only tractable way to run a quanvolution over a
112x112 spatial grid with 64 filters.

Unlike softmax_only, this variant ignores quantum_stages /
quantum_block_indices: the quantum op lives in the stem, not in the
transformer stages.
"""

import numpy as np
import pennylane as qml
import torch
import torch.nn.functional as F
from torch import nn

from MedViT import NORM_EPS
from QMedViT import QMedViT


_ONE_QUBIT_GATES = ("RX", "RY", "RZ", "Rot", "PhaseShift", "T", "Hadamard")
_TWO_QUBIT_GATES = ("CNOT", "SWAP", "SISWAP", "CRZ")


def _sample_circuit_ops(n_qubits: int, connection_prob: float, rng: np.random.Generator):
    """Paper §3.4: per-pair 2q gates with probability `connection_prob`, then
    a uniform random count in [0, 2*n_qubits] of 1q gates, then shuffle."""
    ops = []
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            if rng.random() < connection_prob:
                gate = rng.choice(_TWO_QUBIT_GATES)
                if gate == "CRZ":
                    theta = float(rng.uniform(0, 2 * np.pi))
                    ops.append(("CRZ", (i, j), (theta,)))
                else:
                    ops.append((gate, (i, j), ()))

    n_one = int(rng.integers(0, 2 * n_qubits + 1))
    for _ in range(n_one):
        gate = rng.choice(_ONE_QUBIT_GATES)
        wire = int(rng.integers(0, n_qubits))
        if gate in ("RX", "RY", "RZ", "PhaseShift"):
            theta = float(rng.uniform(0, 2 * np.pi))
            ops.append((gate, (wire,), (theta,)))
        elif gate == "Rot":
            a, b, c = (float(rng.uniform(0, 2 * np.pi)) for _ in range(3))
            ops.append((gate, (wire,), (a, b, c)))
        else:
            ops.append((gate, (wire,), ()))

    rng.shuffle(ops)
    return ops


def _apply_ops(ops):
    gate_table = {
        "RX": qml.RX, "RY": qml.RY, "RZ": qml.RZ,
        "Rot": qml.Rot, "PhaseShift": qml.PhaseShift,
        "T": qml.T, "Hadamard": qml.Hadamard,
        "CNOT": qml.CNOT, "SWAP": qml.SWAP,
        "SISWAP": qml.SISWAP, "CRZ": qml.CRZ,
    }
    for name, wires, params in ops:
        gate_table[name](*params, wires=list(wires))


def _build_lookup_table(ops, n_qubits: int, qdevice: str, qbackend: str | None) -> torch.Tensor:
    """Compute the unitary U of the circuit (no inputs, just the random ops),
    then derive the table classically: for basis input |k>, output state is
    column U[:, k], so <Z_i> per qubit becomes a weighted parity sum over the
    output probabilities. This avoids 2^n separate QNode invocations."""
    dev_kwargs = {"wires": n_qubits}
    if qbackend is not None:
        dev_kwargs["backend"] = qbackend
    dev = qml.device(qdevice, **dev_kwargs)

    @qml.qnode(dev)
    def circuit():
        _apply_ops(ops)
        return qml.state()

    # qml.matrix needs an executed QNode to bind the wire order
    U = qml.matrix(circuit, wire_order=list(range(n_qubits)))()
    U = np.asarray(U)

    probs = np.abs(U) ** 2                      # (D, D); probs[j, k] = P(out=j | in=k)
    dim = 1 << n_qubits
    idx = np.arange(dim, dtype=np.int64)
    # n_qubits - 2 * hamming_weight(j) == sum_i <Z_i> in the computational basis |j>
    hamming = np.array([bin(j).count("1") for j in idx], dtype=np.float64)
    z_sum_per_basis = n_qubits - 2.0 * hamming  # (D,)
    table = probs.T @ z_sum_per_basis           # (D,), expectation per input basis state
    return torch.from_numpy(table.astype(np.float32))


class QuanvStem(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        n_qubits: int | None = None,
        connection_prob: float = 0.3,
        seed: int = 0,
        qdevice: str = "default.qubit",
        qbackend: str | None = None,
    ):
        super().__init__()
        if n_qubits is None:
            n_qubits = kernel_size * kernel_size
        if n_qubits != kernel_size * kernel_size:
            raise ValueError(
                "QuanvStem requires n_qubits == kernel_size**2 (no ancillas in this implementation)"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.n_qubits = n_qubits

        rng = np.random.default_rng(seed)
        tables = []
        for _ in range(out_channels):
            ops = _sample_circuit_ops(n_qubits, connection_prob, rng)
            tables.append(_build_lookup_table(ops, n_qubits, qdevice, qbackend))
        # (out_channels, 2**n_qubits). Buffer (not Parameter) — quanv is frozen.
        self.register_buffer("lookup", torch.stack(tables, dim=0))

        # Bit-position weights for converting a kxk binary patch to its
        # integer basis-state index. PennyLane state-vector convention: wire 0
        # is the MSB of the index, so the leftmost (wire-0) patch cell carries
        # weight 2**(n_qubits-1). Row-major: cell 0 = top-left = wire 0.
        bit_w = (1 << torch.arange(n_qubits - 1, -1, -1, dtype=torch.long)).view(1, 1, n_qubits)
        self.register_buffer("bit_weights", bit_w)

        self.bn = nn.BatchNorm2d(out_channels, eps=NORM_EPS)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        k, s, p = self.kernel_size, self.stride, self.padding
        H_out = (H + 2 * p - k) // s + 1
        W_out = (W + 2 * p - k) // s + 1

        # (B, C*k*k, L) -> per-spatial-cell mean over RGB -> threshold at 0.
        # Mean-over-channels threshold is the natural extension of the paper's
        # ">0" scheme to multi-channel inputs (sum>0 is equivalent up to scale).
        patches = F.unfold(x, kernel_size=k, stride=s, padding=p)         # (B, C*k*k, L)
        L = patches.shape[-1]
        patches = patches.view(B, C, k * k, L).mean(dim=1)                # (B, k*k, L)
        bits = (patches > 0).long().transpose(1, 2)                       # (B, L, k*k)
        idx = (bits * self.bit_weights).sum(dim=-1)                       # (B, L), in [0, 2**n_qubits)

        # Gather one scalar per (filter, patch) from the precomputed table.
        # lookup: (out_channels, D). Flatten the gather: (B*L,) -> (out_channels, B*L).
        flat_idx = idx.reshape(-1)                                        # (B*L,)
        feat = self.lookup.index_select(dim=1, index=flat_idx)            # (out_channels, B*L)
        feat = feat.view(self.out_channels, B, L).permute(1, 0, 2)        # (B, out_channels, L)
        feat = feat.view(B, self.out_channels, H_out, W_out).contiguous()

        return self.act(self.bn(feat))


class QMedViT_Quanv_Stem(QMedViT):
    def __init__(
        self,
        *args,
        quanv_kernel_size: int = 3,
        quanv_connection_prob: float = 0.3,
        quanv_seed: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        stem_out = self.stem[0].conv.out_channels
        self.stem[0] = QuanvStem(
            in_channels=3,
            out_channels=stem_out,
            kernel_size=quanv_kernel_size,
            stride=2,
            padding=quanv_kernel_size // 2,
            connection_prob=quanv_connection_prob,
            seed=quanv_seed,
            qdevice=self.qdevice,
            qbackend=self.qbackend,
        )

    def _should_quantize_block(self, block, stage_id, block_idx) -> bool:
        return False

    def _build_quantum_block(self, block, stage_id, block_idx, **ctx) -> nn.Module:
        return block
