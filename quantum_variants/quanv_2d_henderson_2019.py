"""Quanvolutional-stem variant: replaces every `nn.Conv2d` inside the stem's
ConvBNReLU blocks (MedViT.py:425-428) with a faithful Henderson et al. 2019
quanvolutional layer (Henderson, Shakya, Pradhan, Cook, "Quanvolutional Neural
Networks: Powering Image Recognition with Quantum Circuits", arXiv:1904.04767).

BatchNorm2d and ReLU stay classical; only the convolution operator is swapped.
The quanv layer is the ORIGINAL non-trainable variant from the 2019 paper
(Sections 3.4 and 3.5):

  * fixed random circuits per output channel, frozen at __init__
  * threshold encoding: pixel > threshold => PauliX on that qubit, else |0>
  * the circuit body follows the §3.4 sampler verbatim: every unordered qubit
    pair (i, j) is independently assigned a 2-qubit gate with probability
    `connection_prob` (Erdős–Rényi edge inclusion), the gate type is drawn
    uniformly from {CNOT, SWAP, SqrtSWAP, ControlledU}; additionally a random
    integer count in [0, 2 * n_qubits] of 1-qubit gates is drawn uniformly
    from {X(θ), Y(θ), Z(θ), U(θ), P, T, H} with target qubits sampled
    uniformly and rotation angles sampled from U(0, 2π). After all gates are
    selected, the order is shuffled.
  * decoding via per-qubit <Z> expectations (analytic, default) or finite-shot
    bitstring sampling with mean popcount (faithful to §3.5; the literal
    most-likely-bitstring decoder is left as a TODO)

Implementation choices (paper is underspecified):

  1. Connection probability is a single global value `p_connect` for every
     unordered qubit pair.
  2. For each pair (i, j) with i < j, draw a fresh uniform value in [0, 1)
     from the seeded RNG; if < `p_connect`, the pair gets a 2-qubit gate.
  3. The 2-qubit gate type is uniform over {CNOT, SWAP, SqrtSWAP, ControlledU}.
  4. The number of 1-qubit gates is a uniform integer in [0, 2 * n_qubits]
     inclusive (paper writes "[0, 2n²]" where n is the patch side and
     n_qubits = n²; so 2n² = 2·n_qubits).
  5. The 1-qubit gate type is uniform over {X(θ), Y(θ), Z(θ), U(θ), P, T, H}.
  6. The target qubit of each 1-qubit gate is uniform over {0, ..., n_qubits-1}.
  7. Rotation angles are uniform on [0, 2π].
  8. The final order is produced by concatenating the selected 2-qubit and
     1-qubit gates into one list and calling `rng.shuffle(list)`.

Literal Henderson uses n_qubits = C_in * k * k (27 qubits for the MedViT 3->C
stem with k=3), which exceeds default.qubit's practical limit. We default to
`channel_wise=True`: run the quanv depthwise per input channel with
n_qubits = k*k and sum the contributions across input channels. This is a
NISQ-feasible compromise documented here as a deviation from the literal paper;
the literal C_in*k*k case is available via `channel_wise=False`.
"""

import hashlib
import os
from functools import partial

import numpy as np
import pennylane as qml
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from MedViT import ConvBNReLU
from QMedViT import QMedViT


def _build_quanv_circuit(
    n_qubits: int,
    depth: int,
    qdevice: str,
    qbackend: str | None,
    seed: int,
    decoding: str,
    n_shots: int,
    connection_prob: float = 0.5,
):
    """Construct one fixed random Henderson 2019 §3.4 quanv circuit.

    Implements the §3.4 sampler verbatim: a graph-style sampler that, for every
    unordered qubit pair (i, j), independently decides whether to apply a
    2-qubit gate (with probability `connection_prob`); additionally a random
    integer number of 1-qubit gates in [0, 2 * n_qubits] is sampled. After all
    selections, the gate order is shuffled by the seeded RNG.

    Implementation choices (paper is underspecified):
        1. Connection probability is a single global value `p_connect`
           (parameter `connection_prob`) shared by every unordered qubit pair.
        2. Decision rule per pair: for each pair (i, j) with i < j, draw a
           fresh uniform value in [0, 1) from the seeded RNG; if < `p_connect`,
           the pair gets a 2-qubit gate (Erdős–Rényi edge inclusion).
        3. The 2-qubit gate type is uniform over
           {CNOT, SWAP, SqrtSWAP, ControlledU}.
        4. The number of 1-qubit gates is a uniform integer in
           [0, 2 * n_qubits] inclusive (paper "[0, 2n²]" with n_qubits = n²,
           so 2n² = 2 · n_qubits).
        5. The 1-qubit gate type is uniform over
           {X(θ), Y(θ), Z(θ), U(θ), P, T, H}.
        6. The target qubit of each 1-qubit gate is uniform over
           {0, ..., n_qubits-1}.
        7. Rotation angles are uniform on [0, 2π].
        8. Final ordering: concatenate the selected 2-qubit and 1-qubit gates
           into one list and call `rng.shuffle(list)`.

    PennyLane mapping:
        X(θ)        -> qml.RX(θ, wires=q)                       (1 angle)
        Y(θ)        -> qml.RY(θ, wires=q)                       (1 angle)
        Z(θ)        -> qml.RZ(θ, wires=q)                       (1 angle)
        U(θ)        -> qml.Rot(phi, theta, omega, wires=q)      (3 angles)
        P           -> qml.S(wires=q)                           (0 angles)
        T           -> qml.T(wires=q)                           (0 angles)
        H           -> qml.Hadamard(wires=q)                    (0 angles)
        CNOT        -> qml.CNOT(wires=[i, j])                   (0 angles)
        SWAP        -> qml.SWAP(wires=[i, j])                   (0 angles)
        SqrtSWAP    -> qml.SISWAP(wires=[i, j])                 (0 angles)
        ControlledU -> qml.CRot(phi, theta, omega, wires=[i, j])(3 angles)

    SqrtSWAP fallback: if `qml.SISWAP` is not available in the installed
    PennyLane version, fall back to a manual decomposition that realises the
    square root of SWAP via the standard identity
    SqrtSWAP = CNOT(i,j) · (I ⊗ ((1+i)/2 · I + (1-i)/2 · X)) · CNOT(i,j) ·
               diag(1, 1, 1, e^{iπ/2})
    expressed with available primitives. The simplest equivalent decomposition
    we apply is:
        CNOT(i, j); CRX(π/2, wires=[j, i]); CNOT(i, j)
    (a CNOT-conjugated controlled-RX(π/2)) which yields a unitary
    proportional to SqrtSWAP up to a diagonal phase that is absorbed by the
    threshold encoding / <Z>-decoding (both are insensitive to a global
    phase). This fallback is only used when `qml.SISWAP` is missing.

    `depth` reinterpretation: the paper has no depth concept; a single
    repetition of the §3.4 sampler is the literal procedure. We reinterpret
    `depth` as "repeat the entire §3.4 sampling `depth` times, with each
    repetition shuffled independently; concatenate the resulting lists WITHOUT
    re-shuffling across repetitions." The default `depth=1` therefore matches
    the paper exactly.

    gate_plan tuple schema (one entry per gate, in execution order):
        ("1q", name, (angle, ...), target_qubit)
            name in {"RX", "RY", "RZ", "Rot", "S", "T", "H"}; the angle tuple
            has length 1 for RX/RY/RZ, 3 for Rot, 0 for S/T/H.
        ("2q", name, (angle, ...), wire_i, wire_j)
            name in {"CNOT", "SWAP", "SqrtSWAP", "CRot"}; the angle tuple has
            length 0 for CNOT/SWAP/SqrtSWAP, 3 for CRot.

    Returns:
        (qnode, gate_plan). The gate plan is kept around purely for
        reproducibility / inspection. The QNode takes a binary bitstring (one
        bit per qubit) as input — threshold encoding is applied classically
        before the QNode is called, so the circuit body itself only consists
        of the frozen random gates plus the chosen measurement.
    """
    rng = np.random.default_rng(seed)

    # Detect SqrtSWAP availability once, per build.
    _has_siswap = hasattr(qml, "SISWAP")

    one_qubit_names = ("RX", "RY", "RZ", "Rot", "S", "T", "H")
    two_qubit_names = ("CNOT", "SWAP", "SqrtSWAP", "CRot")

    def _sample_one_repetition():
        """One literal §3.4 sampling, shuffled in place."""
        local_plan = []

        # --- 2-qubit gates: Erdős–Rényi over unordered pairs (i < j) ---
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                if float(rng.random()) < connection_prob:
                    name = two_qubit_names[int(rng.integers(0, len(two_qubit_names)))]
                    if name == "CRot":
                        angles = (
                            float(rng.uniform(0.0, 2.0 * np.pi)),
                            float(rng.uniform(0.0, 2.0 * np.pi)),
                            float(rng.uniform(0.0, 2.0 * np.pi)),
                        )
                    else:
                        angles = ()
                    local_plan.append(("2q", name, angles, int(i), int(j)))

        # --- 1-qubit gates: uniform integer count in [0, 2 * n_qubits] ---
        n_one = int(rng.integers(0, 2 * n_qubits + 1))  # inclusive upper bound
        for _ in range(n_one):
            name = one_qubit_names[int(rng.integers(0, len(one_qubit_names)))]
            target = int(rng.integers(0, n_qubits))
            if name in ("RX", "RY", "RZ"):
                angles = (float(rng.uniform(0.0, 2.0 * np.pi)),)
            elif name == "Rot":
                angles = (
                    float(rng.uniform(0.0, 2.0 * np.pi)),
                    float(rng.uniform(0.0, 2.0 * np.pi)),
                    float(rng.uniform(0.0, 2.0 * np.pi)),
                )
            else:
                angles = ()
            local_plan.append(("1q", name, angles, target))

        # --- Shuffle the combined 1q + 2q list in place ---
        rng.shuffle(local_plan)
        return local_plan

    # `depth` reinterpretation: independent §3.4 samplings, concatenated
    # without re-shuffling across repetitions.
    gate_plan = []
    for _ in range(max(1, int(depth))):
        gate_plan.extend(_sample_one_repetition())

    dev_kwargs = {"wires": n_qubits}
    if decoding == "counts" and n_shots > 0:
        dev_kwargs["shots"] = n_shots
    if qbackend is not None:
        dev_kwargs["backend"] = qbackend
    dev = qml.device(qdevice, **dev_kwargs)

    def _apply_encoding(bits):
        # Threshold encoding: bit == 1 -> PauliX (|1>), bit == 0 -> leave |0>.
        for q in range(n_qubits):
            if int(bits[q]) == 1:
                qml.PauliX(wires=q)

    def _apply_sqrt_swap(i, j):
        # Manual fallback for SqrtSWAP when qml.SISWAP is unavailable.
        # Yields a unitary proportional to SqrtSWAP up to a global/diagonal
        # phase that is invisible to threshold encoding + <Z>/popcount
        # decoding (both insensitive to such phases).
        qml.CNOT(wires=[i, j])
        qml.CRX(np.pi / 2.0, wires=[j, i])
        qml.CNOT(wires=[i, j])

    def _apply_random_circuit():
        for entry in gate_plan:
            kind = entry[0]
            if kind == "1q":
                _, name, angles, q = entry
                if name == "RX":
                    qml.RX(angles[0], wires=q)
                elif name == "RY":
                    qml.RY(angles[0], wires=q)
                elif name == "RZ":
                    qml.RZ(angles[0], wires=q)
                elif name == "Rot":
                    qml.Rot(angles[0], angles[1], angles[2], wires=q)
                elif name == "S":
                    qml.S(wires=q)
                elif name == "T":
                    qml.T(wires=q)
                elif name == "H":
                    qml.Hadamard(wires=q)
                else:
                    raise ValueError(f"unknown 1q gate name: {name!r}")
            else:
                _, name, angles, i, j = entry
                if name == "CNOT":
                    qml.CNOT(wires=[i, j])
                elif name == "SWAP":
                    qml.SWAP(wires=[i, j])
                elif name == "SqrtSWAP":
                    if _has_siswap:
                        qml.SISWAP(wires=[i, j])
                    else:
                        _apply_sqrt_swap(i, j)
                elif name == "CRot":
                    qml.CRot(angles[0], angles[1], angles[2], wires=[i, j])
                else:
                    raise ValueError(f"unknown 2q gate name: {name!r}")

    if decoding == "expval":
        # Analytic, no shots. Diff method is intentionally None — the circuit
        # is frozen and the encoding is non-differentiable (threshold).
        @qml.qnode(dev, interface="numpy", diff_method=None)
        def circuit(bits):
            _apply_encoding(bits)
            _apply_random_circuit()
            # TODO: §3.5 most-likely-bitstring decoding is out of scope here;
            # we return per-qubit <Z> (collapsed downstream to mean prob-|1>).
            return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]
    else:
        # Counts mode (faithful to §3.5): finite-shot bitstring samples.
        @qml.qnode(dev, interface="numpy", diff_method=None)
        def circuit(bits):
            _apply_encoding(bits)
            _apply_random_circuit()
            # TODO: §3.5 most-likely-bitstring decoding is out of scope here;
            # downstream collapses samples to a mean popcount scalar.
            return qml.sample(wires=range(n_qubits))

    # `_apply_random_circuit` (the frozen body, WITHOUT encoding/measurement) is
    # returned so callers can build a broadcasting QNode for fast LUT
    # construction (see Quanv2d._build_lut). It captures the same gate_plan and
    # SqrtSWAP fallback, guaranteeing identical gates to `circuit`.
    return circuit, gate_plan, _apply_random_circuit


class Quanv2d(nn.Module):
    """Henderson 2019 quanvolutional layer.

    Drop-in replacement for `nn.Conv2d` inside `ConvBNReLU` — same input/output
    spatial semantics (with zero-padding applied classically before the sliding
    window so output dims match a Conv2d with the same padding/stride).

    Notes:
        * Strictly non-trainable. There are NO nn.Parameter members; the random
          circuits are frozen at construction.
        * For the `expval` path with threshold encoding, the patch space is
          binary (2^n_qubits bitstrings). We classicalize the patch->bitstring
          mapping and call the QNode once per *unique* bitstring observed in a
          forward, then scatter results back — a major speedup vs. one QNode
          call per patch.
        * Because the circuits are FROZEN and the input is always a
          computational basis state (threshold encoding -> PauliX), each circuit
          is a deterministic function of just 2^n_qubits possible bitstrings.
          When n_qubits is small enough (<= _LUT_MAX_QUBITS, i.e. channel_wise),
          we precompute the full lookup table ONCE at __init__ and the forward
          becomes pure tensor indexing — no QNode calls in the hot path. The
          table is a registered buffer (moves to GPU with the model, persists in
          state_dict). For large n_qubits (the literal non-channel_wise case) the
          table is infeasible, so `self._lut is None` and forward falls back to
          the per-unique-bitstring QNode path.
    """

    # 2^n_qubits entries per circuit; above this the lookup table is infeasible
    # (memory + build time) and we fall back to the QNode path.
    _LUT_MAX_QUBITS = 16

    # Bump whenever the LUT-construction logic changes in a way that alters its
    # values; old on-disk caches with a different version are ignored.
    _LUT_BUILD_VERSION = 1

    # Default cache directory; overridable per-instance via `lut_cache_dir` or
    # globally via the QUANV_LUT_CACHE_DIR environment variable.
    _LUT_CACHE_DIR_DEFAULT = ".quanv_cache"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 1,
        threshold: float = 0.0,
        depth: int = 2,
        decoding: str = "expval",
        n_shots: int = 0,
        channel_wise: bool = True,
        qdevice: str = "default.qubit",
        qbackend: str | None = None,
        seed: int = 0,
        connection_prob: float = 0.5,
        lut_cache_dir: str | None = None,
        rebuild_lut: bool = False,
    ):
        super().__init__()
        if decoding not in ("expval", "counts"):
            raise ValueError(f"decoding must be 'expval' or 'counts', got {decoding!r}")
        if decoding == "counts" and n_shots <= 0:
            raise ValueError("decoding='counts' requires n_shots > 0")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.threshold = float(threshold)
        self.depth = int(depth)
        self.decoding = decoding
        self.n_shots = int(n_shots)
        self.channel_wise = bool(channel_wise)
        self.qdevice = qdevice
        self.qbackend = qbackend
        self.seed = int(seed)
        self.connection_prob = float(connection_prob)
        # Env var overrides the constructor default; explicit arg wins over both.
        self.lut_cache_dir = (
            lut_cache_dir
            if lut_cache_dir is not None
            else os.environ.get("QUANV_LUT_CACHE_DIR", self._LUT_CACHE_DIR_DEFAULT)
        )
        self.rebuild_lut = bool(rebuild_lut)

        k = self.kernel_size
        if self.channel_wise:
            self.n_qubits = k * k
            # Per (output, input) channel circuit, summed over input channels.
            self._qnodes = []
            self._gate_plans = []
            self._bodies = []
            for oc in range(out_channels):
                row_qnodes = []
                row_plans = []
                row_bodies = []
                for ic in range(in_channels):
                    sub_seed = (self.seed * 1_000_003 + oc * 1009 + ic) & 0x7FFFFFFF
                    qnode, plan, body = _build_quanv_circuit(
                        n_qubits=self.n_qubits,
                        depth=self.depth,
                        qdevice=self.qdevice,
                        qbackend=self.qbackend,
                        seed=sub_seed,
                        decoding=self.decoding,
                        n_shots=self.n_shots,
                        connection_prob=self.connection_prob,
                    )
                    row_qnodes.append(qnode)
                    row_plans.append(plan)
                    row_bodies.append(body)
                self._qnodes.append(row_qnodes)
                self._gate_plans.append(row_plans)
                self._bodies.append(row_bodies)
        else:
            self.n_qubits = in_channels * k * k
            self._qnodes = []
            self._gate_plans = []
            self._bodies = []
            for oc in range(out_channels):
                sub_seed = (self.seed * 1_000_003 + oc * 1009) & 0x7FFFFFFF
                qnode, plan, body = _build_quanv_circuit(
                    n_qubits=self.n_qubits,
                    depth=self.depth,
                    qdevice=self.qdevice,
                    qbackend=self.qbackend,
                    seed=sub_seed,
                    decoding=self.decoding,
                    n_shots=self.n_shots,
                    connection_prob=self.connection_prob,
                )
                self._qnodes.append(qnode)
                self._gate_plans.append(plan)
                self._bodies.append(body)

        # Precompute the per-circuit lookup table when feasible. The forward
        # then never touches a QNode (see _forward_lut). Registered as a buffer
        # so it follows the model to GPU and round-trips through state_dict.
        if self.n_qubits <= self._LUT_MAX_QUBITS:
            self.register_buffer("_lut", self._load_or_build_lut())
        else:
            self.register_buffer("_lut", None)

    # ------------------------------------------------------------------ helpers
    def _lut_cache_key(self) -> str:
        """Stable hash over everything the LUT values depend on.

        The table is a frozen function of the circuit configuration only (not of
        any input data), so any two layers sharing these fields produce byte-for
        -byte identical tables and can share a cache file.
        """
        payload = "|".join(
            str(v)
            for v in (
                self._LUT_BUILD_VERSION,
                self.seed,
                self.n_qubits,
                self.kernel_size,
                self.in_channels,
                self.out_channels,
                self.channel_wise,
                self.depth,
                self.decoding,
                self.n_shots,
                self.connection_prob,
                self.qdevice,
                self.qbackend,
            )
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    def _load_or_build_lut(self) -> torch.Tensor:
        """Return the LUT, loading it from disk when a matching cache exists.

        Cache hit  -> torch.load (CPU), no PennyLane calls at all.
        Cache miss -> build (batched where possible), then torch.save.
        `rebuild_lut=True` ignores any existing cache and forces a rebuild.
        """
        key = self._lut_cache_key()
        cache_path = os.path.join(self.lut_cache_dir, f"quanv_lut_{key}.pt")

        if not self.rebuild_lut and os.path.isfile(cache_path):
            try:
                lut = torch.load(cache_path, map_location="cpu")
                print(f"[Quanv2d] LUT LOADED from cache: {cache_path}")
                return lut
            except Exception as exc:  # corrupt/incompatible file -> rebuild
                print(f"[Quanv2d] LUT cache unreadable ({exc!r}); rebuilding.")

        lut = self._build_lut()
        try:
            os.makedirs(self.lut_cache_dir, exist_ok=True)
            torch.save(lut, cache_path)
            print(f"[Quanv2d] LUT BUILT and cached: {cache_path}")
        except Exception as exc:  # caching is best-effort; never fail the build
            print(f"[Quanv2d] LUT BUILT (cache write failed: {exc!r}).")
        return lut

    def _build_lut(self) -> torch.Tensor:
        """Evaluate every frozen circuit on all 2^n_qubits basis states once.

        Returns a float32 tensor of shape [out_channels, in_channels, 2^nq]
        (channel_wise) or [out_channels, 2^nq] (literal). Entry `[..., key]`
        holds the scalar that _eval_qnode_on_unique would produce for the
        bitstring whose packed integer key is `key` (bit j == (key >> j) & 1),
        so the lookup path is numerically identical to the QNode path.
        """
        nq = self.n_qubits
        n_states = 1 << nq
        # Basis states in key order: row k has bit j == (k >> j) & 1. This is
        # exactly the packing used by _eval_qnode_on_unique (powers = 1<<arange),
        # so passing these rows back returns one scalar per key, in key order.
        ks = np.arange(n_states, dtype=np.int64)
        bit_idx = np.arange(nq, dtype=np.int64)
        all_basis = ((ks[:, None] >> bit_idx[None, :]) & 1).astype(np.uint8)

        def scalars_for(qnode, body):
            # Prompt 2: one broadcast QNode call over all basis states (expval
            # only); fall back to the per-state loop if the device/decoding
            # cannot broadcast. Both paths return numerically identical values.
            if self.decoding == "expval":
                try:
                    return self._eval_circuit_batched(body, all_basis)
                except Exception:
                    pass
            return self._eval_qnode_on_unique(qnode, all_basis)

        if self.channel_wise:
            lut = torch.empty(
                (self.out_channels, self.in_channels, n_states), dtype=torch.float32
            )
            for oc in range(self.out_channels):
                for ic in range(self.in_channels):
                    scalars = scalars_for(self._qnodes[oc][ic], self._bodies[oc][ic])
                    lut[oc, ic] = torch.from_numpy(np.asarray(scalars, dtype=np.float32))
        else:
            lut = torch.empty((self.out_channels, n_states), dtype=torch.float32)
            for oc in range(self.out_channels):
                scalars = scalars_for(self._qnodes[oc], self._bodies[oc])
                lut[oc] = torch.from_numpy(np.asarray(scalars, dtype=np.float32))
        return lut

    def _eval_circuit_batched(self, body, all_basis: np.ndarray) -> np.ndarray:
        """Evaluate one circuit on ALL basis states in a single broadcast call.

        Encodes each basis state via RX(pi * bit) instead of a conditional
        PauliX: on a product basis state this differs only by a global phase
        (RX(pi)|0> = -i|1>), which leaves every <Z_i> expectation unchanged, so
        the result matches the PauliX-encoded `circuit` exactly. The RX angle is
        a broadcastable parameter, so PennyLane runs all 2^nq states at once.

        Returns the same per-state scalar as the expval branch of
        _eval_qnode_on_unique: mean over qubits of 0.5 * (1 - <Z_i>).
        """
        nq = self.n_qubits
        dev_kwargs = {"wires": nq}  # analytic (no shots) for expval
        if self.qbackend is not None:
            dev_kwargs["backend"] = self.qbackend
        dev = qml.device(self.qdevice, **dev_kwargs)

        @qml.qnode(dev, interface="numpy", diff_method=None)
        def batched(thetas):
            for q in range(nq):
                qml.RX(thetas[:, q], wires=q)
            body()
            return [qml.expval(qml.PauliZ(q)) for q in range(nq)]

        thetas = np.pi * all_basis.astype(np.float64)  # [n_states, nq]
        z = np.asarray(batched(thetas), dtype=np.float64).reshape(nq, -1)  # [nq, n_states]
        p_one = 0.5 * (1.0 - z)
        return p_one.mean(axis=0).astype(np.float32)  # [n_states]

    def _eval_qnode_on_unique(self, qnode, bitstrings_flat: np.ndarray) -> np.ndarray:
        """Evaluate `qnode` on every row of `bitstrings_flat` (shape [N, nq]),
        deduplicating identical rows. Returns array of shape [N] (scalar per
        patch — mean over qubits of either <Z> or popcount).
        """
        nq = self.n_qubits
        if bitstrings_flat.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)

        # Pack each bitstring into an integer key for deduplication.
        powers = (1 << np.arange(nq, dtype=np.int64))
        keys = bitstrings_flat.astype(np.int64) @ powers
        unique_keys, inverse = np.unique(keys, return_inverse=True)

        # Reconstruct each unique bitstring.
        unique_bits = np.zeros((unique_keys.shape[0], nq), dtype=np.int64)
        for j in range(nq):
            unique_bits[:, j] = (unique_keys >> j) & 1

        unique_scalars = np.empty((unique_keys.shape[0],), dtype=np.float32)
        if self.decoding == "expval":
            # qnode returns a list/tuple of nq scalars; collapse to mean.
            # Decoded value = mean over qubits of <Z_i>, mapped to [0, 1] via
            # (1 - <Z>) / 2 (probability of measuring |1>) so the result is a
            # non-negative pixel-like feature (matches the paper's "fraction
            # of |1> outcomes" intuition).
            for u in range(unique_bits.shape[0]):
                z_vals = qnode(unique_bits[u])
                z_arr = np.asarray(z_vals, dtype=np.float64).reshape(-1)
                p_one = 0.5 * (1.0 - z_arr)
                unique_scalars[u] = float(p_one.mean())
        else:
            # counts mode: average popcount over n_shots samples, normalized
            # by n_qubits. Faithful to Henderson's "decoded as a single scalar
            # value per output channel" via measurement statistics.
            # TODO: a per-unique-bitstring shot batching is possible here too,
            # but kept literal for now.
            for u in range(unique_bits.shape[0]):
                samples = qnode(unique_bits[u])
                samples_arr = np.asarray(samples, dtype=np.int64)
                if samples_arr.ndim == 1:
                    samples_arr = samples_arr.reshape(1, -1)
                popcount = samples_arr.sum(axis=-1).astype(np.float64) / float(nq)
                unique_scalars[u] = float(popcount.mean())

        return unique_scalars[inverse]

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Quanv2d expects [B,C,H,W], got shape {tuple(x.shape)}")
        B, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(
                f"input has {C} channels, Quanv2d was built for {self.in_channels}"
            )

        k = self.kernel_size
        s = self.stride
        p = self.padding

        # F.unfold extracts patches with the same padding/stride semantics as
        # Conv2d (so the output spatial size matches a classical conv exactly).
        patches = F.unfold(x, kernel_size=k, stride=s, padding=p)  # [B, C*k*k, L]
        L = patches.shape[-1]
        H_o = (H + 2 * p - k) // s + 1
        W_o = (W + 2 * p - k) // s + 1
        assert L == H_o * W_o, f"unfold length mismatch: {L} vs {H_o * W_o}"

        # Threshold encoding -> {0, 1} bitstrings. We deliberately .detach();
        # the quanv layer is non-trainable and gradient flow stops here (the
        # threshold itself is non-diff).
        bits = (patches.detach() > self.threshold).to(torch.uint8)

        # Fast path: precomputed lookup table -> pure tensor indexing, no QNode.
        if self._lut is not None:
            return self._forward_lut(bits, B, C, H_o, W_o, L, x)

        bits_np = bits.cpu().numpy()  # [B, C*k*k, L]

        out = torch.zeros((B, self.out_channels, H_o, W_o), dtype=x.dtype, device=x.device)

        if self.channel_wise:
            # bits_np has shape [B, C*k*k, L]; reshape to expose channel.
            bits_per_chan = bits_np.reshape(B, C, k * k, L)
            # Flatten (B, L) into one patch dimension per (oc, ic) circuit.
            for oc in range(self.out_channels):
                acc = np.zeros((B, L), dtype=np.float32)
                for ic in range(self.in_channels):
                    # Shape [B*L, k*k]: one row per patch.
                    flat = bits_per_chan[:, ic, :, :].transpose(0, 2, 1).reshape(B * L, k * k)
                    scalars = self._eval_qnode_on_unique(self._qnodes[oc][ic], flat)
                    acc += scalars.reshape(B, L)
                out_oc = torch.from_numpy(acc).to(dtype=x.dtype, device=x.device)
                out[:, oc] = out_oc.reshape(B, H_o, W_o)
        else:
            # Literal Henderson: one circuit per output channel over the full
            # C*k*k flattened patch.
            flat = bits_np.transpose(0, 2, 1).reshape(B * L, C * k * k)
            for oc in range(self.out_channels):
                scalars = self._eval_qnode_on_unique(self._qnodes[oc], flat)
                out_oc = torch.from_numpy(scalars.reshape(B, L)).to(dtype=x.dtype, device=x.device)
                out[:, oc] = out_oc.reshape(B, H_o, W_o)

        return out

    def _forward_lut(self, bits: torch.Tensor, B, C, H_o, W_o, L, x) -> torch.Tensor:
        """Lookup-table forward: identical math to the QNode path, but the
        per-patch scalar is read from self._lut instead of evaluating circuits.

        Runs ENTIRELY in torch on x.device — no .cpu()/.numpy() and no
        host<->device copy. self._lut is a registered buffer, so it already
        lives on the model's device (moves with .cuda()); `bits` was computed
        from x in forward and is likewise on x.device. The quanv block stays
        non-trainable: `bits` derives from x.detach() and the table is a
        non-learnable buffer, so no gradient flows through here.

        Mirrors the index ordering of the QNode path exactly:
          * the k*k (or C*k*k) bits of a patch are packed with powers 2^j,
            matching _eval_qnode_on_unique's `keys = bits @ (1<<arange(nq))`;
          * channel_wise sums the per-input-channel scalars, matching the `acc`
            accumulation in the QNode path.
        """
        nq = self.n_qubits
        device = x.device
        # `bits` is already on x.device; only an integer cast is needed for key
        # packing (no device move). powers/keys/gather all stay on x.device.
        bits = bits.to(torch.long)  # [B, C*k*k, L]
        powers = 2 ** torch.arange(nq, device=device, dtype=torch.long)

        if self.channel_wise:
            # [B, C, k*k, L] — same split as bits_np.reshape(B, C, k*k, L).
            bits_chan = bits.view(B, C, nq, L)
            keys = (bits_chan * powers.view(1, 1, nq, 1)).sum(dim=2)  # [B, C, L]
            acc = torch.zeros(
                (B, self.out_channels, L), dtype=self._lut.dtype, device=device
            )
            for ic in range(self.in_channels):
                lut_ic = self._lut[:, ic, :]          # [out_channels, 2^nq]
                gathered = lut_ic[:, keys[:, ic, :]]  # [out_channels, B, L]
                acc += gathered.permute(1, 0, 2)      # [B, out_channels, L]
            out = acc.view(B, self.out_channels, H_o, W_o)
        else:
            keys = (bits * powers.view(1, nq, 1)).sum(dim=1)  # [B, L]
            gathered = self._lut[:, keys]                     # [out_channels, B, L]
            out = gathered.permute(1, 0, 2).reshape(B, self.out_channels, H_o, W_o)

        return out.to(dtype=x.dtype)  # already on x.device; dtype-only cast


class QuanvConvBNReLU(nn.Module):
    """ConvBNReLU clone with the inner Conv2d swapped for Quanv2d.

    BatchNorm2d and ReLU are unchanged; only `.conv` becomes quantum. Kept as
    a standalone module (rather than monkey-patching ConvBNReLU) so the
    classical class stays untouched per the spec.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int = 1,
        *,
        threshold: float,
        depth: int,
        decoding: str,
        n_shots: int,
        channel_wise: bool,
        qdevice: str,
        qbackend: str | None,
        seed: int,
        lut_cache_dir: str | None = None,
        rebuild_lut: bool = False,
    ):
        super().__init__()
        # `groups` is part of the ConvBNReLU signature but the stem uses the
        # default groups=1 everywhere (MedViT.py:425-428). The quanv layer has
        # no analogous grouping concept; we ignore it (asserting default).
        assert groups == 1, "Quanv2d only supports groups=1 (stem default)"
        from MedViT import NORM_EPS

        self.conv = Quanv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,  # matches ConvBNReLU's hard-coded padding=1
            threshold=threshold,
            depth=depth,
            decoding=decoding,
            n_shots=n_shots,
            channel_wise=channel_wise,
            qdevice=qdevice,
            qbackend=qbackend,
            seed=seed,
            lut_cache_dir=lut_cache_dir,
            rebuild_lut=rebuild_lut,
        )
        self.norm = nn.BatchNorm2d(out_channels, eps=NORM_EPS)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class QMedViTHenderson2019(QMedViT):
    """QMedViT variant that replaces every ConvBNReLU in the stem (and only
    there) with a Henderson 2019 quanvolutional ConvBNReLU. The rest of the
    network — LTBs, ECBs, MHSA/MHCA, MLPs, BN, ReLU — is untouched.
    """

    def __init__(
        self,
        *args,
        quanv_threshold: float = 0.0,
        quanv_depth: int = 2,
        quanv_decoding: str = "expval",
        quanv_n_shots: int = 0,
        quanv_channel_wise: bool = True,
        quanv_seed: int = 0,
        quanv_lut_cache_dir: str | None = None,
        quanv_rebuild_lut: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.quanv_threshold = float(quanv_threshold)
        self.quanv_depth = int(quanv_depth)
        self.quanv_decoding = quanv_decoding
        self.quanv_n_shots = int(quanv_n_shots)
        self.quanv_channel_wise = bool(quanv_channel_wise)
        self.quanv_seed = int(quanv_seed)
        self.quanv_lut_cache_dir = quanv_lut_cache_dir
        self.quanv_rebuild_lut = bool(quanv_rebuild_lut)

        # Walk self.stem and replace every ConvBNReLU with QuanvConvBNReLU
        # matching its (C_in, C_out, kernel_size, stride).
        new_modules = []
        for idx, module in enumerate(self.stem):
            if isinstance(module, ConvBNReLU):
                # Pull conv hyperparams from the classical module.
                c_in = module.conv.in_channels
                c_out = module.conv.out_channels
                ks = module.conv.kernel_size[0]
                st = module.conv.stride[0]
                qcbr = QuanvConvBNReLU(
                    in_channels=c_in,
                    out_channels=c_out,
                    kernel_size=ks,
                    stride=st,
                    groups=module.conv.groups,
                    threshold=self.quanv_threshold,
                    depth=self.quanv_depth,
                    decoding=self.quanv_decoding,
                    n_shots=self.quanv_n_shots,
                    channel_wise=self.quanv_channel_wise,
                    qdevice=self.qdevice,
                    qbackend=self.qbackend,
                    seed=self.quanv_seed + 1000 * idx,
                    lut_cache_dir=self.quanv_lut_cache_dir,
                    rebuild_lut=self.quanv_rebuild_lut,
                )
                new_modules.append(qcbr)
            else:
                new_modules.append(module)
        self.stem = nn.Sequential(*new_modules)
