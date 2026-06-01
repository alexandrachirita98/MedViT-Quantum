"""Dump the exact QASM of a circuit built by `_build_quanv_circuit`.

Usage:
    python quantum_variants/dump_henderson_qasm.py \
        --n-qubits 4 --depth 2 --seed 0 --bits 1010

The script imports the live function from
`quantum_variants/quanv_2d_henderson_2019.py`, instantiates one circuit with
the given parameters, runs it on the chosen input bitstring (so the QNode tape
gets recorded), and prints the resulting OpenQASM 2.0.

Any edit to `_build_quanv_circuit` will be picked up here automatically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pennylane as qml

from quantum_variants.quanv_2d_henderson_2019 import _build_quanv_circuit


def parse_bits(s: str, n_qubits: int) -> np.ndarray:
    if len(s) != n_qubits:
        raise ValueError(
            f"--bits must have exactly {n_qubits} characters, got {len(s)}"
        )
    if any(c not in "01" for c in s):
        raise ValueError(f"--bits must be a binary string, got {s!r}")
    return np.array([int(c) for c in s], dtype=np.int64)


def dump_qasm(n_qubits: int, depth: int, seed: int, bits_str: str) -> str:
    qnode, _plan = _build_quanv_circuit(
        n_qubits=n_qubits,
        depth=depth,
        qdevice="default.qubit",
        qbackend=None,
        seed=seed,
        decoding="expval",
        n_shots=0,
    )

    bits = parse_bits(bits_str, n_qubits)

    # Run once to materialize the tape (PennyLane records gates on call).
    qnode(bits)
    tape = qnode.tape

    # PennyLane >= 0.30 exposes `to_openqasm` on the tape; fall back to the
    # `qml.workflow` helper for older versions.
    if hasattr(tape, "to_openqasm"):
        qasm = tape.to_openqasm(wires=list(range(n_qubits)), measure_all=True)
    else:
        qasm = qml.transforms.to_openqasm(qnode)(bits)

    return qasm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--bits",
        type=str,
        default=None,
        help="Binary string of length n_qubits used as the threshold-encoded "
        "input. Defaults to all-zeros (no X gates in the encoding phase).",
    )
    args = parser.parse_args()

    bits_str = args.bits if args.bits is not None else "0" * args.n_qubits

    print(dump_qasm(args.n_qubits, args.depth, args.seed, bits_str))


if __name__ == "__main__":
    main()
