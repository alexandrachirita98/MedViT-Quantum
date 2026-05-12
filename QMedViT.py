"""
QMedViT: extensible base for quantum-augmented MedViT variants.

The base class is a no-op; subclasses (in `quantum_variants/`) override two
hooks to decide which blocks to touch and what to swap in.

The shared quantum kernel (RY angle encoding + StronglyEntanglingLayers +
PauliZ(0) expval) lives here as `_build_qkernel` and is reused by every variant.
"""

import pennylane as qml
import torch
from timm.models.registry import register_model
from torch import nn

from MedViT import MedViT


def _build_qkernel(n_qubits: int, q_depth: int, qdevice: str, qbackend: str | None):
    dev_kwargs = {"wires": 2 * n_qubits}
    if qbackend is not None:
        dev_kwargs["backend"] = qbackend
    dev = qml.device(qdevice, **dev_kwargs)

    weight_shape = (q_depth, 2 * n_qubits, 3)

    @qml.qnode(dev, interface="torch", diff_method="best")
    def circuit(q_angles, k_angles, weights):
        for i in range(n_qubits):
            qml.RY(q_angles[i], wires=i)
        for i in range(n_qubits):
            qml.RY(k_angles[i], wires=n_qubits + i)
        qml.templates.StronglyEntanglingLayers(weights, wires=range(2 * n_qubits))
        return qml.expval(qml.PauliZ(0))

    return circuit, weight_shape


class QMedViT(MedViT):
    def __init__(
        self,
        stem_chs,
        depths,
        path_dropout,
        attn_drop=0,
        drop=0,
        num_classes=1000,
        strides=(1, 2, 2, 2),
        sr_ratios=(8, 4, 2, 1),
        head_dim=32,
        mix_block_ratio=0.75,
        use_checkpoint=False,
        n_qubits=3,
        q_depth=3,
        qdevice="default.qubit",
        qbackend=None,
        quantum_stages=(3,),
        quantum_block_indices=None,
        qpu_mode=False,
        qpu_shots=5000,
    ):
        super().__init__(
            stem_chs=stem_chs,
            depths=depths,
            path_dropout=path_dropout,
            attn_drop=attn_drop,
            drop=drop,
            num_classes=num_classes,
            strides=list(strides),
            sr_ratios=list(sr_ratios),
            head_dim=head_dim,
            mix_block_ratio=mix_block_ratio,
            use_checkpoint=use_checkpoint,
        )

        self.depths = depths
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.head_dim = head_dim
        self.mix_block_ratio = mix_block_ratio
        self.attn_drop = attn_drop
        self.drop = drop
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.qdevice = qdevice
        self.qbackend = qbackend
        self.quantum_stages = tuple(quantum_stages)
        # Per-stage within-stage block indices to quantize. None = all blocks
        # in `quantum_stages`. Use e.g. (2,) to keep only the last LTB quantum.
        self.quantum_block_indices = (
            None if quantum_block_indices is None else tuple(quantum_block_indices)
        )
        # QPU-style execution: per-sample, finite shots, no U-extraction
        # shortcut. Set qpu_mode=True for inference on real quantum hardware
        # (or a shots-based simulator). Default False = analytic simulator path.
        self.qpu_mode = qpu_mode
        self.qpu_shots = qpu_shots
        self.path_dropout = path_dropout

        stage_of_block = []
        for stage_id, d in enumerate(depths):
            stage_of_block.extend([stage_id] * d)
        dpr = [x.item() for x in torch.linspace(0, path_dropout, sum(depths))]

        for idx, block in enumerate(self.features):
            stage_id = stage_of_block[idx]
            ctx = {"dpr": dpr[idx], "sr_ratio": sr_ratios[stage_id]}
            if not self._should_quantize_block(block, stage_id, idx):
                continue
            self.features[idx] = self._build_quantum_block(block, stage_id, idx, **ctx)

    def _should_quantize_block(self, block, stage_id, block_idx) -> bool:
        return False

    def _build_quantum_block(self, block, stage_id, block_idx, **ctx) -> nn.Module:
        return block

    def load_classical_pretrained(self, state_dict, strict=False):
        return self.load_state_dict(state_dict, strict=strict)


def _resolve_variant(variant):
    from quantum_variants import VARIANTS

    if isinstance(variant, str):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant '{variant}', expected one of {list(VARIANTS)}")
        return VARIANTS[variant]
    return variant


@register_model
def QMedViT_small(pretrained=False, pretrained_cfg=None, variant="softmax", **kwargs):
    cls = _resolve_variant(variant)
    return cls(stem_chs=[64, 32, 64], depths=[3, 4, 10, 3], path_dropout=0.1, **kwargs)


@register_model
def QMedViT_base(pretrained=False, pretrained_cfg=None, variant="softmax", **kwargs):
    cls = _resolve_variant(variant)
    return cls(stem_chs=[64, 32, 64], depths=[3, 4, 20, 3], path_dropout=0.2, **kwargs)


@register_model
def QMedViT_large(pretrained=False, pretrained_cfg=None, variant="softmax", **kwargs):
    cls = _resolve_variant(variant)
    return cls(stem_chs=[64, 32, 64], depths=[3, 4, 30, 3], path_dropout=0.2, **kwargs)
