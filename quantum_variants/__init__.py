from quantum_variants.softmax_only import Q_E_MHSA, QLTB, QMedViT_Softmax_Only
from quantum_variants.quanv_2d_henderson_2019 import QMedViTHenderson2019
from quantum_variants.quanv_2d_hur_2022 import QMedViTHur2022

VARIANTS = {
    "softmax": QMedViT_Softmax_Only,
    "henderson2019": QMedViTHenderson2019,
    "hur2022": QMedViTHur2022,
}

__all__ = [
    "Q_E_MHSA",
    "QLTB",
    "QMedViT_Softmax_Only",
    "QMedViTHenderson2019",
    "QMedViTHur2022",
    "VARIANTS",
]
