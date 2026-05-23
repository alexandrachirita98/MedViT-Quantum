from quantum_variants.softmax_only import Q_E_MHSA, QLTB, QMedViT_Softmax_Only
from quantum_variants.quanv_stem import QuanvStem, QMedViT_Quanv_Stem

VARIANTS = {
    "softmax": QMedViT_Softmax_Only,
    "quanv_stem": QMedViT_Quanv_Stem,
}

__all__ = [
    "Q_E_MHSA",
    "QLTB",
    "QMedViT_Softmax_Only",
    "QuanvStem",
    "QMedViT_Quanv_Stem",
    "VARIANTS",
]
