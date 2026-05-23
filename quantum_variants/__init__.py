from quantum_variants.softmax_only import Q_E_MHSA, QLTB, QMedViT_Softmax_Only
from quantum_variants.e_mhca_only import Q_MHCA, QMedViT_E_MHCA_Only
from quantum_variants.quanv_stem import QuanvStem, QMedViT_Quanv_Stem

VARIANTS = {
    "softmax": QMedViT_Softmax_Only,
    "emhca": QMedViT_E_MHCA_Only,
    "quanv_stem": QMedViT_Quanv_Stem,
}

__all__ = [
    "Q_E_MHSA",
    "QLTB",
    "QMedViT_Softmax_Only",
    "Q_MHCA",
    "QMedViT_E_MHCA_Only",
    "QuanvStem",
    "QMedViT_Quanv_Stem",
    "VARIANTS",
]
