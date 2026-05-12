from quantum_variants.softmax_only import Q_E_MHSA, QLTB, QMedViT_Softmax_Only
from quantum_variants.e_mhca_only import Q_MHCA, QMedViT_E_MHCA_Only

VARIANTS = {
    "softmax": QMedViT_Softmax_Only,
    "emhca": QMedViT_E_MHCA_Only,
}

__all__ = [
    "Q_E_MHSA",
    "QLTB",
    "QMedViT_Softmax_Only",
    "Q_MHCA",
    "QMedViT_E_MHCA_Only",
    "VARIANTS",
]
