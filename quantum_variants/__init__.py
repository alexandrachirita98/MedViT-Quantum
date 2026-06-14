from quantum_variants.softmax_only import Q_E_MHSA, QLTB, QMedViT_Softmax_Only
from quantum_variants.qpa_scoring import (
    Q_E_MHSA_Scoring,
    QLTB_Scoring,
    QMedViT_QPA_Scoring,
)
from quantum_variants.qpa_tuned import (
    Q_E_MHSA_Tuned,
    QLTB_Tuned,
    QMedViT_QPA_Tuned,
)
from quantum_variants.quanv_2d_henderson_2019 import QMedViTHenderson2019
from quantum_variants.quanv_2d_hur_2022 import QMedViTHur2022

VARIANTS = {
    "softmax": QMedViT_Softmax_Only,
    "qpa": QMedViT_QPA_Scoring,
    "qpa_tuned": QMedViT_QPA_Tuned,
    "henderson2019": QMedViTHenderson2019,
    "hur2022": QMedViTHur2022,
}

__all__ = [
    "Q_E_MHSA",
    "QLTB",
    "QMedViT_Softmax_Only",
    "Q_E_MHSA_Scoring",
    "QLTB_Scoring",
    "QMedViT_QPA_Scoring",
    "Q_E_MHSA_Tuned",
    "QLTB_Tuned",
    "QMedViT_QPA_Tuned",
    "QMedViTHenderson2019",
    "QMedViTHur2022",
    "VARIANTS",
]
