from ca_sae.sae.batch_topk import BatchTopKSAE
from ca_sae.sae.ca_sae import ClassAlignedSAE
from ca_sae.sae.ca_sae_no_mlp import ClassAlignedSAE_NO_MLP
from ca_sae.sae.core import Dictionary
from ca_sae.sae.matryoshka_batch_topk import MatryoshkaBatchTopKSAE
from ca_sae.sae.top_afa import TopAFASAE

SUPPORTED_ARCHITECTURES = {
    "batch_topk": BatchTopKSAE,
    "matryoshka": MatryoshkaBatchTopKSAE,
    "top_afa": TopAFASAE,
    "ca_sae_no_mlp": ClassAlignedSAE_NO_MLP,
    "ca_sae": ClassAlignedSAE,
}


def is_class_aligned(model: Dictionary) -> bool:
    return isinstance(model, ClassAlignedSAE) or isinstance(
        model, ClassAlignedSAE_NO_MLP
    )
