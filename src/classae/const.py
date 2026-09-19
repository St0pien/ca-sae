from classae.sae.batch_topk import BatchTopKSAE
from classae.sae.classae import ClasSAE
from classae.sae.classae_no_mlp import ClasSAE_NO_MLP
from classae.sae.classae_no_topk import ClasSAE_NO_TOPK
from classae.sae.core import Dictionary
from classae.sae.matryoshka_batch_topk import MatryoshkaBatchTopKSAE
from classae.sae.top_afa import TopAFASAE

SUPPORTED_ARCHITECTURES = {
    "batch_topk": BatchTopKSAE,
    "matryoshka": MatryoshkaBatchTopKSAE,
    "top_afa": TopAFASAE,
    "classae_no_mlp": ClasSAE_NO_MLP,
    "classae_no_topk": ClasSAE_NO_TOPK,
    "classae": ClasSAE,
}


def is_class_aligned(model: Dictionary) -> bool:
    return (
        isinstance(model, ClasSAE)
        or isinstance(model, ClasSAE_NO_MLP)
        or isinstance(model, ClasSAE_NO_TOPK)
    )
