from ca_sae.sae.batch_top_k import BatchTopKSAE
from ca_sae.sae.ca_sae import ClassAlignedSAE

SUPPORTED_ARCHITECTURES = {
    "batch_topk": BatchTopKSAE,
    "ca_sae": ClassAlignedSAE,
}
