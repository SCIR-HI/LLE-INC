"""
The :mod:`sklearn.manifold` module implements data embedding techniques.
"""

from ._locally_linear import locally_linear_embedding, LocallyLinearEmbedding
from ._locally_linear_mod import locally_linear_embedding_mod, LocallyLinearEmbeddingMod
from ._isomap import Isomap
from ._mds import MDS, smacof
from ._spectral_embedding import SpectralEmbedding, spectral_embedding
from ._t_sne import TSNE, trustworthiness

__all__ = [
    "locally_linear_embedding",
    "LocallyLinearEmbedding",
    "locally_linear_embedding_mod",
    "LocallyLinearEmbeddingMod",
    "Isomap",
    "MDS",
    "smacof",
    "SpectralEmbedding",
    "spectral_embedding",
    "TSNE",
    "trustworthiness",
]
