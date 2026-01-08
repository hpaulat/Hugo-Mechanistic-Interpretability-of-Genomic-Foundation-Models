from __future__ import annotations

from dataclasses import dataclass   # way to store "bundle" of related objects
from functools import lru_cache     # cache loaded models to avoid re-loading, calling twice just returns cached version
from typing import Iterable, Literal    # type hints for function arguments

import numpy as np   
import haiku as hk      # neural network library
import jax      # PRNG keys and runtime backend
import jax.numpy as jnp     # JAX's Numpy-like array
from nucleotide_transformer.pretrained import get_pretrained_model      # function to load pre-trained NTV2 models

Array = jnp.ndarray

# loads ntv2 model once, turns DNA strings into token IDs, runs model and returns mebeddings, avoids re-downloading repeatedly

@dataclass(frozen=True) # container class to hold model components
class NTV2Bundle:
    params: hk.Params   # weights
    forward: hk.Transformed # transformed function
    tokenizer: object   # tokenizer
    config: object  # metadata
    layer: int  # which layer embeddings you asked to save
    max_positions: int  # max sequence length


    
@lru_cache(maxsize=4)   # just remember last 4 distinct combinations of arguments
def load_ntv2(
    model_name: str = "250M_multi_species_v2",
    layer: int = 20,
    max_positions: int = 512,
) -> NTV2Bundle:
    """
    Loads Nucleotide Transformer v2 (JAX/Haiku) and returns a reusable bundle.
    This will download weights on first run and cache them on disk automatically.
    """
    params, forward_fn, tokenizer, config = get_pretrained_model(
        model_name=model_name,
        embeddings_layers_to_save=(layer,),
        max_positions=max_positions,
    )
    forward = hk.transform(forward_fn)
    return NTV2Bundle(
        params=params,
        forward=forward,
        tokenizer=tokenizer,
        config=config,
        layer=layer,
        max_positions=max_positions,
    )

def tokenize(
    tokenizer: object,
    sequences: Iterable[str],
    dtype: jnp.dtype = jnp.int32,
) -> Array:
    """
    Returns token IDs as a JAX array of shape (B, L).
    """
    token_ids = [b[1] for b in tokenizer.batch_tokenize(list(sequences))]
    return jnp.asarray(token_ids, dtype=dtype)

def embed(
    sequences: Iterable[str],
    *,
    model_name: str = "250M_multi_species_v2",
    layer: int = 20,
    max_positions: int = 512,
    pooling: Literal["none", "mean"] = "none",
    seed: int = 0,
    return_numpy: bool = True,
) -> np.ndarray | Array:
    """
    Computes embeddings for a batch of sequences.

    Returns:
      - pooling="none": (B, L, D)
      - pooling="mean": (B, D)  (mean over tokens)
    """
    bundle = load_ntv2(model_name=model_name, layer=layer, max_positions=max_positions)
    tokens = tokenize(bundle.tokenizer, sequences)

    outs = bundle.forward.apply(bundle.params, jax.random.PRNGKey(seed), tokens)
    key = f"embeddings_{bundle.layer}"
    emb: Array = outs[key]  # (B, L, D)

    if pooling == "mean":
        # Simple mean over sequence length dimension
        emb = emb.mean(axis=1)

    if return_numpy:
        return np.asarray(emb)
    return emb