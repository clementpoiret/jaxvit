from typing import Optional, Union

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import lax

KeyArray = Union[jax.Array, jax.random.KeyArray]


class Identity(nn.Module):
    """Identity layer."""

    @nn.compact
    def __call__(self, x):
        return x


class PatchDropout(nn.Module):
    """Create a Patch Dropout layer.

    https://arxiv.org/abs/2212.00794

    Note: When using :meth:`Module.apply() <flax.linen.Module.apply>`, make sure
        to include an RNG seed named `'dropout'`. For example::

        model.apply({'params': params}, inputs=inputs, train=True, rngs={'dropout':
        dropout_rng})

    Attributes:
        rate: the dropout rate (_not_ the keep rate!)
        num_prefix_tokens: number of tokens to keep at the beginning of the
            sequence
        deterministic: if `True`, disables dropout
        ordered: if `True`, kept indices are ordered
        return_indices: if `True`, returns the indices of the kept tokens
        rng_collection: the rng collection name to use when requesting an rng key.
    """

    rate: float
    num_prefix_tokens: int = 1
    deterministic: Optional[bool] = None
    ordered: Optional[bool] = None
    return_indices: Optional[bool] = None
    rng_collection: str = "dropout"

    @nn.compact
    def __call__(self,
                 inputs,
                 deterministic: Optional[bool] = None,
                 rng: Optional[KeyArray] = None):
        """Applies dropout on patch tokens.

        Args:
            inputs: the inputs that should be randomly dropped out.
            deterministic: if `True`, disables dropout.
            rng: an optional PRNGKey used as the random key, if not specified, one
            will be generated using ``make_rng`` with the ``rng_collection`` name.

        Returns:
            The inputs with dropout applied.
        """
        assert 0 <= self.rate < 1.0

        if rng is None:
            rng = self.make_rng(self.rng_collection)

        deterministic = nn.merge_param("deterministic", self.deterministic,
                                       deterministic)

        if self.rate == 0.0 or self.deterministic:
            if self.return_indices:
                return inputs, None
            return inputs

        # rng1, rng2 = jax.random.split(rng, 2)

        if self.num_prefix_tokens > 0:
            prefix, tokens = jnp.split(inputs, [self.num_prefix_tokens], axis=1)
        else:
            prefix, tokens = None, inputs

        num_keep = max(1, int(tokens.shape[1] * (1.0 - self.rate)))

        indices = jnp.arange(tokens.shape[1])
        keep_indices = jax.random.permutation(rng, indices)[:num_keep]

        if self.ordered:
            # NOTE: does not need to maintain patch order in typical transformer use,
            # but possibly useful for debug / visualization
            keep_indices = jnp.sort(keep_indices)

        kept_tokens = jnp.take(tokens, keep_indices, axis=1)

        if prefix is not None:
            kept_tokens = jnp.concatenate([prefix, kept_tokens], axis=1)

        if self.return_indices:
            return kept_tokens, keep_indices

        return kept_tokens
