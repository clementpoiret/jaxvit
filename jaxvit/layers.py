from typing import Callable, Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
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


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    embed_dim: int = 768
    norm_layer: Optional[Callable] = None
    flatten: bool = True
    bias: bool = True
    dynamic_img_pad: bool = False

    def setup(self):
        self.patch_size = (self.patch_size, self.patch_size)
        self.grid_size = tuple(
            [s // p for s, p in zip(self.img_size, self.patch_size)])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv(features=self.embed_dim,
                            kernel_size=self.patch_size,
                            strides=self.patch_size,
                            padding="VALID",
                            use_bias=self.bias,
                            name="patch_embed")
        self.norm = self.norm_layer(
            self.embed_dim) if self.norm_layer else Identity()

    def __call__(self, x):
        B, H, W, C = x.shape
        x = self.proj(x)  # B Ph Pw C
        if self.flatten:
            x = jnp.reshape(x, [B, -1, C])  # BLC

        x = self.norm(x)

        return x


def drop_path(x,
              drop_prob: float = 0.,
              deterministic: Optional[bool] = False,
              scale_by_keep: bool = True,
              rng: Optional[KeyArray] = None):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or deterministic:
        return x

    keep_prob = 1 - drop_prob

    random_tensor = jax.random.bernoulli(rng, p=keep_prob, shape=x.shape)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor = random_tensor / keep_prob

    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    drop_prob: float = 0.
    deterministic: Optional[bool] = False
    scale_by_keep: bool = True
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
        assert 0 <= self.drop_prob < 1.0

        if rng is None:
            rng = self.make_rng(self.rng_collection)

        deterministic = nn.merge_param("deterministic", self.deterministic,
                                       deterministic)

        if self.drop_prob == 0.0 or self.deterministic:
            return inputs

        return drop_path(inputs, self.drop_prob, self.deterministic,
                         self.scale_by_keep, rng)
