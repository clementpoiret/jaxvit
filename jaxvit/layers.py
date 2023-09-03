from typing import Callable, Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp

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

    Attributes:
        drop_prob: the dropout probability (_not_ the keep rate!)
        deterministic: if `True`, disables dropout
        scale_by_keep: if `True`, scales the output by the keep rate
        rng_collection: the rng collection name to use when requesting an rng key.
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


class LayerScale(nn.Module):
    """Layer scale for feed forward and attention layers.

    Attributes:
        dim: the dimension of the input
        init_values: the initial value of the scaling factor
    """
    dim: int
    init_values: float = 1e-5

    @nn.compact
    def __call__(self, x):
        """Applies the layer scale.

        Args:
            x: the input tensor

        Returns:
            The scaled tensor
        """
        gamma = self.param("gamma", lambda k, s: jnp.ones(s) * self.init_values,
                           (self.dim,))
        return x * gamma


class Attention(nn.Module):
    """Multi-head self-attention with relative positional encoding.

    Attributes:
        dim: the dimension of the input
        num_heads: the number of attention heads
        qkv_bias: if `True`, add a learnable bias to q, k, v
        qk_norm: if `True`, normalize the query and key
        attn_drop: the dropout rate for attention weights
        proj_drop: the dropout rate for the outputs
        norm_layer: the normalization layer to apply to the output
    """
    dim: int
    fused_attn: bool = False
    num_heads: int = 8
    qkv_bias: bool = False
    qk_norm: bool = False
    attn_drop: float = 0.
    proj_drop: float = 0.
    norm_layer: Optional[Callable] = nn.LayerNorm

    def setup(self):
        assert dim % num_heads == 0, "dimension must be divisible by number of heads"
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim**-0.5
        # self.fused_attn = use_fused_attn()

        self.qkv = nn.Dense(features=3 * self.dim,
                            use_bias=self.qkv_bias,
                            name="qkv")
        self.q_norm = self.norm_layer(self.dim) if self.qk_norm else Identity()
        self.k_norm = self.norm_layer(self.dim) if self.qk_norm else Identity()
        self.attn_drop = nn.Dropout(rate=self.attn_drop)
        self.proj = nn.Dense(features=self.dim, name="proj")
        self.proj_drop = nn.Dropout(rate=self.proj_drop)

    def __call__(self,
                 x,
                 deterministic: Optional[bool] = None,
                 rng: Optional[KeyArray] = None):
        """ Applies multi-head self-attention.

        Args:
            x: the input tensor
            deterministic: if `True`, disables dropout
            rng: an optional PRNGKey used as the random key, if not specified, one
                will be generated using ``make_rng`` with the ``rng_collection`` name.

        Returns:
            The output of the self-attention layer.
        """
        B, N, C = x.shape

        # JAX code:
        qkv = self.qkv(x)
        qkv = jnp.reshape(qkv, [B, N, 3, self.num_heads, self.head_dim])
        qkv = jnp.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            raise NotImplementedError("Fused attention not implemented yet.")
        else:
            q = q * self.scale
            attn = q @ jnp.transpose(k, [0, 1, 3, 2])
            attn = nn.activation.softmax(attn, axis=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = jnp.transpose(x, [0, 2, 3, 1])
        x = jnp.reshape(x, [B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
