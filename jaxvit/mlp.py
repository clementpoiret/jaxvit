from typing import Callable, Optional, Union

import flax.linen as nn
import jax
from jax.tree_util import Partial

from jaxvit.layers import Attention, DropPath, Identity, LayerScale
from jaxvit.mlp import Mlp

KeyArray = Union[jax.Array, jax.random.KeyArray]


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    in_features: int
    hidden_features: Optional[int] = None
    out_features: Optional[int] = None
    act_layer: Callable = nn.activation.gelu
    norm_layer: Optional[Callable] = None
    bias: bool = True
    drop: float = 0.
    use_conv: bool = False

    def setup(self):
        out_features = self.out_features or self.in_features
        hidden_features = self.hidden_features or self.in_features
        self.linear_layer = Partial(
            nn.Conv2d, kernel_size=1) if self.use_conv else nn.Dense

        self.fc1 = self.linear_layer(hidden_features, bias=self.bias)
        self.act = self.act_layer()
        self.drop1 = nn.Dropout(self.drop)
        self.norm = self.norm_layer() if self.norm_layer else Identity()
        self.fc2 = self.linear_layer(out_features, bias=self.bias)
        self.drop2 = nn.Dropout(self.drop)

    def __call__(self,
                 inputs,
                 deterministic: Optional[bool] = None,
                 rng: KeyArray = None):
        # Split rng for drop1 and drop2
        rng1, rng2 = jax.random.split(rng, 2)

        x = self.fc1(inputs)
        x = self.act(x)
        x = self.drop1(x, deterministic=deterministic, rng=rng1)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x, deterministic=deterministic, rng=rng2)

        return x


class Block(nn.Module):
    """ Transformer block

    Args:
        dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: Ratio of mlp hidden dim to embedding dim
        qkv_bias: If True, add a learnable bias to query, key, value
        qk_norm: If True, normalize the query and key
        proj_drop: Dropout, probability to drop units of the projection
        attn_drop: Dropout, probability to drop units of the attention
        init_values: If set, use LayerScale
        drop_path: Stochastic depth rate
        act_layer: Activation layer
        norm_layer: Normalization layer
        mlp_layer: MLP layer
    """

    dim: int
    num_heads: int
    mlp_ratio: float = 4.
    qkv_bias: bool = False
    qk_norm: bool = False
    proj_drop: float = 0.
    attn_drop: float = 0.
    init_values: Optional[float] = None
    drop_path: float = 0.
    act_layer: Callable = nn.activation.gelu
    norm_layer: Callable = nn.LayerNorm
    mlp_layer: Callable = Mlp

    def setup(self):
        self.norm1 = self.norm_layer()
        self.attn = Attention(dim=self.dim,
                              num_heads=self.num_heads,
                              qkv_bias=self.qkv_bias,
                              qk_norm=self.qk_norm,
                              proj_drop=self.proj_drop,
                              attn_drop=self.attn_drop,
                              norm_layer=self.norm_layer)
        self.ls1 = LayerScale(
            self.dim,
            init_values=self.init_values) if self.init_values else Identity()
        self.drop_path1 = DropPath(
            self.drop_path) if self.drop_path > 0. else Identity()

        self.norm2 = self.norm_layer()
        self.mlp = self.mlp_layer(in_features=self.dim,
                                  hidden_features=int(self.dim *
                                                      self.mlp_ratio),
                                  act_layer=self.act_layer,
                                  drop=self.proj_drop)
        self.ls2 = LayerScale(
            self.dim,
            init_values=self.init_values) if self.init_values else Identity()
        self.drop_path2 = DropPath(
            self.drop_path) if self.drop_path > 0. else Identity()

    def __call__(self,
                 inputs,
                 deterministic: Optional[bool] = None,
                 rng: KeyArray = None):
        # Split rng for drop_path1 and drop_path2
        rng1, rng2 = jax.random.split(rng, 2)

        x = inputs + self.drop_path1(self.ls1(self.attn(self.norm1(inputs))),
                                     deterministic=deterministic,
                                     rng=rng1)
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))),
                                deterministic=deterministic,
                                rng=rng2)

        return x
