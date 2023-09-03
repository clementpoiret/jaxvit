from typing import Callable, Optional, Union

import flax.linen as nn
import jax
from jax.tree_util import Partial

from jaxvit.layers import Identity

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
        self.norm = self.norm_layer(
            hidden_features) if self.norm_layer else Identity()
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
