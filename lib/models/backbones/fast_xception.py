import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jnd
import haiku as hk
from functools import partial

from .. import nnutils as nn


class FastXception(hk.Module):
    """Xception backbone like the one used in CALFIN"""
    def __call__(self, x):
        B, H, W, C = x.shape

        # Backbone
        x, skip1 = XceptionBlock([64, 64, 64], stride=2, return_skip=True)(x)
        x, skip2 = XceptionBlock([128, 128, 128], stride=2, return_skip=True)(x)
        x, skip3 = XceptionBlock([256, 256, 512], stride=2, return_skip=True)(x)
        for i in range(8):
            x = XceptionBlock([512, 512, 512], skip_type='sum', stride=1)(x)

        x = XceptionBlock([512, 512, 512], stride=2)(x)
        x = XceptionBlock([512, 512, 512], stride=1, rate=(1, 2, 4))(x)

        # ASPP
        # Image Feature branch
        bD = hk.max_pool(x, window_shape=2, strides=2, padding='SAME')
        bD = ConvLNAct(64, 1)(bD)
        bD = nn.upsample(bD, factor=2)

        b0 = ConvLNAct(64, 1)(x)
        b1 = ParallelSepConv(64, rate=1)(x)
        b2 = ParallelSepConv(64, rate=2)(x)
        b3 = ParallelSepConv(64, rate=3)(x)
        b4 = ParallelSepConv(64, rate=4)(x)
        b5 = ParallelSepConv(64, rate=5)(x)
        x = jnp.concatenate([bD, b0, b1, b2, b3, b4, b5], axis=-1)

        x = ConvLNAct(512, 1)(x)
        skip3 = ConvLNAct(64, 1)(skip3)

        return [skip3, x]


class XceptionBlock(hk.Module):
    def __init__(self, depth_list, stride, skip_type='conv',
                 rate=[1, 1, 1], return_skip=False):
        super().__init__()
        self.blocks = []
        for i in range(3):
            self.blocks.append(ParallelSepConv(
                depth_list[i],
                stride=stride if i == 2 else 1,
                rate=rate[i],
            ))

        self.ln = hk.LayerNorm(axis=-1, param_axis=-1,
                        create_scale=True, create_offset=True)

        if skip_type == 'conv':
            self.shortcut = hk.Conv2D(depth_list[-1], 1, stride=stride)
        elif skip_type == 'sum':
            self.shortcut = nn.identity
        self.return_skip = return_skip


    def __call__(self, inputs):
        residual = inputs
        for i, block in enumerate(self.blocks):
            residual = block(residual)
            if i == 1:
                skip = residual

        shortcut = self.shortcut(inputs)
        outputs = jax.nn.relu(self.ln(residual + shortcut))

        if self.return_skip:
            return outputs, skip
        else:
            return outputs


class ConvLNAct(hk.Module):
    def __init__(self, *args, ln=True, act='relu', **kwargs):
        super().__init__()
        kwargs['with_bias'] = False
        self.conv = hk.Conv2D(*args, **kwargs)

        if ln:
            self.ln = hk.LayerNorm(axis=-1, param_axis=-1,
                        create_scale=True, create_offset=True)
        else:
            self.ln = nn.identity

        if act is None:
            self.act = nn.identity
        elif hasattr(jax.nn, act):
            self.act = getattr(jax.nn, act)
        else:
            raise ValueError(f"no activation called {act}")

    def __call__(self, x):
        x = self.conv(x)
        x = self.ln(x)
        x = self.act(x)
        return x


class ParallelSepConv(hk.Module):
    def __init__(self, filters, stride=1, kernel_size=3, rate=1):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.rate = rate
        self.filters = filters

    def __call__(self, x):
        x1 = hk.Conv2D(self.filters, 1, stride=self.stride, with_bias=False)(x)

        groups = min(x.shape[-1], self.filters)
        x2 = hk.Conv2D(self.filters, self.kernel_size, stride=self.stride,
                       rate=self.rate, feature_group_count=groups)(x)

        return jax.nn.relu(x1 + x2)
