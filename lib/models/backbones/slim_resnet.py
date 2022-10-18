# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Resnet Backbones -> Slight adaptation by removing the final Pooling and FC layer"""

import types
from typing import Mapping, Optional, Sequence, Union, Any

import jax
import jax.numpy as jnp
import haiku as hk
FloatStrOrBool = Union[str, float, bool]


def LayerNorm(**kwargs):
    return hk.LayerNorm(axis=-1, param_axis=-1,
        create_scale=True, create_offset=True, **kwargs)



class BlockV1(hk.Module):
  """SlimResNet V1 block with optional bottleneck."""

  def __init__(
      self,
      channels: int,
      stride: Union[int, Sequence[int]],
      use_projection: bool,
      bn_config: Mapping[str, FloatStrOrBool],
      bottleneck: bool,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.use_projection = use_projection

    bn_config = dict(bn_config)
    bn_config.setdefault("create_scale", True)
    bn_config.setdefault("create_offset", True)
    bn_config.setdefault("decay_rate", 0.999)

    if self.use_projection:
      self.proj_conv = hk.Conv2D(
          output_channels=channels,
          kernel_shape=1,
          stride=stride,
          with_bias=False,
          padding="SAME",
          name="shortcut_conv")

      self.proj_layernorm = LayerNorm(name="shortcut_layernorm")

    channel_div = 4 if bottleneck else 1
    conv_0 = hk.Conv2D(
        output_channels=channels // channel_div,
        kernel_shape=1 if bottleneck else 3,
        stride=1 if bottleneck else stride,
        with_bias=False,
        padding="SAME",
        name="conv_0")
    bn_0 = LayerNorm(name="layernorm_0")

    conv_1 = hk.Conv2D(
        output_channels=channels // channel_div,
        kernel_shape=3,
        stride=stride if bottleneck else 1,
        with_bias=False,
        padding="SAME",
        name="conv_1")

    bn_1 = LayerNorm(name="layernorm_1")
    layers = ((conv_0, bn_0), (conv_1, bn_1))

    if bottleneck:
      conv_2 = hk.Conv2D(
          output_channels=channels,
          kernel_shape=1,
          stride=1,
          with_bias=False,
          padding="SAME",
          name="conv_2")

      bn_2 = LayerNorm(name="layernorm_2", scale_init=jnp.zeros)
      layers = layers + ((conv_2, bn_2),)

    self.layers = layers

  def __call__(self, inputs):
    out = shortcut = inputs

    if self.use_projection:
      shortcut = self.proj_conv(shortcut)
      shortcut = self.proj_layernorm(shortcut)

    for i, (conv_i, bn_i) in enumerate(self.layers):
      out = conv_i(out)
      out = bn_i(out)
      if i < len(self.layers) - 1:  # Don't apply relu on last layer
        out = jax.nn.relu(out)

    return jax.nn.relu(out + shortcut)


class BlockV2(hk.Module):
  """SlimResNet V2 block with optional bottleneck."""

  def __init__(
      self,
      channels: int,
      stride: Union[int, Sequence[int]],
      use_projection: bool,
      bn_config: Mapping[str, FloatStrOrBool],
      bottleneck: bool,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.use_projection = use_projection

    bn_config = dict(bn_config)
    bn_config.setdefault("create_scale", True)
    bn_config.setdefault("create_offset", True)

    if self.use_projection:
      self.proj_conv = hk.Conv2D(
          output_channels=channels,
          kernel_shape=1,
          stride=stride,
          with_bias=False,
          padding="SAME",
          name="shortcut_conv")

    channel_div = 4 if bottleneck else 1
    conv_0 = hk.Conv2D(
        output_channels=channels // channel_div,
        kernel_shape=1 if bottleneck else 3,
        stride=1 if bottleneck else stride,
        with_bias=False,
        padding="SAME",
        name="conv_0")

    bn_0 = LayerNorm(name="layernorm_0")

    conv_1 = hk.Conv2D(
        output_channels=channels // channel_div,
        kernel_shape=3,
        stride=stride if bottleneck else 1,
        with_bias=False,
        padding="SAME",
        name="conv_1")

    bn_1 = LayerNorm(name="layernorm_1")
    layers = ((conv_0, bn_0), (conv_1, bn_1))

    if bottleneck:
      conv_2 = hk.Conv2D(
          output_channels=channels,
          kernel_shape=1,
          stride=1,
          with_bias=False,
          padding="SAME",
          name="conv_2")

      # NOTE: Some implementations of SlimResNet50 v2 suggest initializing
      # gamma/scale here to zeros.
      bn_2 = LayerNorm(name="layernorm_2")
      layers = layers + ((conv_2, bn_2),)

    self.layers = layers

  def __call__(self, inputs):
    x = shortcut = inputs

    for i, (conv_i, bn_i) in enumerate(self.layers):
      x = bn_i(x)
      x = jax.nn.relu(x)
      if i == 0 and self.use_projection:
        shortcut = self.proj_conv(x)
      x = conv_i(x)

    return x + shortcut


class BlockGroup(hk.Module):
  """Higher level block for SlimResNet implementation."""

  def __init__(
      self,
      channels: int,
      num_blocks: int,
      stride: Union[int, Sequence[int]],
      bn_config: Mapping[str, FloatStrOrBool],
      resnet_v2: bool,
      bottleneck: bool,
      use_projection: bool,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)

    block_cls = BlockV2 if resnet_v2 else BlockV1

    self.blocks = []
    for i in range(num_blocks):
      self.blocks.append(
          block_cls(channels=channels,
                    stride=(1 if i else stride),
                    use_projection=(i == 0 and use_projection),
                    bottleneck=bottleneck,
                    bn_config=bn_config,
                    name="block_%d" % (i)))

  def __call__(self, inputs):
    out = inputs
    for block in self.blocks:
      out = block(out)
    return out


def check_length(length, value, name):
  if len(value) != length:
    raise ValueError(f"`{name}` must be of length 4 not {len(value)}")


class SlimResNet(hk.Module):
  """SlimResNet model."""

  CONFIGS = {
      18: {
          "blocks_per_group": (2, 2, 2, 2),
          "bottleneck": False,
          "channels_per_group": (16, 32, 64, 128),
          "use_projection": (False, True, True, True),
      },
      34: {
          "blocks_per_group": (3, 4, 6, 3),
          "bottleneck": False,
          "channels_per_group": (16, 32, 64, 128),
          "use_projection": (False, True, True, True),
      },
      50: {
          "blocks_per_group": (3, 4, 6, 3),
          "bottleneck": True,
          "channels_per_group": (16, 32, 64, 128),
          "use_projection": (True, True, True, True),
      },
      101: {
          "blocks_per_group": (3, 4, 23, 3),
          "bottleneck": True,
          "channels_per_group": (16, 32, 64, 128),
          "use_projection": (True, True, True, True),
      },
      152: {
          "blocks_per_group": (3, 8, 36, 3),
          "bottleneck": True,
          "channels_per_group": (16, 32, 64, 128),
          "use_projection": (True, True, True, True),
      },
      200: {
          "blocks_per_group": (3, 24, 36, 3),
          "bottleneck": True,
          "channels_per_group": (16, 32, 64, 128),
          "use_projection": (True, True, True, True),
      },
  }

  BlockGroup = BlockGroup  # pylint: disable=invalid-name
  BlockV1 = BlockV1  # pylint: disable=invalid-name
  BlockV2 = BlockV2  # pylint: disable=invalid-name

  def __init__(
      self,
      blocks_per_group: Sequence[int],
      bn_config: Optional[Mapping[str, FloatStrOrBool]] = None,
      resnet_v2: bool = False,
      bottleneck: bool = True,
      channels_per_group: Sequence[int] = (256, 512, 1024, 2048),
      use_projection: Sequence[bool] = (True, True, True, True),
      logits_config: Optional[Mapping[str, Any]] = None,
      name: Optional[str] = None,
      initial_conv_config: Optional[Mapping[str, FloatStrOrBool]] = None,
  ):
    """Constructs a SlimResNet model.

    Args:
      blocks_per_group: A sequence of length 4 that indicates the number of
        blocks created in each group.
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers. By default the
        ``decay_rate`` is ``0.9`` and ``eps`` is ``1e-5``.
      resnet_v2: Whether to use the v1 or v2 SlimResNet implementation. Defaults to
        ``False``.
      bottleneck: Whether the block should bottleneck or not. Defaults to
        ``True``.
      channels_per_group: A sequence of length 4 that indicates the number
        of channels used for each block in each group.
      use_projection: A sequence of length 4 that indicates whether each
        residual block should use projection.
      logits_config: A dictionary of keyword arguments for the logits layer.
      name: Name of the module.
      initial_conv_config: Keyword arguments passed to the constructor of the
        initial :class:`~haiku.Conv2D` module.
    """
    super().__init__(name=name)
    self.resnet_v2 = resnet_v2

    bn_config = dict(bn_config or {})
    bn_config.setdefault("decay_rate", 0.9)
    bn_config.setdefault("eps", 1e-5)
    bn_config.setdefault("create_scale", True)
    bn_config.setdefault("create_offset", True)

    logits_config = dict(logits_config or {})
    logits_config.setdefault("w_init", jnp.zeros)
    logits_config.setdefault("name", "logits")

    # Number of blocks in each group for SlimResNet.
    check_length(4, blocks_per_group, "blocks_per_group")
    check_length(4, channels_per_group, "channels_per_group")

    initial_conv_config = dict(initial_conv_config or {})
    initial_conv_config.setdefault("output_channels", 16)
    initial_conv_config.setdefault("kernel_shape", 7)
    initial_conv_config.setdefault("stride", 2)
    initial_conv_config.setdefault("with_bias", False)
    initial_conv_config.setdefault("padding", "SAME")
    initial_conv_config.setdefault("name", "initial_conv")

    self.initial_conv = hk.Conv2D(**initial_conv_config)

    if not self.resnet_v2:
      self.initial_layernorm = LayerNorm(name="initial_layernorm")
                                            

    self.block_groups = []
    strides = (1, 2, 2, 2)
    for i in range(4):
      self.block_groups.append(
          BlockGroup(channels=channels_per_group[i],
                     num_blocks=blocks_per_group[i],
                     stride=strides[i],
                     bn_config=bn_config,
                     resnet_v2=resnet_v2,
                     bottleneck=bottleneck,
                     use_projection=use_projection[i],
                     name="block_group_%d" % (i)))


  def __call__(self, inputs):
    out = inputs
    out = self.initial_conv(out)
    if not self.resnet_v2:
      out = self.initial_layernorm(out)
      out = jax.nn.relu(out)

    out = hk.max_pool(out,
                      window_shape=(1, 3, 3, 1),
                      strides=(1, 2, 2, 1),
                      padding="SAME")

    xs = []
    for block_group in self.block_groups:
      out = block_group(out)
      xs.append(out)

    return xs


class SlimResNet18(SlimResNet):
  """SlimResNet18."""

  def __init__(
      self,
      bn_config: Optional[Mapping[str, FloatStrOrBool]] = None,
      resnet_v2: bool = False,
      logits_config: Optional[Mapping[str, Any]] = None,
      name: Optional[str] = None,
      initial_conv_config: Optional[Mapping[str, FloatStrOrBool]] = None,
  ):
    """Constructs a SlimResNet model.

    Args:
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 SlimResNet implementation. Defaults
        to ``False``.
      logits_config: A dictionary of keyword arguments for the logits layer.
      name: Name of the module.
      initial_conv_config: Keyword arguments passed to the constructor of the
        initial :class:`~haiku.Conv2D` module.
    """
    super().__init__(bn_config=bn_config,
                     initial_conv_config=initial_conv_config,
                     resnet_v2=resnet_v2,
                     logits_config=logits_config,
                     name=name,
                     **SlimResNet.CONFIGS[18])


class SlimResNet34(SlimResNet):
  """SlimResNet34."""

  def __init__(
      self,
      bn_config: Optional[Mapping[str, FloatStrOrBool]] = None,
      resnet_v2: bool = False,
      logits_config: Optional[Mapping[str, Any]] = None,
      name: Optional[str] = None,
      initial_conv_config: Optional[Mapping[str, FloatStrOrBool]] = None,
  ):
    """Constructs a SlimResNet model.

    Args:
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 SlimResNet implementation. Defaults
        to ``False``.
      logits_config: A dictionary of keyword arguments for the logits layer.
      name: Name of the module.
      initial_conv_config: Keyword arguments passed to the constructor of the
        initial :class:`~haiku.Conv2D` module.
    """
    super().__init__(bn_config=bn_config,
                     initial_conv_config=initial_conv_config,
                     resnet_v2=resnet_v2,
                     logits_config=logits_config,
                     name=name,
                     **SlimResNet.CONFIGS[34])


class SlimResNet50(SlimResNet):
  """SlimResNet50."""

  def __init__(
      self,
      bn_config: Optional[Mapping[str, FloatStrOrBool]] = None,
      resnet_v2: bool = False,
      logits_config: Optional[Mapping[str, Any]] = None,
      name: Optional[str] = None,
      initial_conv_config: Optional[Mapping[str, FloatStrOrBool]] = None,
  ):
    """Constructs a SlimResNet model.

    Args:
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 SlimResNet implementation. Defaults
        to ``False``.
      logits_config: A dictionary of keyword arguments for the logits layer.
      name: Name of the module.
      initial_conv_config: Keyword arguments passed to the constructor of the
        initial :class:`~haiku.Conv2D` module.
    """
    super().__init__(bn_config=bn_config,
                     initial_conv_config=initial_conv_config,
                     resnet_v2=resnet_v2,
                     logits_config=logits_config,
                     name=name,
                     **SlimResNet.CONFIGS[50])


class SlimResNet101(SlimResNet):
  """SlimResNet101."""

  def __init__(
      self,
      bn_config: Optional[Mapping[str, FloatStrOrBool]] = None,
      resnet_v2: bool = False,
      logits_config: Optional[Mapping[str, Any]] = None,
      name: Optional[str] = None,
      initial_conv_config: Optional[Mapping[str, FloatStrOrBool]] = None,
  ):
    """Constructs a SlimResNet model.

    Args:
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 SlimResNet implementation. Defaults
        to ``False``.
      logits_config: A dictionary of keyword arguments for the logits layer.
      name: Name of the module.
      initial_conv_config: Keyword arguments passed to the constructor of the
        initial :class:`~haiku.Conv2D` module.
    """
    super().__init__(bn_config=bn_config,
                     initial_conv_config=initial_conv_config,
                     resnet_v2=resnet_v2,
                     logits_config=logits_config,
                     name=name,
                     **SlimResNet.CONFIGS[101])


class SlimResNet152(SlimResNet):
  """SlimResNet152."""

  def __init__(
      self,
      bn_config: Optional[Mapping[str, FloatStrOrBool]] = None,
      resnet_v2: bool = False,
      logits_config: Optional[Mapping[str, Any]] = None,
      name: Optional[str] = None,
      initial_conv_config: Optional[Mapping[str, FloatStrOrBool]] = None,
  ):
    """Constructs a SlimResNet model.

    Args:
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 SlimResNet implementation. Defaults
        to ``False``.
      logits_config: A dictionary of keyword arguments for the logits layer.
      name: Name of the module.
      initial_conv_config: Keyword arguments passed to the constructor of the
        initial :class:`~haiku.Conv2D` module.
    """
    super().__init__(bn_config=bn_config,
                     initial_conv_config=initial_conv_config,
                     resnet_v2=resnet_v2,
                     logits_config=logits_config,
                     name=name,
                     **SlimResNet.CONFIGS[152])


class SlimResNet200(SlimResNet):
  """SlimResNet200."""

  def __init__(
      self,
      bn_config: Optional[Mapping[str, FloatStrOrBool]] = None,
      resnet_v2: bool = False,
      logits_config: Optional[Mapping[str, Any]] = None,
      name: Optional[str] = None,
      initial_conv_config: Optional[Mapping[str, FloatStrOrBool]] = None,
  ):
    """Constructs a SlimResNet model.

    Args:
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 SlimResNet implementation. Defaults
        to ``False``.
      logits_config: A dictionary of keyword arguments for the logits layer.
      name: Name of the module.
      initial_conv_config: Keyword arguments passed to the constructor of the
        initial :class:`~haiku.Conv2D` module.
    """
    super().__init__(bn_config=bn_config,
                     initial_conv_config=initial_conv_config,
                     resnet_v2=resnet_v2,
                     logits_config=logits_config,
                     name=name,
                     **SlimResNet.CONFIGS[200])

