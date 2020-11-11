#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""EfficientNet models."""

# from pycls.core.config import cfg
# from pycls.models.blocks import (
#     SE,
#     activation,
#     conv2d,
#     conv2d_cx,
#     drop_connect,
#     gap2d,
#     gap2d_cx,
#     init_weights,
#     linear,
#     linear_cx,
#     norm2d_cx,
# )
from torch.nn import Dropout, Module

# Modifications for Detectron2/AdelaiDet
from detectron2.modeling.backbone import Backbone
from detectron2.layers import get_norm, CNNBlockBase
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN
from detectron2.layers import ShapeSpec
from .fpn import LastLevelP6, LastLevelP6P7
from adet.layers.pycls_blocks import (
    SE,
    get_activation,
    conv2d,
    drop_connect,
    gap2d,
    init_weights,
    linear,
)


class EffHead(CNNBlockBase):
    """EfficientNet head: 1x1, BN, AF, AvgPool, Dropout, FC."""

    def __init__(self, w_in, w_out, dropout_ratio=0.0, num_classes=None, norm="BN", activation_fun="silu"):
        super().__init__(w_in, w_out, 1)
        self.conv = conv2d(w_in, w_out, 1)
        self.conv_bn = get_norm(norm, w_out)
        self.conv_af = get_activation(activation_fun)
        self.num_classes = num_classes
        if num_classes is not None:
            self.avg_pool = gap2d(w_out)
            self.dropout = Dropout(p=dropout_ratio) if dropout_ratio > 0 else None
            self.fc = linear(w_out, num_classes, bias=True)

    def forward(self, x):
        x = self.conv_af(self.conv_bn(self.conv(x)))

        if self.num_classes is not None:
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.dropout(x) if self.dropout else x
            x = self.fc(x)
        
        return x


class MBConv(Module):
    """Mobile inverted bottleneck block with SE."""

    def __init__(self, w_in, exp_r, k, stride, se_r, w_out, DC_RATIO=0.0, norm="BN", activation_fun="silu"):
        # Expansion, kxk dwise, BN, AF, SE, 1x1, BN, skip_connection
        super(MBConv, self).__init__()
        self.exp = None
        w_exp = int(w_in * exp_r)
        if w_exp != w_in:
            self.exp = conv2d(w_in, w_exp, 1)
            self.exp_bn = get_norm(norm, w_exp)
            self.exp_af = get_activation(activation_fun)
        self.dwise = conv2d(w_exp, w_exp, k, stride=stride, groups=w_exp)
        self.dwise_bn = get_norm(norm, w_exp)
        self.dwise_af = get_activation(activation_fun)
        self.se = SE(w_exp, int(w_in * se_r))
        self.lin_proj = conv2d(w_exp, w_out, 1)
        self.lin_proj_bn = get_norm(norm, w_out)
        self.has_skip = stride == 1 and w_in == w_out
        self.DC_RATIO = DC_RATIO

    def forward(self, x):
        f_x = self.exp_af(self.exp_bn(self.exp(x))) if self.exp else x
        f_x = self.dwise_af(self.dwise_bn(self.dwise(f_x)))
        f_x = self.se(f_x)
        f_x = self.lin_proj_bn(self.lin_proj(f_x))
        if self.has_skip:
            if self.training and self.DC_RATIO > 0.0:
                f_x = drop_connect(f_x, self.DC_RATIO)
            f_x = x + f_x
        return f_x


class EffStage(CNNBlockBase):
    """EfficientNet stage."""

    def __init__(self, w_in, exp_r, k, stride, se_r, w_out, d, DC_RATIO=0.0, norm="BN", activation_fun="silu"):
        super().__init__(w_in,w_out,stride)
        for i in range(d):
            block = MBConv(w_in, exp_r, k, stride, se_r, w_out, DC_RATIO, norm, activation_fun)
            self.add_module("b{}".format(i + 1), block)
            stride, w_in = 1, w_out

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class StemIN(CNNBlockBase):
    """EfficientNet stem for ImageNet: 3x3, BN, AF."""

    def __init__(self, w_in, w_out, norm="BN", activation_fun="silu"):
        super().__init__(w_in, w_out, 2)
        self.conv = conv2d(w_in, w_out, 3, stride=2)
        self.bn = get_norm(norm, w_out)
        self.af = get_activation(activation_fun)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class EffNet(Backbone):
    """EfficientNet model."""

    def __init__(self, params):
        super(EffNet, self).__init__()
        vs = ["sw", "ds", "ws", "exp_rs", "se_r", "ss", "ks", "hw", "nc", "dropout_ratio","dc_ratio","norm","activation_fun"]
        sw, ds, ws, exp_rs, se_r, ss, ks, hw, nc, dropout_ratio, dc_ratio, norm, activation_fun = [params[v] for v in vs]
        stage_params = list(zip(ds, ws, exp_rs, ss, ks))
        self.stem = StemIN(3, sw, dc_ratio, norm, activation_fun)

        self._out_feature_strides = {"stem": self.stem.stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}

        prev_w = sw    
        for i, (d, w, exp_r, stride, k) in enumerate(stage_params):
            stage = EffStage(prev_w, exp_r, k, stride, se_r, w, d, dc_ratio, norm, activation_fun)
            name = "s{}".format(i+1)
            self.add_module(name, stage)
            self._out_feature_channels[name] = w
            self._out_feature_strides[name] = stride
            prev_w = w

        self.head = EffHead(prev_w, hw,dropout_ratio, nc, norm, activation_fun)
        self.apply(init_weights)


    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x
    
    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }



@BACKBONE_REGISTRY.register()
def build_effnet_backbone(cfg, input_shape: ShapeSpec):
    params = {
        "sw": cfg.MODEL.EFFNET.STEM_W,
        "ds": cfg.MODEL.EFFNET.DEPTHS,
        "ws": cfg.MODEL.EFFNET.WIDTHS,
        "exp_rs": cfg.MODEL.EFFNET.EXP_RATIOS,
        "se_r": cfg.MODEL.EFFNET.SE_R,
        "ss": cfg.MODEL.EFFNET.STRIDES,
        "ks": cfg.MODEL.EFFNET.KERNELS,
        "hw": cfg.MODEL.EFFNET.HEAD_W,
        "dropout_ratio": cfg.MODEL.EFFNET.DROPOUT_RATIO,
        "norm": cfg.MODEL.EFFNET.NORM,
        "activation_fun": cfg.MODEL.EFFNET.ACTIVATION_FUN,
        "nc": None
    }
    model = EffNet(params)
    return model



