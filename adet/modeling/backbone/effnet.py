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
#     norm2d,
#     norm2d_cx,
# )
from torch.nn import Dropout, Module

# Modifications for Detectron2/AdelaiDet
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN
from detectron2.layers import ShapeSpec
from .fpn import LastLevelP6, LastLevelP6P7
from .bifpn import BiFPN

from adet.layers.pycls_blocks import (
    SE,
    activation,
    conv2d,
    conv2d_cx,
    drop_connect,
    gap2d,
    gap2d_cx,
    init_weights,
    linear,
    linear_cx,
    norm2d,
    norm2d_cx,
)


class EffHead(Backbone):
    """EfficientNet head: 1x1, BN, AF, AvgPool, Dropout, FC."""

    def __init__(self, cfg, w_in, w_out, num_classes):
        super(EffHead, self).__init__()
        dropout_ratio = cfg.MODEL.EffNet.DROPOUT_RATIO
        self.conv = conv2d(w_in, w_out, 1)
        self.conv_bn = norm2d(w_out)
        self.conv_af = activation()
        self.avg_pool = gap2d(w_out)
        self.dropout = Dropout(p=dropout_ratio) if dropout_ratio > 0 else None
        self.fc = linear(w_out, num_classes, bias=True)

    def forward(self, x):
        x = self.conv_af(self.conv_bn(self.conv(x)))
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x) if self.dropout else x
        x = self.fc(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, num_classes):
        cx = conv2d_cx(cx, w_in, w_out, 1)
        cx = norm2d_cx(cx, w_out)
        cx = gap2d_cx(cx, w_out)
        cx = linear_cx(cx, w_out, num_classes, bias=True)
        return cx


class MBConv(Module):
    """Mobile inverted bottleneck block with SE."""

    def __init__(self, cfg, w_in, exp_r, k, stride, se_r, w_out):
        # Expansion, kxk dwise, BN, AF, SE, 1x1, BN, skip_connection
        super(MBConv, self).__init__()
        self.exp = None
        w_exp = int(w_in * exp_r)
        if w_exp != w_in:
            self.exp = conv2d(w_in, w_exp, 1)
            self.exp_bn = norm2d(w_exp)
            self.exp_af = activation()
        self.dwise = conv2d(w_exp, w_exp, k, stride=stride, groups=w_exp)
        self.dwise_bn = norm2d(w_exp)
        self.dwise_af = activation()
        self.se = SE(w_exp, int(w_in * se_r))
        self.lin_proj = conv2d(w_exp, w_out, 1)
        self.lin_proj_bn = norm2d(w_out)
        self.has_skip = stride == 1 and w_in == w_out

    def forward(self, x):
        f_x = self.exp_af(self.exp_bn(self.exp(x))) if self.exp else x
        f_x = self.dwise_af(self.dwise_bn(self.dwise(f_x)))
        f_x = self.se(f_x)
        f_x = self.lin_proj_bn(self.lin_proj(f_x))
        if self.has_skip:
            if self.training and cfg.MODEL.EffNet.DC_RATIO > 0.0:
                f_x = drop_connect(f_x, cfg.MODEL.EffNet.DC_RATIO)
            f_x = x + f_x
        return f_x

    @staticmethod
    def complexity(cx, w_in, exp_r, k, stride, se_r, w_out):
        w_exp = int(w_in * exp_r)
        if w_exp != w_in:
            cx = conv2d_cx(cx, w_in, w_exp, 1)
            cx = norm2d_cx(cx, w_exp)
        cx = conv2d_cx(cx, w_exp, w_exp, k, stride=stride, groups=w_exp)
        cx = norm2d_cx(cx, w_exp)
        cx = SE.complexity(cx, w_exp, int(w_in * se_r))
        cx = conv2d_cx(cx, w_exp, w_out, 1)
        cx = norm2d_cx(cx, w_out)
        return cx


class EffStage(Module):
    """EfficientNet stage."""

    def __init__(self, cfg, w_in, exp_r, k, stride, se_r, w_out, d):
        super(EffStage, self).__init__()
        for i in range(d):
            block = MBConv(cfg, w_in, exp_r, k, stride, se_r, w_out)
            self.add_module("b{}".format(i + 1), block)
            stride, w_in = 1, w_out

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

    @staticmethod
    def complexity(cx, w_in, exp_r, k, stride, se_r, w_out, d):
        for _ in range(d):
            cx = MBConv.complexity(cx, w_in, exp_r, k, stride, se_r, w_out)
            stride, w_in = 1, w_out
        return cx


class StemIN(Module):
    """EfficientNet stem for ImageNet: 3x3, BN, AF."""

    def __init__(self, cfg, w_in, w_out):
        super(StemIN, self).__init__()
        self.conv = conv2d(w_in, w_out, 3, stride=2)
        self.bn = norm2d(w_out)
        self.af = activation()

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 3, stride=2)
        cx = norm2d_cx(cx, w_out)
        return cx


# Modify to 'Backbone' to fit in Detectron2
class EffNet(Backbone):
    """EfficientNet model."""

    @staticmethod
    def get_params():
        return {
            "sw": cfg.EN.STEM_W,
            "ds": cfg.EN.DEPTHS,
            "ws": cfg.EN.WIDTHS,
            "exp_rs": cfg.EN.EXP_RATIOS,
            "se_r": cfg.EN.SE_R,
            "ss": cfg.EN.STRIDES,
            "ks": cfg.EN.KERNELS,
            "hw": cfg.EN.HEAD_W,
            "nc": cfg.MODEL.NUM_CLASSES,
        }

    def __init__(self, cfg, params=None):
        super(EffNet, self).__init__()
        p = EffNet.get_params() if not params else params
        vs = ["sw", "ds", "ws", "exp_rs", "se_r", "ss", "ks", "hw", "nc"]
        sw, ds, ws, exp_rs, se_r, ss, ks, hw, nc = [p[v] for v in vs]
        stage_params = list(zip(ds, ws, exp_rs, ss, ks))
        self.stem = StemIN(cfg, 3, sw)
        prev_w = sw
        for i, (d, w, exp_r, stride, k) in enumerate(stage_params):
            stage = EffStage(cfg, prev_w, exp_r, k, stride, se_r, w, d)
            self.add_module("s{}".format(i + 1), stage)
            prev_w = w
        # self.head = EffHead(cfg, prev_w, hw, nc)
        # self.apply(init_weights)

    # def forward(self, x):
    #     for module in self.children():
    #         x = module(x)
    #     return x

# Modify EffNet forward function
    def forward(self, x):
        outputs = {}
        # outputs['stem'] = self.stem
        for m in self._modules:
            module = self._modules[m]
            x = module(x)
            outputs[m] = x

        return outputs

    @staticmethod
    def complexity(cx, params=None):
        """Computes model complexity (if you alter the model, make sure to update)."""
        p = EffNet.get_params() if not params else params
        vs = ["sw", "ds", "ws", "exp_rs", "se_r", "ss", "ks", "hw", "nc"]
        sw, ds, ws, exp_rs, se_r, ss, ks, hw, nc = [p[v] for v in vs]
        stage_params = list(zip(ds, ws, exp_rs, ss, ks))
        cx = StemIN.complexity(cx, 3, sw)
        prev_w = sw
        for d, w, exp_r, stride, k in stage_params:
            cx = EffStage.complexity(cx, prev_w, exp_r, k, stride, se_r, w, d)
            prev_w = w
        cx = EffHead.complexity(cx, prev_w, hw, nc)
        return cx

@BACKBONE_REGISTRY.register()
def build_effnet_backbone(cfg, input_shape: ShapeSpec):
    params = {
        "sw": cfg.MODEL.EffNet.STEM_W,
        "ds": cfg.MODEL.EffNet.DEPTHS,
        "ws": cfg.MODEL.EffNet.WIDTHS,
        "exp_rs": cfg.MODEL.EffNet.EXP_RATIOS,
        "se_r": cfg.MODEL.EffNet.SE_R,
        "ss": cfg.MODEL.EffNet.STRIDES,
        "ks": cfg.MODEL.EffNet.KERNELS,
        "hw": cfg.MODEL.EffNet.HEAD_W,
        "nc": cfg.MODEL.FCOS.NUM_CLASSES, # Not used since EffHead is removed
    }
    model = EffNet(cfg, params)
    model._out_features = cfg.MODEL.EffNet.OUT_FEATURES
    model._out_feature_channels = dict(zip(cfg.MODEL.EffNet.OUT_FEATURES, cfg.MODEL.EffNet.OUT_FEATURE_CHANNELS))
    model._out_feature_strides = dict(zip(cfg.MODEL.EffNet.OUT_FEATURES, cfg.MODEL.EffNet.OUT_FEATURE_STRIDES))
    # model._out_feature_channels = {"s3": 48, "s4": 96, "s5": 136, "s6": 232, "s7": 384}
    # model._out_feature_strides = {"s3": 4, "s4": 8, "s5": 16, "s6": 32, "s7": 64}

    return model


@BACKBONE_REGISTRY.register()
def build_effnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    bottom_up = build_effnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    # top_levels = cfg.MODEL.FCOS.TOP_LEVELS

    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7(out_channels, out_channels, "p5"),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone


