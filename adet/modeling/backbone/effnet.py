#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""EfficientNet models."""
from torch import nn
from torch.nn import BatchNorm2d
#from detectron2.layers.batch_norm import NaiveSyncBatchNorm as BatchNorm2d
from detectron2.layers import Conv2d
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone import Backbone

from .effnet_blocks import (
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
from torch.nn import Dropout, Module


class EffHead(Module):
    """EfficientNet head: 1x1, BN, AF, AvgPool, Dropout, FC."""

    def __init__(self, w_in, w_out, num_classes):
        super(EffHead, self).__init__()
        dropout_ratio = cfg.MODEL.EN.DROPOUT_RATIO
        self.conv = conv2d(w_in, w_out, 1)
        self.conv_bn = norm2d(cfg, w_out)
        self.conv_af = activation(cfg)
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
            self.exp_bn = norm2d(cfg, w_exp)
            self.exp_af = activation(cfg)
        self.dwise = conv2d(w_exp, w_exp, k, stride=stride, groups=w_exp)
        self.dwise_bn = norm2d(cfg, w_exp)
        self.dwise_af = activation(cfg)
        self.se = SE(cfg, w_exp, int(w_in * se_r))
        self.lin_proj = conv2d(w_exp, w_out, 1)
        self.lin_proj_bn = norm2d(cfg, w_out)
        self.has_skip = stride == 1 and w_in == w_out

    def forward(self, x):
        f_x = self.exp_af(self.exp_bn(self.exp(x))) if self.exp else x
        f_x = self.dwise_af(self.dwise_bn(self.dwise(f_x)))
        f_x = self.se(f_x)
        f_x = self.lin_proj_bn(self.lin_proj(f_x))
        if self.has_skip:
            if self.training and cfg.MODEL.EN.DC_RATIO > 0.0:
                f_x = drop_connect(f_x, cfg.MODEL.EN.DC_RATIO)
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

    def __init__(self, w_in, w_out):
        super(StemIN, self).__init__()
        self.conv = conv2d(w_in, w_out, 3, stride=2)
        self.bn = norm2d(cfg, w_out)
        self.af = activation(cfg)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 3, stride=2)
        cx = norm2d_cx(cx, w_out)
        return cx


class EffNet(Backbone):
    """EfficientNet model."""

    @staticmethod
    def get_params(cfg):
        return {
            "sw": cfg.MODEL.EN.STEM_W,
            "ds": cfg.MODEL.EN.DEPTHS,
            "ws": cfg.MODEL.EN.WIDTHS,
            "exp_rs": cfg.MODEL.EN.EXP_RATIOS,
            "se_r": cfg.MODEL.EN.SE_R,
            "ss": cfg.MODEL.EN.STRIDES,
            "ks": cfg.MODEL.EN.KERNELS,
            "hw": cfg.MODEL.EN.HEAD_W,
            "nc": cfg.MODEL.NUM_CLASSES,
        }

    def __init__(self, cfg, params=None):
        super(EffNet, self).__init__()
        p = EffNet.get_params(cfg) if not params else params
        vs = ["sw", "ds", "ws", "exp_rs", "se_r", "ss", "ks", "hw", "nc"]
        sw, ds, ws, exp_rs, se_r, ss, ks, hw, nc = [p[v] for v in vs]
        stage_params = list(zip(ds, ws, exp_rs, ss, ks))
        self.stem = StemIN(3, sw)
        prev_w = sw
        for i, (d, w, exp_r, stride, k) in enumerate(stage_params):
            stage = EffStage(cfg, prev_w, exp_r, k, stride, se_r, w, d)
            self.add_module("s{}".format(i + 1), stage)
            prev_w = w
        #self.head = EffHead(prev_w, hw, nc)
        self.apply(init_weights(cfg=cfg))

    def forward(self, x):
        #out_stage = ['s3','s4','s5','s6','s7'] 
        output ={}
        for name, module in self.named_children():
            x = module(x)
            if name in ['s3','s4','s5','s6','s7']:
                output[name] = x
        return output

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
def build_effnet_backbone(cfg, input_shape):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    #out_features = cfg.MODEL.RESNETS.OUT_FEATURES
    out_features = cfg.MODEL.EN.OUT_FEATURES

    #out_feature_channels = {"res2": 24, "res3": 32,
    #                        "res4": 96, "res5": 320}
    #out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
    model = EffNet(cfg)
    model._out_features = out_features
    #model._out_feature_channels = out_feature_channels
    #model._out_feature_strides = out_feature_strides
    return model
