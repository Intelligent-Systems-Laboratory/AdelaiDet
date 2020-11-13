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
from detectron2.layers import ShapeSpec
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

    def __init__(self, w_in, w_out, num_classes, dropout_ratio=0.0, norm="BN", activation_fun="silu"):
        super().__init__(w_in, w_out, 1)
        self.conv = conv2d(w_in, w_out, 1)
        self.conv_bn = get_norm(norm, w_out)
        self.conv_af = get_activation(activation_fun)
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


class MBConv(CNNBlockBase):
    """Mobile inverted bottleneck block with SE."""

    def __init__(self, w_in, exp_r, k, stride, se_r, w_out, DC_RATIO=0.0, norm="BN", activation_fun="silu"):
        # Expansion, kxk dwise, BN, AF, SE, 1x1, BN, skip_connection
        super().__init__(w_in,w_out,stride)
        self.exp = None
        w_exp = int(w_in * exp_r)
        if w_exp != w_in:
            self.exp = conv2d(w_in, w_exp, 1)
            self.exp_bn = get_norm(norm, w_exp)
            self.exp_af = get_activation(activation_fun)
        self.dwise = conv2d(w_exp, w_exp, k, stride=stride, groups=w_exp)
        self.dwise_bn = get_norm(norm, w_exp)
        self.dwise_af = get_activation(activation_fun)
        self.se = SE(w_exp, int(w_in * se_r), activation_fun)
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

    def __init__(self, params, out_features=None):
        super(EffNet, self).__init__()
        vs = ["sw", "ds", "ws", "exp_rs", "se_r", "ss", "ks", "hw", "nc", "dropout_ratio","dc_ratio","norm","activation_fun"]
        sw, ds, ws, exp_rs, se_r, ss, ks, hw, nc, dropout_ratio, dc_ratio, norm, activation_fun = [params[v] for v in vs]
        stage_params = list(zip(ds, ws, exp_rs, ss, ks))
        self.stem = StemIN(3, sw, norm, activation_fun)

        self._out_feature_strides = {"stem": self.stem.stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}

        prev_w = sw
        current_stride = self.stem.stride
        self.stage_names, self.stages = [], []
        for i, (d, w, exp_r, stride, k) in enumerate(stage_params):
            stage = EffStage(prev_w, exp_r, k, stride, se_r, w, d, dc_ratio, norm, activation_fun)
            name = "s{}".format(i+1)
            self.add_module(name, stage)
            self.stage_names.append(name)
            self.stages.append(stage)
            self._out_feature_channels[name] = w
            self._out_feature_strides[name] = current_stride = current_stride * stride
            prev_w = w
        
        self.stage_names = tuple(self.stage_names) #static for scripting

        self.num_classes = nc
        if nc is not None:
            self.head = EffHead(prev_w, hw,dropout_ratio, nc, norm, activation_fun)
        
        # Name of final layer
        if out_features is None:
            self._out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))

        #initialiaze all weights
        self.apply(init_weights)

    def forward(self, x):
        outputs = {}

        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        
        for name, stage in zip(self.stage_names, self.stages):
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        
        if self.num_classes is not None:
            x = self.head(x)
            outputs["head"] = x

        return outputs

    
    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def freeze(self, freeze_at=0):
        """
        Freeze the first several stages of the EfficientNet. Commonly used in
        fine-tuning.
        Each stage in EfficientNet is defined by :paper: `EfficientNet` Table 1 Typically, layers that produce the same feature map spatial size are defined as one
        "stage" by :paper:`FPN`. However, it doesn't apply on stage 6 and stage 2.
        Args:
            freeze_at (int): number of stages to freeze.
                `1` means freezing the stem. `2` means freezing the stem and
                one stage, etc.
        Returns:
            nn.Module: this EfficientNet itself
        """
        if freeze_at >= 1:
            self.stem.freeze()
        for idx, stage in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                for block in stage.children():
                    block.freeze()
        return self



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
        "dc_ratio": cfg.MODEL.EFFNET.DC_RATIO,
        "norm": cfg.MODEL.EFFNET.NORM,
        "activation_fun": cfg.MODEL.EFFNET.ACTIVATION_FUN,
        "nc": None
    }
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features = cfg.MODEL.EFFNET.OUT_FEATURES
    return EffNet(params, out_features).freeze(freeze_at)



