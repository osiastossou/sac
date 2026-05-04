"""
detector.py
===========
Détecteur d'objets léger pour VisDrone.

Architecture :
    Backbone  : 4 blocs de convolution (échangeable via CONV_REGISTRY)
    Neck      : FPN simplifié à 2 niveaux
    Head      : Conv 1×1 → (classes + 4 bbox coords)

Le backbone utilise exclusivement la couche de convolution passée en argument,
ce qui permet de comparer les 9 opérateurs à architecture par ailleurs identique.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .convolutions import build_conv


class ConvBlock(nn.Module):
    """Bloc : Conv adaptative + BN (intégré à la conv) + ReLU."""

    def __init__(self, conv_name, in_ch, out_ch, stride=1, **conv_kwargs):
        super().__init__()
        self.conv = build_conv(conv_name, in_ch, out_ch,
                               stride=stride, **conv_kwargs)

    def forward(self, x):
        return self.conv(x)


class TinyDetector(nn.Module):
    """
    Détecteur léger pour VisDrone.

    Args:
        conv_name   : nom de l'opérateur (clé dans CONV_REGISTRY)
        num_classes : nombre de classes VisDrone (10 par défaut)
        base_ch     : largeur du backbone (par défaut 32)
        conv_kwargs : arguments supplémentaires passés à la conv
    """

    def __init__(self, conv_name: str = "standard",
                 num_classes: int = 10,
                 base_ch: int = 32,
                 **conv_kwargs):
        super().__init__()
        self.conv_name = conv_name
        self.num_classes = num_classes
        C = base_ch

        # ── Stem ─────────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )

        # ── Backbone (4 stages) ───────────────────────────────────────────────
        #   stage1 : C  → C    stride=1   (P2 — 1/4)
        #   stage2 : C  → 2C   stride=2   (P3 — 1/8)
        #   stage3 : 2C → 4C   stride=2   (P4 — 1/16)
        #   stage4 : 4C → 8C   stride=2   (P5 — 1/32)
        self.stage1 = ConvBlock(conv_name, C,    C,    stride=1, **conv_kwargs)
        self.stage2 = ConvBlock(conv_name, C,    2*C,  stride=2, **conv_kwargs)
        self.stage3 = ConvBlock(conv_name, 2*C,  4*C,  stride=2, **conv_kwargs)
        self.stage4 = ConvBlock(conv_name, 4*C,  8*C,  stride=2, **conv_kwargs)

        # ── Neck FPN (2 niveaux : P4, P5) ─────────────────────────────────────
        self.lat_p5 = nn.Conv2d(8*C, 4*C, 1, bias=False)
        self.lat_p4 = nn.Conv2d(4*C, 4*C, 1, bias=False)
        self.out_p5 = nn.Conv2d(4*C, 4*C, 3, padding=1, bias=False)
        self.out_p4 = nn.Conv2d(4*C, 4*C, 3, padding=1, bias=False)

        # ── Head : un head partagé par niveau ────────────────────────────────
        head_ch = 4 * C
        self.cls_head = nn.Conv2d(head_ch, num_classes, 1)
        self.reg_head = nn.Conv2d(head_ch, 4, 1)

    # ── FPN ──────────────────────────────────────────────────────────────────
    def _fpn(self, c4, c5):
        p5 = self.lat_p5(c5)
        p4 = self.lat_p4(c4) + F.interpolate(p5, scale_factor=2,
                                               mode='nearest')
        p5 = self.out_p5(p5)
        p4 = self.out_p4(p4)
        return p4, p5

    # ── Forward ──────────────────────────────────────────────────────────────
    def forward(self, x):
        """
        Returns:
            cls_preds : list[(B, num_classes, H', W')] par niveau FPN
            reg_preds : list[(B, 4, H', W')]           par niveau FPN
        """
        s = self.stem(x)        # /2
        c1 = self.stage1(s)     # /2  (P2)
        c2 = self.stage2(c1)    # /4  (P3)
        c3 = self.stage3(c2)    # /8  (P4)
        c4 = self.stage4(c3)    # /16 (P5)

        p4, p5 = self._fpn(c3, c4)

        cls_preds = [self.cls_head(p4), self.cls_head(p5)]
        reg_preds = [self.reg_head(p4), self.reg_head(p5)]
        return cls_preds, reg_preds


def build_detector(conv_name: str, num_classes: int = 10,
                   base_ch: int = 32, **conv_kwargs) -> TinyDetector:
    return TinyDetector(conv_name=conv_name,
                        num_classes=num_classes,
                        base_ch=base_ch,
                        **conv_kwargs)
