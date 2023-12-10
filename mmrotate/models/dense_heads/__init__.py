# Copyright (c) OpenMMLab. All rights reserved.

from .rotated_rtmdet_head import RotatedRTMDetHead, RotatedRTMDetSepBNHead
from .rotated_rtmdet_tid_head import RotatedRTMDetTIDSepBNHead

__all__ = [
    'RotatedRTMDetHead', 'RotatedRTMDetSepBNHead', 'RotatedRTMDetTIDSepBNHead'
]
