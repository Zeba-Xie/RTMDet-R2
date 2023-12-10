# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.structures.bbox import (HorizontalBoxes, bbox_overlaps,
                                   get_box_tensor)
from torch import Tensor

from mmrotate.registry import TASK_UTILS
from mmrotate.structures.bbox import (QuadriBoxes, RotatedBoxes,
                                      fake_rbbox_overlaps, rbbox_overlaps)
from mmrotate.models.losses.prob_iou_loss import probiou_loss, probiou_loss2

@TASK_UTILS.register_module()
class RBboxOverlaps2D(object):
    """2D Overlaps Calculator for Rotated Bboxes."""

    def __call__(self,
                 bboxes1: RotatedBoxes,
                 bboxes2: RotatedBoxes,
                 mode: str = 'iou',
                 is_aligned: bool = False) -> Tensor:
        """Calculate IoU between 2D rotated bboxes.

        Args:
            bboxes1 (:obj:`RotatedBoxes` or Tensor): bboxes have shape (m, 5)
                in <cx, cy, w, h, t> format, shape (m, 6) in
                <cx, cy, w, h, t, score> format.
            bboxes2 (:obj:`RotatedBoxes` or Tensor): bboxes have shape (n, 5)
                in <cx, cy, w, h, t> format, shape (n, 6) in
                <cx, cy, w, h, t, score> format, or be empty.
            mode (str): 'iou' (intersection over union), 'iof' (intersection
                over foreground). Defaults to 'iou'.
            is_aligned (bool): If True, then m and n must be equal.
                Defaults to False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 5, 6]
        assert bboxes2.size(-1) in [0, 5, 6]

        if bboxes1.size(-1) == 6:
            bboxes1 = bboxes1[..., :5]
        if bboxes2.size(-1) == 6:
            bboxes2 = bboxes2[..., :5]

        bboxes1 = get_box_tensor(bboxes1)
        bboxes2 = get_box_tensor(bboxes2)

        return rbbox_overlaps(bboxes1, bboxes2, mode, is_aligned)

    def test(self,
             bboxes1,
             bboxes2
             ):
        return rbbox_overlaps(bboxes1, bboxes2, 'iou', False)


    def __repr__(self) -> str:
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str


@TASK_UTILS.register_module()
class FakeRBboxOverlaps2D(object):
    """2D Overlaps Calculator for Minimum Circumscribed Horizental Bboxes of
    Rotated Bboxes."""

    def __call__(self,
                 bboxes1: RotatedBoxes,
                 bboxes2: RotatedBoxes,
                 mode: str = 'iou',
                 is_aligned: bool = False) -> Tensor:
        """Calculate IoU between 2D minimum circumscribed hbbs of rbbs.

        Args:
            bboxes1 (:obj:`RotatedBoxes` or Tensor): bboxes have shape (m, 5)
                in <cx, cy, w, h, t> format, shape (m, 6) in
                <cx, cy, w, h, t, score> format.
            bboxes2 (:obj:`RotatedBoxes` or Tensor): bboxes have shape (n, 5)
                in <cx, cy, w, h, t> format, shape (n, 6) in
                <cx, cy, w, h, t, score> format, or be empty.
            mode (str): 'iou' (intersection over union), 'iof' (intersection
                over foreground).
            is_aligned (bool): If True, then m and n must be equal.
                Defaults to False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 5, 6]
        assert bboxes2.size(-1) in [0, 5, 6]

        if bboxes1.size(-1) == 6:
            bboxes1 = bboxes1[..., :5]
        if bboxes2.size(-1) == 6:
            bboxes2 = bboxes2[..., :5]

        if not isinstance(bboxes1, RotatedBoxes):
            bboxes1 = RotatedBoxes(bboxes1)
        if not isinstance(bboxes2, RotatedBoxes):
            bboxes2 = RotatedBoxes(bboxes2)

        return fake_rbbox_overlaps(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self) -> str:
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str


def cast_tensor_type(x: Tensor,
                     scale: float = 1.,
                     dtype: str = None) -> Tensor:
    if dtype == 'fp16':
        # scale is for preventing overflows
        x = (x / scale).half()
    return x


@TASK_UTILS.register_module()
class RBbox2HBboxOverlaps2D:
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __init__(self, scale: float = 1., dtype: str = None) -> None:
        self.scale = scale
        self.dtype = dtype

    def __call__(self,
                 bboxes1: RotatedBoxes,
                 bboxes2: HorizontalBoxes,
                 mode: str = 'iou',
                 is_aligned: bool = False) -> Tensor:
        """Convert gt from rbb to hbb, and calculate IoU between hbboxes.

        Args:
            bboxes1 (:obj:`RotatedBoxes` or Tensor): bboxes have shape (m, 5)
                in <cx, cy, w, h, t> format, shape (m, 6) in
                <cx, cy, w, h, t, score> format.
            bboxes2 (:obj:`HorizontalBoxes` or Tensor): bboxes have shape
                (n, 4) in <x1, y1, x2, y2> format, shape (n, 5) in
                <x1, y1, x2, y2, score> format, or be empty.
            mode (str): 'iou' (intersection over union), 'iof' (intersection
                over foreground).
            is_aligned (bool): If True, then m and n must be equal.
                Defaults to False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 5, 6]
        assert bboxes2.size(-1) in [0, 4, 5]

        if bboxes1.size(-1) == 6:
            bboxes1 = bboxes1[..., :5]
        if bboxes2.size(-1) == 5:
            bboxes2 = bboxes2[..., :4]

        if not isinstance(bboxes1, RotatedBoxes):
            bboxes1 = RotatedBoxes(bboxes1)
        # convert rbb to minimum circumscribed hbb in <x1, y1, x2, y2> format.
        bboxes1 = bboxes1.convert_to('hbox').tensor
        bboxes2 = get_box_tensor(bboxes2)

        if self.dtype == 'fp16':
            # change tensor type to save cpu and cuda memory and keep speed
            bboxes1 = cast_tensor_type(bboxes1, self.scale, self.dtype)
            bboxes2 = cast_tensor_type(bboxes2, self.scale, self.dtype)
            overlaps = bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)
            if not overlaps.is_cuda and overlaps.dtype == torch.float16:
                # resume cpu float32
                overlaps = overlaps.float()
            return overlaps

        return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self) -> str:
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + f'(' \
            f'scale={self.scale}, dtype={self.dtype})'
        return repr_str


@TASK_UTILS.register_module()
class QBbox2HBboxOverlaps2D:
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __init__(self, scale: float = 1., dtype: str = None) -> None:
        self.scale = scale
        self.dtype = dtype

    def __call__(self,
                 bboxes1: QuadriBoxes,
                 bboxes2: HorizontalBoxes,
                 mode: str = 'iou',
                 is_aligned: bool = False) -> Tensor:
        """Convert gt from qbb to hbb, and calculate IoU between hbboxes.

        Args:
            bboxes1 (:obj:`QuadriBoxes` or Tensor): bboxes have shape (m, 8)
                in <x1, y1, ..., x4, y4> format, shape (m, 9) in
                <x1, y1, ..., x4, y4, score> format.
            bboxes2 (:obj:`HorizontalBoxes` or Tensor): bboxes have shape
                (n, 4) in <x1, y1, x2, y2> format, shape (n, 5) in
                <x1, y1, x2, y2, score> format, or be empty.
            mode (str): 'iou' (intersection over union), 'iof' (intersection
                over foreground).
            is_aligned (bool): If True, then m and n must be equal.
                Defaults to False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 8, 9]
        assert bboxes2.size(-1) in [0, 4, 5]

        if bboxes1.size(-1) == 9:
            bboxes1 = bboxes1[..., :8]
        if bboxes2.size(-1) == 5:
            bboxes2 = bboxes2[..., :4]

        if not isinstance(bboxes1, QuadriBoxes):
            bboxes1 = QuadriBoxes(bboxes1)
        # convert qbb to minimum circumscribed hbb in <x1, y1, x2, y2> format.
        if bboxes1.size(0) == 0:
            bboxes1 = bboxes1.new_zeros(0, 4)
        else:
            bboxes1 = bboxes1.convert_to('hbox').tensor

        bboxes2 = get_box_tensor(bboxes2)

        if self.dtype == 'fp16':
            # change tensor type to save cpu and cuda memory and keep speed
            bboxes1 = cast_tensor_type(bboxes1, self.scale, self.dtype)
            bboxes2 = cast_tensor_type(bboxes2, self.scale, self.dtype)
            overlaps = bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)
            if not overlaps.is_cuda and overlaps.dtype == torch.float16:
                # resume cpu float32
                overlaps = overlaps.float()
            return overlaps

        return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self) -> str:
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + f'(' \
            f'scale={self.scale}, dtype={self.dtype})'
        return repr_str


@TASK_UTILS.register_module()
class RBboxProbIoU2D(object):
    """2D ProbIoU Calculator for Rotated Bboxes."""

    def __init__(self,
                 mode: str = 'l1',
                 deform=True,
                 with_form='none',
                 use_pv2=False,
                 eps=1e-3) -> None:
        assert mode in ['l1']
        assert with_form in ['none','square', 'sqrt']
        self.mode = mode
        self.deform = deform
        self.with_form = with_form
        self.use_probiou_loss2 = use_pv2
        self.eps = eps

    def __call__(self,
                 bboxes1: RotatedBoxes,
                 bboxes2: RotatedBoxes,
                 ) -> Tensor:
        """Calculate ProbIoU between 2D rotated bboxes.

        Args:
            bboxes1 (:obj:`RotatedBoxes` or Tensor): bboxes have shape (m, 5)
                in <cx, cy, w, h, t> format, shape (m, 6) in
                <cx, cy, w, h, t, score> format.
            bboxes2 (:obj:`RotatedBoxes` or Tensor): bboxes have shape (n, 5)
                in <cx, cy, w, h, t> format, shape (n, 6) in
                <cx, cy, w, h, t, score> format, or be empty.
            mode (str): . Defaults to 'l1'.
        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 5, 6]
        assert bboxes2.size(-1) in [0, 5, 6]
        
        assert bboxes1.size(-1) == bboxes2.size(-1)

        if bboxes1.size(-1) == 6:
            bboxes1 = bboxes1[..., :5]
        if bboxes2.size(-1) == 6:
            bboxes2 = bboxes2[..., :5]

        bboxes1 = get_box_tensor(bboxes1)
        bboxes2 = get_box_tensor(bboxes2)

        return self.get_probiou(bboxes1, bboxes2, self.mode, self.deform, with_form=self.with_form)

    def get_probiou(self,
                    bboxes1,
                    bboxes2,
                    mode='l1',
                    deform=True,
                    with_form='none'
                    ):
        # resolve `rbbox_overlaps` abnormal when input rbbox is too small.
        clamped_bboxes1 = bboxes1.detach().clone()
        clamped_bboxes2 = bboxes2.detach().clone()
        clamped_bboxes1[:, 2:4].clamp_(min=1e-3)
        clamped_bboxes2[:, 2:4].clamp_(min=1e-3)

        # resolve `rbbox_overlaps` abnormal when coordinate value is too large.
        clamped_bboxes1[:, :2].clamp_(min=-1e7, max=1e7)
        clamped_bboxes2[:, :2].clamp_(min=-1e7, max=1e7)

        rows = clamped_bboxes1.size(0)
        cols = clamped_bboxes2.size(0)
        bbox_dim = clamped_bboxes1.size(-1)

        clamped_bboxes1 = clamped_bboxes1[:, None, :].expand(rows, cols, bbox_dim)
        clamped_bboxes2 = clamped_bboxes2[None, :, :].expand(rows, cols, bbox_dim)

        clamped_bboxes1 = clamped_bboxes1.reshape(-1, bbox_dim)
        clamped_bboxes2 = clamped_bboxes2.reshape(-1, bbox_dim)

        if self.use_probiou_loss2:
            probiou = probiou_loss2(clamped_bboxes1, clamped_bboxes2, mode=mode, deform=deform, eps=self.eps)
        else:
            probiou = probiou_loss(clamped_bboxes1, clamped_bboxes2, mode=mode, deform=deform, eps=self.eps)

        probiou = probiou.reshape(rows, cols, -1)

        probiou = torch.squeeze(probiou, -1)

        probiou = 1 - probiou

        # > 0
        # zeros = torch.zeros_like(probiou)
        # probiou = torch.max(zeros, probiou)

        # square
        if with_form == 'square':
            probiou = torch.square(probiou)
        elif with_form == 'sqrt':
            zeros = torch.zeros_like(probiou)
            probiou = torch.max(zeros, probiou)
            probiou = torch.sqrt(probiou)

        return probiou

    def test(self,
             bboxes1,
             bboxes2,
             mode='l1',
             deform=True,
             with_form='none'
             ):

        return self.get_probiou(bboxes1, bboxes2, mode, deform=deform, with_form=with_form)

    def __repr__(self) -> str:
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str
