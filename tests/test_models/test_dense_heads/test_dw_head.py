# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.structures import InstanceData

from mmrotate.models.dense_heads import DWHead


class TestDWHead(TestCase):

    def test_dw_head_loss(self):
        """Tests dw head loss when truth is empty and non-empty."""
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'pad_shape': (s, s, 3),
            'scale_factor': 1,
        }]
        dw_head = DWHead(
            num_classes=4,
            in_channels=1,
            feat_channels=1,
            stacked_convs=1,
            norm_cfg=None)

        # dw head expects a multiple levels of features per image
        feats = (
            torch.rand(1, 1, s // stride[1], s // stride[0])
            for stride in dw_head.prior_generator.strides)
        cls_scores, bbox_preds, centernesses = dw_head.forward(feats)

        # Test that empty ground truth encourages the network to
        # predict background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 4))
        gt_instances.labels = torch.LongTensor([])

        empty_gt_losses = dw_head.loss_by_feat(cls_scores, bbox_preds,
                                                 centernesses, [gt_instances],
                                                 img_metas)
        # When there is no truth, the cls loss should be nonzero but
        # box loss and centerness loss should be zero
        empty_cls_pos_loss = empty_gt_losses['loss_cls_pos'].item()
        empty_loc_loss = empty_gt_losses['loss_loc'].item()
        empty_cls_neg_loss = empty_gt_losses['loss_cls_neg'].item()

        # print(empty_cls_pos_loss)
        # print(empty_loc_loss)
        # print(empty_cls_neg_loss)

        self.assertEqual(empty_cls_pos_loss, 0, 'cls pos loss should be zero')
        self.assertEqual(empty_loc_loss, 0,
            'there should be no box loss when there are no true boxes')
        self.assertGreater(empty_cls_neg_loss, 0,
            'cls neg loss should be non-zero')

        # When truth is non-empty then all cls, box loss and centerness loss
        # should be nonzero for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.Tensor(
            [[23.6667, 23.8757, 238.6326, 151.8874]])
        gt_instances.labels = torch.LongTensor([2])

        one_gt_losses = dw_head.loss_by_feat(cls_scores, bbox_preds,
                                               centernesses, [gt_instances],
                                               img_metas)
        onegt_cls_pos_loss = one_gt_losses['loss_cls_pos'].item()
        onegt_loc_loss = one_gt_losses['loss_loc'].item()
        onegt_cls_neg_loss = one_gt_losses['loss_cls_neg'].item()
        self.assertGreater(onegt_cls_pos_loss, 0, 'cls pos loss should be non-zero')
        self.assertGreater(onegt_loc_loss, 0, 'loc loss should be non-zero')
        self.assertGreater(onegt_cls_neg_loss, 0, 'cls neg loss should be non-zero')

