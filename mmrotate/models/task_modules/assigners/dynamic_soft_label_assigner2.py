# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import TASK_UTILS
from mmdet.structures.bbox import BaseBoxes
from mmdet.utils import ConfigType
from mmdet.models.task_modules.assigners.assign_result import AssignResult
from mmdet.models.task_modules.assigners.base_assigner import BaseAssigner
from mmdet.models.task_modules.assigners.dynamic_soft_label_assigner import center_of_mass
from mmdet.structures.bbox import get_box_tensor

INF = 100000000
EPS = 1.0e-7

@TASK_UTILS.register_module()
class DynamicSoftLabelAssigner2(BaseAssigner):
    """Computes matching between predictions and ground truth with dynamic soft
    label assignment.

    Args:
        soft_center_radius (float): Radius of the soft center prior.
            Defaults to 3.0.
        topk (int): Select top-k predictions to calculate dynamic k
            best matches for each gt. Defaults to 13.
        iou_weight (float): The scale factor of iou cost. Defaults to 3.0.
        iou_calculator (ConfigType): Config of overlaps Calculator.
            Defaults to dict(type='BboxOverlaps2D').
    """

    def __init__(
        self,
        soft_center_radius: float = 3.0,
        topk: int = 13,
        iou_weight: float = 3.0,
        iou_calculator: ConfigType = dict(type='BboxOverlaps2D'),
        use_wh_metric=False,
        wh_metric_weight=0.01,
        max_k=None,
        use_iou_cut=False,
        iou_cut_threshold=0.2
    ) -> None:
        self.soft_center_radius = soft_center_radius
        self.topk = topk
        self.iou_weight = iou_weight
        self.iou_calculator = TASK_UTILS.build(iou_calculator)

        self.use_wh_metric = use_wh_metric
        self.wh_metric_weight = wh_metric_weight

        self.max_k=max_k

        self.use_iou_cut = use_iou_cut
        self.iou_cut_threshold = iou_cut_threshold

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               **kwargs) -> AssignResult:
        """Assign gt to priors.

        Args:
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors or points, or the bboxes predicted by the
                previous stage, has shape (n, 4). The bboxes predicted by
                the current model or stage will be named ``bboxes``,
                ``labels``, and ``scores``, the same as the ``InstanceData``
                in other places.
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                and ``labels``, with shape (k, ).
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes``
                attribute data that is ignored during training and testing.
                Defaults to None.
        Returns:
            obj:`AssignResult`: The assigned result.
        """
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        num_gt = gt_bboxes.size(0)

        decoded_bboxes = pred_instances.bboxes
        pred_scores = pred_instances.scores
        if hasattr(pred_instances,'filters'):
            pred_filters = pred_instances.filters
        else:
            pred_filters = None
        priors = pred_instances.priors
        num_bboxes = decoded_bboxes.size(0)

        # assign 0 by default
        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes, ),
                                                   0,
                                                   dtype=torch.long)
        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            assigned_labels = decoded_bboxes.new_full((num_bboxes, ),
                                                      -1,
                                                      dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        prior_center = priors[:, :2]
        if isinstance(gt_bboxes, BaseBoxes):
            is_in_gts = gt_bboxes.find_inside_points(prior_center)
        else:
            # Tensor boxes will be treated as horizontal boxes by defaults
            lt_ = prior_center[:, None] - gt_bboxes[:, :2]
            rb_ = gt_bboxes[:, 2:] - prior_center[:, None]

            deltas = torch.cat([lt_, rb_], dim=-1)
            is_in_gts = deltas.min(dim=-1).values > 0

        valid_mask = is_in_gts.sum(dim=1) > 0

        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]

        if pred_filters is not None:
            valid_pred_filters = pred_filters[valid_mask]
            valid_pred_filters = valid_pred_filters.unsqueeze(1).repeat(1, num_gt, 1)
        else:
            valid_pred_filters = None

        num_valid = valid_decoded_bbox.size(0)

        if num_valid == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes, ))
            assigned_labels = decoded_bboxes.new_full((num_bboxes, ),
                                                      -1,
                                                      dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
        if hasattr(gt_instances, 'masks'):
            gt_center = center_of_mass(gt_instances.masks, eps=EPS)
        elif isinstance(gt_bboxes, BaseBoxes):
            gt_center = gt_bboxes.centers
        else:
            # Tensor boxes will be treated as horizontal boxes by defaults
            gt_center = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2.0
        valid_prior = priors[valid_mask]
        strides = valid_prior[:, 2]
        distance = (valid_prior[:, None, :2] - gt_center[None, :, :]
                    ).pow(2).sum(-1).sqrt() / strides[:, None]
        soft_center_prior = torch.pow(10, distance - self.soft_center_radius)

        pairwise_ious = self.iou_calculator(valid_decoded_bbox, gt_bboxes)

        iou_cost = -torch.log(pairwise_ious + EPS) * self.iou_weight

        if self.use_iou_cut:
            iou_cost[pairwise_ious.clone() < self.iou_cut_threshold] = float('nan')

        gt_onehot_label = (
            F.one_hot(gt_labels.to(torch.int64),
                      pred_scores.shape[-1]).float().unsqueeze(0).repeat(
                          num_valid, 1, 1))
        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)

        soft_label = gt_onehot_label * pairwise_ious[..., None]
        if valid_pred_filters is not None:
            scale_factor = soft_label - valid_pred_scores.sigmoid() * valid_pred_filters.sigmoid()
            soft_cls_cost = F.binary_cross_entropy_with_logits(
                valid_pred_scores, soft_label,
                reduction='none') * scale_factor.abs().pow(2.0)
        else:
            scale_factor = soft_label - valid_pred_scores.sigmoid()
            soft_cls_cost = F.binary_cross_entropy_with_logits(
                valid_pred_scores, soft_label,
                reduction='none') * scale_factor.abs().pow(2.0)
            
        soft_cls_cost = soft_cls_cost.sum(dim=-1)

        cost_matrix = soft_cls_cost + iou_cost + soft_center_prior
        
        if self.use_wh_metric:
            tensor_gt_bbox = get_box_tensor(gt_bboxes)
            wh_rate_gt = (tensor_gt_bbox[..., 2] / tensor_gt_bbox[..., 3]).unsqueeze(1)
            wh_rate_pred = (valid_decoded_bbox[..., 2] / valid_decoded_bbox[..., 3]).unsqueeze(1)
            diff_wh_rate = (wh_rate_pred[:, None, :] - wh_rate_gt[None, :, :]).abs().squeeze(-1)
            cost_matrix += (diff_wh_rate * self.wh_metric_weight)

        matched_pred_ious, matched_gt_inds = self.dynamic_k_matching(
            cost_matrix, pairwise_ious, num_gt, valid_mask)

        # convert to AssignResult format
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
        max_overlaps = assigned_gt_inds.new_full((num_bboxes, ),
                                                 -INF,
                                                 dtype=torch.float32)
        max_overlaps[valid_mask] = matched_pred_ious
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def dynamic_k_matching(self, cost: Tensor, pairwise_ious: Tensor,
                           num_gt: int,
                           valid_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Use IoU and matching cost to calculate the dynamic top-k positive
        targets. Same as SimOTA.

        Args:
            cost (Tensor): Cost matrix.
            pairwise_ious (Tensor): Pairwise iou matrix.
            num_gt (int): Number of gt.
            valid_mask (Tensor): Mask for valid bboxes.

        Returns:
            tuple: matched ious and gt indexes.
        """
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        # select candidate topk ious for dynamic-k calculation
        candidate_topk = min(self.topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)

        # calculate dynamic k for each gt
        if self.max_k is None:
            dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        else:
            dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1, max=int(self.max_k))

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[:, gt_idx], k=dynamic_ks[gt_idx], largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            cost_min, cost_argmin = torch.min(
                cost[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1
        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(1) > 0
        valid_mask[valid_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        matched_pred_ious = (matching_matrix *
                             pairwise_ious).sum(1)[fg_mask_inboxes]
        return matched_pred_ious, matched_gt_inds
