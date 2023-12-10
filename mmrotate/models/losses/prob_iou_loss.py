import torch.nn as nn
from mmrotate.registry import MODELS
import torch
from .rotated_iou_loss import rotated_iou_loss

def gbb_form(boxes):
    return torch.cat((boxes[:, :2], torch.pow(boxes[:, 2:4], 2) / 12.0, boxes[:, 4:]), 1)


def rotated_form(a_, b_, angles):
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    a = a_ * torch.pow(cos_a, 2) + b_ * torch.pow(sin_a, 2)
    b = a_ * torch.pow(sin_a, 2) + b_ * torch.pow(cos_a, 2)
    c = (a_ - b_) * cos_a * sin_a
    return a, b, c

def probiou_loss(pred, target, eps=1e-3, mode='l1', deform=True):

    """
        pred    -> a matrix [N,5](x,y,w,h,angle - in radians) containing ours predicted box ;in case of HBB angle == 0
        target  -> a matrix [N,5](x,y,w,h,angle - in radians) containing ours target    box ;in case of HBB angle == 0
        eps     -> threshold to avoid infinite values
        mode    -> ('l1' in [0,1] or 'l2' in [0,inf]) metrics according our paper
        deform  -> Is it deformed on an official basis. Default to Ture(Refer to FCOSR).

        The function of formula deformation is to reduce calculation errors and accelerate calculation speed

    """

    gbboxes1 = gbb_form(pred)
    gbboxes2 = gbb_form(target)

    x1, y1, a1_, b1_, c1_ = gbboxes1[:, 0], gbboxes1[:, 1], gbboxes1[:, 2], gbboxes1[:, 3], gbboxes1[:, 4]
    x2, y2, a2_, b2_, c2_ = gbboxes2[:, 0], gbboxes2[:, 1], gbboxes2[:, 2], gbboxes2[:, 3], gbboxes2[:, 4]

    a1, b1, c1 = rotated_form(a1_, b1_, c1_)
    a2, b2, c2 = rotated_form(a2_, b2_, c2_)

    if deform:
        # come from FCOSR
        t1 = 0.25 * ((a1 + a2) * (torch.pow(y1 - y2, 2)) + (b1 + b2) * (torch.pow(x1 - x2, 2))) + \
             0.5 * ((c1+c2)*(x2-x1)*(y1-y2))
        t2 = (a1 + a2) * (b1 + b2) - torch.pow(c1 + c2, 2)
        t3_ = (a1 * b1 - c1 * c1) * (a2 * b2 - c2 * c2)
        t3 = 0.5 * torch.log(t2 / (4 * torch.sqrt(torch.relu(t3_)) + eps))
        B_d = (t1 / t2) + t3
    else:
        # come from Official
        t1 = (((a1 + a2) * (torch.pow(y1 - y2, 2)) + (b1 + b2) * (torch.pow(x1 - x2, 2))) /
              ((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2)) + eps)) * 0.25
        t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2)) + eps)) * 0.5
        t3 = torch.log(((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2))) /
             (4 * torch.sqrt((a1 * b1 - torch.pow(c1, 2)) * (a2 * b2 - torch.pow(c2, 2))) + eps) + eps) * 0.5
        B_d = t1 + t2 + t3

    B_d = torch.clamp(B_d, eps, 100.0)
    l1 = torch.sqrt(1.0 - torch.exp(-B_d) + eps)
    l_i = torch.pow(l1, 2.0)
    l2 = -torch.log(1.0 - l_i + eps)

    if mode == 'l1':
        loss = l1
    if mode == 'l2':
        loss = l2

    return loss

def probiou_loss2(pred, target, eps=1e-3, mode='l1', deform=True):

    """
        pred    -> a matrix [N,5](x,y,w,h,angle - in radians) containing ours predicted box ;in case of HBB angle == 0
        target  -> a matrix [N,5](x,y,w,h,angle - in radians) containing ours target    box ;in case of HBB angle == 0
        eps     -> threshold to avoid infinite values
        mode    -> ('l1' in [0,1] or 'l2' in [0,inf]) metrics according our paper
        deform  -> Is it deformed on an official basis. Default to Ture(Refer to FCOSR).

    """

    assert deform is True
    assert mode == 'l1'

    gbboxes1 = gbb_form(pred)
    gbboxes2 = gbb_form(target)

    x1, y1, a1_, b1_, c1_ = gbboxes1[:, 0], gbboxes1[:, 1], gbboxes1[:, 2], gbboxes1[:, 3], gbboxes1[:, 4]
    x2, y2, a2_, b2_, c2_ = gbboxes2[:, 0], gbboxes2[:, 1], gbboxes2[:, 2], gbboxes2[:, 3], gbboxes2[:, 4]

    a1, b1, c1 = rotated_form(a1_, b1_, c1_)
    a2, b2, c2 = rotated_form(a2_, b2_, c2_)


    t1 = 0.25 * ((a1 + a2) * (torch.pow(y1 - y2, 2)) + (b1 + b2) * (torch.pow(x1 - x2, 2))) + \
         0.50 * ((c1+c2)*(x2-x1)*(y1-y2))
    t2 = (a1 + a2) * (b1 + b2) - torch.pow(c1 + c2, 2)
    t3_ = (a1 * b1 - c1 * c1) * (a2 * b2 - c2 * c2)
    t3 = 0.5 * torch.log(t2 / (4 * torch.sqrt(torch.relu(t3_)) + eps))
    B_d = (t1 / t2) + t3

    B_d = torch.clamp(B_d, eps, 100.0)

    l1 = torch.sqrt(1.0 - torch.exp(-B_d))

    return l1

@MODELS.register_module()
class ProbiouLoss(nn.Module):
    # eps=1e-3
    def __init__(self, mode='l1', eps=1e-3, loss_weight=1.0, deform=True):
        super(ProbiouLoss, self).__init__()
        self.mode = mode
        self.eps = eps
        self.loss_weight = loss_weight
        self.deform = deform
    def forward(self, loc_p, loc_t, weight, avg_factor=None):
        if avg_factor is None:
            avg_factor = 1.0
        loss = self.loss_weight * torch.sum(probiou_loss(loc_p, loc_t, self.eps, self.mode, deform=self.deform) *
                                            weight)[None] / avg_factor
        return loss

@MODELS.register_module()
class ProbiouLoss2(nn.Module):
    # eps=1e-3
    def __init__(self, mode='l1', eps=1e-3, loss_weight=1.0, deform=True):
        super(ProbiouLoss2, self).__init__()
        self.mode = mode
        self.eps = eps
        self.loss_weight = loss_weight
        self.deform = deform
    def forward(self, loc_p, loc_t, weight, avg_factor=None):
        if avg_factor is None:
            avg_factor = 1.0
        loss = self.loss_weight * torch.sum(probiou_loss2(loc_p, loc_t, self.eps, self.mode, deform=self.deform) *
                                            weight)[None] / avg_factor
        return loss

@MODELS.register_module()
class ProbiouRiouLoss(nn.Module):
    '''
    ProbiouLoss and RotatedIoULoss

    mode:
    1. avg: loss = (p_loss + r_loss) / 2.0
    2. multi: loss = torch.sqrt(p_loss * r_loss)
    3. compensate: delta = torch.abs(r_loss-p_loss) / (r_loss + p_loss)
                   loss = delta * p_loss + (1 - delta) * r_loss

    '''
    def __init__(self, mode='avg',loss_weight=1.0, prob_eps=1e-3, use_pv2=False):
        super(ProbiouRiouLoss, self).__init__()

        assert mode in ['avg', 'multi', 'compensate']

        self.mode = mode
        self.loss_weight = loss_weight
        self.prob_eps = prob_eps
        self.use_probiou_loss2 = use_pv2

    def forward(self, loc_p, loc_t, weight, avg_factor=None):
        if avg_factor is None:
            avg_factor = 1.0

        if self.use_probiou_loss2:
            p_loss = probiou_loss2(loc_p, 
                                loc_t, 
                                eps=self.prob_eps, 
                                mode='l1', 
                                deform=True)
        else:
            p_loss = probiou_loss(loc_p, 
                                loc_t, 
                                eps=self.prob_eps, 
                                mode='l1', 
                                deform=True)
        r_loss = rotated_iou_loss(
                                loc_p,
                                loc_t,
                                mode="linear",
                                eps=1e-6)
        
        if self.mode == 'avg':
            loss = (p_loss + r_loss) / 2.0
        elif self.mode == 'multi':
            loss = torch.sqrt(torch.abs(p_loss * r_loss))
        elif self.mode == 'compensate':
            # detach()
            delta = torch.abs(r_loss-p_loss) / (r_loss + p_loss)
            delta = delta.detach()
            loss = delta * p_loss + (1 - delta) * r_loss

        loss = self.loss_weight * torch.sum(loss * weight)[None] / avg_factor

        return loss


def main():
    p = torch.rand(8, 5)
    t = torch.rand(8, 5)
    t = p
    loss = probiou_loss(p, t, deform=False)
    produce_loss = torch.mean(loss)
    print(produce_loss.item())

if __name__ == '__main__':
    main()