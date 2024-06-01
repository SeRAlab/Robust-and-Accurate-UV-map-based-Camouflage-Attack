# # Loss functions

# import torch
# import torch.nn as nn

# from utils.general import bbox_iou
# from utils.torch_utils import is_parallel


# def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
#     # return positive, negative label smoothing BCE targets
#     return 1.0 - 0.5 * eps, 0.5 * eps


# class FocalLoss(nn.Module):
#     # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
#     def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
#         super(FocalLoss, self).__init__()
#         self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = loss_fcn.reduction
#         self.loss_fcn.reduction = 'none'  # required to apply FL to each element

#     def forward(self, pred, true):
#         loss = self.loss_fcn(pred, true)
#         # p_t = torch.exp(-loss)
#         # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

#         # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
#         pred_prob = torch.sigmoid(pred)  # prob from logits
#         p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
#         alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
#         modulating_factor = (1.0 - p_t) ** self.gamma
#         loss *= alpha_factor * modulating_factor

#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:  # 'none'
#             return loss


# class ComputeLoss:
#     # Compute losses
#     def __init__(self, model, autobalance=False):
#         super(ComputeLoss, self).__init__()
#         device = next(model.parameters()).device  # get model device  next(model.parameters())可以获得模型的第一个参数
#         h = model.hyp  # hyperparameters 

#         # Define criteria
#         BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device)) #定义了二元交叉熵损失函数， cls_pw是分类问题的正样本权重（用来控制不均匀数据）只是定义还没有用
#         BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))#定义了二元交叉熵损失函数， cls_pw是obj问题的正样本权重（用来控制不均匀数据）只是定义还没有用

#         # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
#         self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

#         # Focal loss
#         g = h['fl_gamma']  # focal loss gamma
#         if g > 0:
#             BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g) #只是定义还没有用 用了正类权重

#         det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module 如果是并行模型，那么就要靠模型的model.module就是指定模型的主模型，model.module.model就是指模型的分层结构，最后一层就是检测层
#         self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
#         self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
#         self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
#         for k in 'na', 'nc', 'nl', 'anchors':
#             setattr(self, k, getattr(det, k))

#     def __call__(self, p, targets):  # predictions, targets, model
#         device = targets.device
#         lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
#         tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

#         # Losses
#         for i, pi in enumerate(p):  # layer index, layer predictions
#             # b [0,0,0]
#             # a [0, 1, 2]
#             # gj [10, 10, 10]
#             # gi [10, 10, 10]
#             b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
#             tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

#             n = b.shape[0]  # number of targets
#             if n:
#                 ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets  目标对应的预测子集

#                 # Regression
#                 pxy = ps[:, :2].sigmoid() * 2. - 0.5
#                 pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
#                 pbox = torch.cat((pxy, pwh), 1)  # predicted box
#                 iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
#                 # lbox += (1.0 - iou).mean()  # original iou loss
#                 lbox += iou.mean()  # adversarial iou loss

#                 tobj[b, a, gj, gi] = 1.0 # (1.0 - self.gr) + self.gr * (1-iou).detach().clamp(0).type(tobj.dtype)  # iou ratio

#                 # Classification
#                 if self.nc > 1:  # cls loss (only if multiple classes)
#                     t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
#                     t[range(n), tcls[i]] = self.cp
#                     lcls += torch.max(torch.mean(ps[:, 5:] * t, dim=0))  #

#             obji = self.BCEobj(pi[..., 4], tobj)
#             lobj += obji * self.balance[i]  # obj loss
#             if self.autobalance:
#                 self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

#         if self.autobalance:
#             self.balance = [x / self.balance[self.ssi] for x in self.balance]
#         lbox *= self.hyp['box']
#         lobj *= self.hyp['obj']
#         lcls *= self.hyp['cls']
#         bs = tobj.shape[0]  # batch size

#         loss = lbox + lobj + lcls
#         return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

#     def build_targets(self, p, targets):
#         # Build targets for compute_loss(), input targets(image,class,x,y,w,h) torch.Size([nt, 6]) (6分别是第几张图类别，四个坐标)
#         na, nt = self.na, targets.shape[0]  # number of anchors, targets
#         tcls, tbox, indices, anch = [], [], [], []
# #         "normalized to gridspace gain" 是一个相对于网格空间增益进行归一化的概念。

# # 在目标检测任务中，通常将输入图像划分为一个个网格（grid），每个网格被用作特征提取和目标预测的基本单位。在一些检测模型中，目标的预测可能会受到不同尺度的网格的影响，而这些尺度之间的比例关系可能需要进行统一或归一化。

# # "normalized to gridspace gain" 可能是指通过对每个网格预测的结果进行归一化，以保持不同尺度的预测对目标检测结果的贡献相对均衡。这种归一化可以通过调整目标检测模型中的某些权重或参数来实现。

# # 具体来说，"normalized to gridspace gain" 可能是指在目标检测模型中对每个网格预测结果进行缩放或加权，以便更好地平衡不同尺度的目标检测结果。这样可以确保在整个图像上进行目标检测时，不同尺度的目标都能得到适当的关注和处理。
#         gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
#         ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt) arange（）生成一个0到na-1的张量
#         targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices ， (targets.repeat(na, 1, 1)变成 targets(image,class,x,y,w,h) cat是按照维度拼接，最后的结果就是torch.size(na,nt,7) 比原来的6多了一个可能锚点权重的感觉

#         g = 0.5  # bias
#         off = torch.tensor([[0, 0],
#                             # [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
#                             # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
#                             ], device=targets.device).float() * g  # offsets

#         for i in range(self.nl):
#             anchors = self.anchors[i]
#             # tensor([ 1.,  1., 80., 80., 80., 80.,  1.], device='cuda:0')
#             gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

#             # Match targets to anchors
#             t = targets * gain
#             if nt:
#                 # Matches
#                 r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
#                 j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
#                 # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
#                 t = t[j]  # filter

#                 # Offsets
#                 gxy = t[:, 2:4]  # grid xy
#                 gxi = gain[[2, 3]] - gxy  # inverse
#                 j, k = ((gxy % 1. < g) & (gxy > 1.)).T
#                 l, m = ((gxi % 1. < g) & (gxi > 1.)).T
#                 j = torch.stack((torch.ones_like(j),))
#                 t = t.repeat((off.shape[0], 1, 1))[j]
#                 offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
#             else:
#                 t = targets[0]
#                 offsets = 0

#             # Define
#             b, c = t[:, :2].long().T  # image, class
#             gxy = t[:, 2:4]  # grid xy
#             gwh = t[:, 4:6]  # grid wh
#             gij = (gxy - offsets).long()
#             gi, gj = gij.T  # grid xy indices

#             # Append
#             a = t[:, 6].long()  # anchor indices
#             indices.append((b, a, gj.clamp_(0, gain[3].round().int() - 1), gi.clamp_(0, gain[2].round().int() - 1)))  # image, anchor, grid indices
#             tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
#             anch.append(anchors[a])  # anchors
#             tcls.append(c)  # class

#         return tcls, tbox, indices, anch


# Loss functions

import torch
import torch.nn as nn

from utils.general import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets 为了避免过拟合 而去平滑分类结果，这里不用平滑

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma 这里也是0 不需要进行和这个操作
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7 和不同尺寸的特征图的obj权重 nl是检测器特征层数 {3: [4.0, 1.0, 0.4]}表示一个包含一个键值对的字典，其中键为3，值为列表[4.0, 1.0, 0.4]。.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])表示从该字典中获取特定键det.nl的值，如果该键不存在，则返回默认值[4.0, 1.0, 0.25, 0.06, .02]
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index 自动平衡算法的起始位置 这里也不采用
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors': #锚框数量、类别数量、特征图数量和锚框尺寸[3,3,2]一个grid 3个特征图，共3种，长宽
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets # target[1,6] 分别是在图片中的第几个，类别，x，y，框的坐标

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            # b [0,0,0]
            # a [0, 1, 2]
            # gj [10, 10, 10]
            # gi [10, 10, 10]
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets  在第i个特征图找到ground truth对应的猫框的输出

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                # lbox += (1.0 - iou).mean()  # original iou loss
                lbox += iou.mean()  # adversarial iou loss

                tobj[b, a, gj, gi] = 0.0 # (1.0 - self.gr) + self.gr * (1-iou).detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification

                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    # if torch.max(torch.mean(ps[:, 5:] * t, dim=0))!=0:
                    #     print(all)
                    # class31231 = ps[:, 5:].detach()
                    # all=torch.sum(ps[:, 5:])
                    sigmoid=nn.Sigmoid()
                    # class31231=class31231.sigmoid()
                    # allnew=torch.sum(class31231)
                    # car_confidence=class31231[:,2]
                    # truck_confidence=ps[:,7]
                    # bus_confidence = ps[:, 5]
                    normalclass=ps[:, 5:].sigmoid()
                    # lcls += self.BCEcls(ps[:, 5:], t)

                    lcls += torch.max(torch.mean(normalclass * t, dim=0))  #

                #obji = self.BCEobj(ps[:, 4], torch.full_like(ps[:, 4], 0, device=device))
                obji = self.BCEobj(pi[..., 4], tobj)
                lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets，一个图中有几个框
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain，为框
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
        # 为扩充标记的box添加偏置，具体扩充规则为在下边
        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            # [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl): #对每个尺寸的特征图都这样做
            anchors,shape = self.anchors[i],p[i].shape
            # tensor([ 1.,  1., 80., 80., 80., 80.,  1.], device='cuda:0')
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio 一个label在一个尺寸特征图下不同猫框的比例
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare 比较是否大于猫框和检测结果是否阈值
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j),))
                t = t.repeat((off.shape[0], 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box (实际上是框内的偏移量（格子数）和宽高（格子数）)
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch