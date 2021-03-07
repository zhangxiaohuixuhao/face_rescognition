import torch
from torch.autograd import Function
from torch.nn.modules.distance import PairwiseDistance


class TripletLoss(Function):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(2) # to calculate distance
    def forward(self, anchor, positive, negative):
        # [在triplet_loss.py 的 13行位置处，填写代码，完成triplet loss的计算]
        # 下面几步已经实现了损失的计算因此对于填空是希望计算哪一部分，并不明确。
        pos_dist=self.pdist.forward(anchor,positive) # distance of anchor and positive
        neg_dist=self.pdist.forward(anchor,negative) # distance of anchor and negative
        hinge_dist = torch.clamp(self.margin + pos_dist - neg_dist, min=0.0) # ensure loss is no less than zero
        loss = torch.mean(hinge_dist)
        return loss
