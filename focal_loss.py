import torch
from torch import nn
import torch.nn.functional as F


class MultiClassFocalLoss(nn.Module):

    def __init__(self, gama, weight=None, ignore_index=-100):
        self.gama = gama
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=-1)
        # 这里没有负号
        focal_logporbs = ((1 - probs) ** self.gama) * torch.log(probs)
        return F.nll_loss(focal_logporbs, targets, self.weight, ignore_index=self.ignore_index)


class MultiLabelFocalLoss(nn.Module):

    def __init__(self, gama, weight=None, ignore_index=-100, reduction="mean"):
        self.gama = gama
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = F.sigmoid(logits)
        zeros = torch.zeros_like(probs)
        pos_probs = torch.where(targets > zeros, probs, 0)
        neg_probs = torch.where(targets > zeros, zeros, targets - probs)

        loss = -(1 - pos_probs) ** self.gama * torch.log(torch.clamp(probs, 1e-8, 1.0)) - (
                    1 - neg_probs) ** self.gama * torch.log(torch.clamp(neg_probs, 1 - probs, 1.0))

        if self.reduction:
            loss = loss.mean()
        else:
            loss = loss.sum()

        return loss


# circle loss

class MultiLabelCircleLoss(nn.Module):

    def __init__(self, reduction="mean", inf=1e12):
        """Circle Loss of Multi Label
        多标签分类的交叉熵
        quota: [将“softmax+交叉熵”推广到多标签分类问题](https://spaces.ac.cn/archives/7359)
        args:
            reduction: str, Specifies the reduction to apply to the output, 输出形式.
                            eg.``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``
            inf: float, Minimum of maths, 无穷大.  eg. 1e12
        returns:
            Tensor of loss.
        """
        super(MultiLabelCircleLoss, self).__init__()
        self.reduction = reduction
        self.inf = inf

    def forward(self, logits, labels):
        batch_size, label_size = labels.shape[0:2]

        labels = labels.reshape(batch_size * label_size, -1)
        logits = logits.reshape(batch_size * label_size, -1)

        zeros = torch.zeros_like(logits[..., :1])
        # 调整正负标签logits正负号
        logits = (1-2*labels) * logits

        logits_neg = logits - labels * self.inf # 把其他labels不为0的项变成负无穷
        logits_pos = logits - (1 - labels) * self.inf # 把其他labels不为1的项变成负无穷

        logits_neg = torch.cat([logits_neg, zeros], dim=-1)
        logits_pos = torch.cat([logits_pos, zeros], dim=-1)

        neg_loss = torch.logsumexp(logits_neg, dim=-1)
        pos_loss = torch.logsumexp(logits_pos, dim=-1)
        loss = neg_loss + pos_loss
        return loss.mean() if "mean" == self.reduction else loss.sum()


label, logits = [[1, 1, 1, 1], [0, 0, 0, 1]], [[0, 1, 1, 0], [1, 0, 0, 1], ]
label, logits = torch.tensor(label).float(), torch.tensor(logits).float()
loss = MultiLabelCircleLoss()(logits, label)


