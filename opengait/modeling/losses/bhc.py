import torch
import torch.nn.functional as F

from .base import BaseLoss, gather_and_scale_wrapper


class BHCLoss(BaseLoss):
    def __init__(self, m1=0.2, m2=0.5, loss_term_weight=1.0):
        super(BHCLoss, self).__init__(loss_term_weight)
        self.m1 = m1
        self.m2_init = m2
        self.m2 = m2
        self.iteration = 0

    @gather_and_scale_wrapper
    def forward(self, embeddings, labels):
        self.iteration += 1
        num_doublings = self.iteration // 20000
        self.m2 = self.m2_init * (2 ** num_doublings)

        # embeddings: [n, c, p], label: [n]
        embeddings = embeddings.permute(
            2, 0, 1).contiguous().float()  # [n, c, p] -> [p, n, c]

        ref_embed, ref_label = embeddings, labels
        dist = self.ComputeDistance(embeddings, ref_embed)  # [p, n, n]
        mean_dist = dist.mean((1, 2))  # [p]
        ap_dist, an_dist = self.Convert2Triplets(labels, ref_label, dist)

        # Hard Positive: Max distance to the same identity
        d_p = torch.max(ap_dist, dim=2)[0].squeeze(-1)  # [p, n, 1] -> [p, n]
        
        # Hard Negative: Min distance to a different identity
        d_n = torch.min(an_dist, dim=3)[0].squeeze(2)   # [p, n, 1, 1] -> [p, n]

        # BHC components
        l_pull = F.relu(d_p - self.m1)
        l_push = F.relu(self.m2 - d_n)

        loss_pull_avg, loss_pull_num = self.AvgNonZeroReducer(l_pull)
        loss_push_avg, loss_push_num = self.AvgNonZeroReducer(l_push)
        
        loss = l_pull + l_push

        hard_loss = torch.max(loss, -1)[0]
        loss_avg, loss_num = self.AvgNonZeroReducer(loss)

        self.info.update({
            'loss_pull_avg': loss_pull_avg.detach().clone(),
            'loss_pull_num': loss_pull_num.detach().clone(),
            'loss_push_avg': loss_push_avg.detach().clone(),
            'loss_push_num': loss_push_num.detach().clone(),
            'm2': torch.tensor(self.m2, device=embeddings.device, dtype=torch.float32),
            'loss': loss_avg.detach().clone(),
            'hard_loss': hard_loss.detach().clone(),
            'loss_num': loss_num.detach().clone(),
            'mean_dist': mean_dist.detach().clone(),
            'd_p_mean': d_p.mean().detach().clone(),
            'd_n_mean': d_n.mean().detach().clone()
        })

        return loss_avg, self.info

    def AvgNonZeroReducer(self, loss):
        eps = 1.0e-9
        loss_sum = loss.sum(-1)
        loss_num = (loss != 0).sum(-1).float()

        loss_avg = loss_sum / (loss_num + eps)
        loss_avg[loss_num == 0] = 0
        return loss_avg, loss_num

    def ComputeDistance(self, x, y):
        """
            x: [p, n_x, c]
            y: [p, n_y, c]
        """
        x2 = torch.sum(x ** 2, -1).unsqueeze(2)  # [p, n_x, 1]
        y2 = torch.sum(y ** 2, -1).unsqueeze(1)  # [p, 1, n_y]
        inner = x.matmul(y.transpose(1, 2))  # [p, n_x, n_y]
        dist = x2 + y2 - 2 * inner
        dist = torch.sqrt(F.relu(dist))  # [p, n_x, n_y]
        return dist

    def Convert2Triplets(self, row_labels, clo_label, dist):
        """
            row_labels: tensor with size [n_r]
            clo_label : tensor with size [n_c]
        """
        matches = (row_labels.unsqueeze(1) ==
                   clo_label.unsqueeze(0)).bool()  # [n_r, n_c]
        diffenc = torch.logical_not(matches)  # [n_r, n_c]
        p, n, _ = dist.size()
        ap_dist = dist[:, matches].view(p, n, -1, 1)
        an_dist = dist[:, diffenc].view(p, n, 1, -1)
        return ap_dist, an_dist
