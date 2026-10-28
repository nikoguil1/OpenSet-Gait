import torch
import torch.nn.functional as F

from .base import BaseLoss, gather_and_scale_wrapper

#  NGuil: Se ha cambiado la funcion del Loss por la propuesta en el paper "Batch Hard Contrastive Loss and Its Application" en la ecuación 5. 

class BHCLossV3(BaseLoss):
    def __init__(self, m=8, loss_term_weight=1.0):
        super(BHCLossV3, self).__init__(loss_term_weight)
        self.m_init = m
        self.m = m
        self.iteration = 0

    @gather_and_scale_wrapper
    def forward(self, embeddings, labels):
        self.iteration += 1
        num_increments = self.iteration // 15000
        self.m = self.m_init + (2 * num_increments)

        # embeddings: [n, c, p], label: [n]
        embeddings = embeddings.permute(
            2, 0, 1).contiguous().float()  # [n, c, p] -> [p, n, c]

        ref_embed, ref_label = embeddings, labels
        # Computed using actual Euclidean distance (with sqrt) to apply Grouped BHCn accurately
        dist = self.ComputeDistance(embeddings, ref_embed)  # [p, n, n]
        
        ap_dist, an_dist = self.Convert2Triplets(labels, ref_label, dist)

        # ==========================================================
        # Equation (6) Grouped BHCn Implementation
        # ==========================================================
        
        # --- Positive Pairs (+) ---
        # 1. Evaluate LCn: For positive pairs, LCn is the Euclidean distance directly
        ap_loss = ap_dist  # [p, n, K, 1]

        # 2. Average over all G groups (dim=0 corresponds to group 'p')
        ap_loss_mean = ap_loss.mean(dim=0)  # shape: [n, K, 1]
        
        # 3. Find the "Hardest Positive"
        # max_{k,l} -> maximum loss among the K samples of the same identity
        d_p = torch.max(ap_loss_mean, dim=1)[0].squeeze(-1)  # shape: [n]

        # 4. Final Square ( ... )^2:
        l_pull = d_p ** 2  

        # --- Negative Pairs (-) ---
        # 1. Evaluate LCn: For negative pairs, penalize distances below margin 'm'
        an_loss = F.relu(self.m - an_dist)  # [p, n, 1, n-K]

        # 2. Average over all G groups (dim=0 corresponds to group 'p')
        an_loss_mean = an_loss.mean(dim=0)  # shape: [n, 1, n-K]
        
        # 3. Find the "Hardest Negative"
        # max_{k,l} -> maximum loss (corresponds to the easiest to confuse negative pair)
        d_n = torch.max(an_loss_mean, dim=2)[0].squeeze(1)   # shape: [n]

        # 4. Final Square ( ... )^2:
        l_push = d_n ** 2  
        
        # Total loss component for BHC
        loss = l_pull + l_push # shape: [n]

        hard_loss = torch.max(loss)
        loss_avg, loss_num = self.AvgNonZeroReducer(loss)

        self.info.update({
            'loss_pull_avgV3': l_pull.mean().detach().clone(),
            'loss_push_avgV3': l_push.mean().detach().clone(),
            'lossV3': loss_avg.detach().clone(),
            'hard_lossV3': hard_loss.detach().clone(),
            'loss_numV3': loss_num.detach().clone(),
            'm_value': torch.tensor(self.m, device=embeddings.device, dtype=torch.float32),
            'd_p_meanV3': d_p.mean().detach().clone(),
            'd_n_meanV3': d_n.mean().detach().clone()
        })

        return loss_avg, self.info

    def AvgNonZeroReducer(self, loss):
        eps = 1.0e-9
        loss_sum = loss.sum()
        loss_num = (loss != 0).sum().float()

        loss_avg = loss_sum / (loss_num + eps)
        if loss_num == 0:
            loss_avg = torch.zeros_like(loss_avg)
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
        # We KEEP the sqrt to correctly implement the grouped averaging over Euclidean distances
        # Added a small epsilon (1e-12) to prevent NaN gradients exactly at distance 0
        dist = torch.sqrt(F.relu(dist) + 1e-12)  # [p, n_x, n_y]
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
