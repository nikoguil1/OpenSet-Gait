'''
Modifed fromhttps://github.com/BNU-IVC/FastPoseGait/blob/main/fastposegait/modeling/losses/supconloss.py
'''

import torch.nn as nn
import torch
from .base import BaseLoss, gather_and_scale_wrapper


class SupConLoss_Re(BaseLoss):
    def __init__(self, temperature=0.01):
        super(SupConLoss_Re, self).__init__()
        self.train_loss = SupConLoss(temperature=temperature)

    @gather_and_scale_wrapper
    def forward(self, embeddings, labels=None, mask=None):
        loss = self.train_loss(embeddings, labels)
        
        info = {'loss': loss.detach().clone()}
        if hasattr(self.train_loss, 'monitor_info'):
            for k, v in self.train_loss.monitor_info.items():
                info[k] = v.detach().clone() if hasattr(v, 'detach') else v
                
        self.info.update(info)
        return loss, self.info


class SupConLoss_Lp(BaseLoss):
    def __init__(self, temperature=0.01):
        super(SupConLoss_Lp, self).__init__()
        self.train_loss = SupConLoss(
            temperature=temperature, base_temperature=temperature, reduce_zero=True, p=2)

    @gather_and_scale_wrapper
    def forward(self, embeddings, labels=None, mask=None):
        loss = self.train_loss(embeddings.unsqueeze(1), labels)
        
        info = {'loss': loss.detach().clone()}
        if hasattr(self.train_loss, 'monitor_info'):
            for k, v in self.train_loss.monitor_info.items():
                info[k] = v.detach().clone() if hasattr(v, 'detach') else v
                
        self.info.update(info)
        return loss, self.info


class SupConLoss_Euc(BaseLoss):
    def __init__(self, temperature=0.01):
        super(SupConLoss_Euc, self).__init__()
        # Un-normalized Euclidean distance (Softmax across raw space distances)
        self.train_loss = SupConLoss(
            temperature=temperature, base_temperature=temperature, reduce_zero=True, p=2, normalize=False)

    @gather_and_scale_wrapper
    def forward(self, embeddings, labels=None, mask=None):
        loss = self.train_loss(embeddings.unsqueeze(1), labels)
        
        info = {'loss': loss.detach().clone()}
        if hasattr(self.train_loss, 'monitor_info'):
            for k, v in self.train_loss.monitor_info.items():
                info[k] = v.detach().clone() if hasattr(v, 'detach') else v
                
        self.info.update(info)
        return loss, self.info


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.01, contrast_mode='all',
                 base_temperature=0.07, reduce_zero=False, p=None, normalize=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.reduce_zero = reduce_zero
        self.p = p
        self.normalize = normalize

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute distance mat
        if self.p is None:
            mat = torch.matmul(
                anchor_feature, contrast_feature.T)
        else:
            if self.normalize:
                anchor_feature = torch.nn.functional.normalize(
                    anchor_feature, p=self.p, dim=1)
                contrast_feature = torch.nn.functional.normalize(
                    contrast_feature, p=self.p, dim=1)
            
            # Manual Euclidean calculation to avoid torch.cdist NaN gradient at 0
            x_sq = torch.sum(anchor_feature ** 2, dim=1, keepdim=True)
            y_sq = torch.sum(contrast_feature ** 2, dim=1, keepdim=True)
            dist_sq = x_sq + y_sq.T - 2 * torch.matmul(anchor_feature, contrast_feature.T)
            mat = -torch.sqrt(torch.clamp(dist_sq, min=1e-12))
            
        mat = mat / self.temperature

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # for numerical stability (ignore self-contrast in max finding)
        mat_masked = mat.clone()
        mat_masked[logits_mask == 0] = -1e9
        logits_max, _ = torch.max(mat_masked, dim=1, keepdim=True)
        logits = mat - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / \
            (mask.sum(1)+torch.finfo(mat.dtype).tiny)
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        if self.reduce_zero:
            loss = loss[loss > 0]

        # Extract monitoring metrics for Intra-class and Inter-class separation
        with torch.no_grad():
            raw_dist = -mat * self.temperature if self.p is not None else mat * self.temperature
            neg_mask = logits_mask - mask
            ap_dist = (raw_dist * mask).sum() / (mask.sum() + 1e-9)
            an_dist = (raw_dist * neg_mask).sum() / (neg_mask.sum() + 1e-9)
            
            prob = exp_logits / (exp_logits.sum(1, keepdim=True) + 1e-9)
            ap_prob = (prob * mask).sum() / (mask.sum() + 1e-9)
            
            self.monitor_info = {
                'ap_dist': ap_dist,
                'an_dist': an_dist,
                'ap_prob': ap_prob,
                'temperature': self.temperature
            }

        return loss.mean()
