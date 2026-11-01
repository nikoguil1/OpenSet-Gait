"""Baseline3D of Su et al., "Open-Set Biometrics: Beyond Good Closed-Set Models", ECCV 2024.

Ported from https://github.com/prevso1088/open-set-biometrics (commit 76ea429,
opengait/modeling/models/baseline_3d.py and baseline_3d_open_set.py): a 3D backbone
([n, c, s, h, w] output, e.g. SwinGaitSu) + temporal max pooling + HPP + SeparateFCs + BNNecks.
Their two models only differ in the 'open_set' training feature; here it is the
model_cfg flag open_set_loss, as in DeepGaitV2.
"""
import torch

from ..base_model import BaseModel
from ..modules import HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks


class Baseline3D(BaseModel):

    def build_network(self, model_cfg):
        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])
        # feed the embeddings to an OpenSetLoss with log_prefix 'open_set' (Su et al., ECCV 2024)
        self.open_set_loss = model_cfg.get('open_set_loss', False)

    def forward(self, inputs):
        ipts, labs, typs, vies, seqL = inputs

        sils = ipts[0]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)  # [n, 1, s, h, w]

        del ipts
        outs = self.Backbone(sils)  # [n, c, s, h, w]

        # Temporal Pooling, TP
        outs = self.TP(outs, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        embed_1 = self.FCs(feat)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]
        embed = embed_1

        n, _, s, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.view(n*s, -1, h, w)
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        if self.open_set_loss:
            retval['training_feat']['open_set'] = {'embeddings': embed_1, 'labels': labs}
        return retval
