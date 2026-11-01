"""CPU tests for DeepGaitV2 + OpenSetLoss trained with the batches of Su et al., ECCV 2024
(configs/deepgaitv2/DeepGaitV2_Gait3D_OpenSetLoss_SuSampler.yaml): TripletNoRepeatSampler batches
(subjects with 1-4 sequences), TripletLossSu and one clip length of 10-50 frames per batch.

Run: python tests/test_deepgaitv2_su_sampler.py   (or pytest tests/)
"""
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'opengait'))
from modeling import losses  # noqa: E402
from modeling.models.deepgaitv2 import DeepGaitV2  # noqa: E402

CFG_DIR = os.path.join(os.path.dirname(__file__), '..', 'configs', 'deepgaitv2')
NEW_CFG = os.path.join(CFG_DIR, 'DeepGaitV2_Gait3D_OpenSetLoss_SuSampler.yaml')
OLD_CFG = os.path.join(CFG_DIR, 'DeepGaitV2_Gait3D_OpenSetLoss.yaml')


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg):
    """DeepGaitV2.build_network without BaseModel.__init__ (no data loaders or DDP)."""
    model = DeepGaitV2.__new__(DeepGaitV2)
    torch.nn.Module.__init__(model)
    model.build_network(cfg['model_cfg'])
    model.init_parameters()
    return model


def test_unbalanced_batch_variable_length():
    cfg = load_cfg(NEW_CFG)
    loss_fns = {}
    for c in cfg['loss_cfg']:
        c = dict(c)
        prefix = c.pop('log_prefix')
        c.pop('log_accuracy', None)
        loss_fns[prefix] = getattr(losses, c.pop('type'))(**c)
    torch.manual_seed(0)
    model = build_model(cfg).train()
    labs = torch.tensor([0, 0, 0, 0, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 6])
    for s in (10, 50):  # frames_num_per_batch: one length for the whole batch, seqL = None
        model.zero_grad()
        feat = model(([torch.rand(len(labs), s, 64, 44)], labs, None, None, None))['training_feat']
        assert feat['triplet']['embeddings'].shape == (16, 256, 16)
        total = 0
        for prefix, fn in loss_fns.items():
            forward = type(fn).forward
            loss = getattr(forward, '__wrapped__', forward)(fn, **feat[prefix])[0]  # skip the DDP gather
            assert torch.isfinite(loss).all(), prefix
            total = total + loss.mean()
        total.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)


def test_config_differs_only_in_batches():
    old, new = load_cfg(OLD_CFG), load_cfg(NEW_CFG)
    assert [l['type'] for l in new['loss_cfg']] == ['TripletLossSu', 'CrossEntropyLoss', 'OpenSetLoss']
    expected = dict(old['trainer_cfg']['sampler'], type='TripletNoRepeatSampler', sample_type='unfixed_ordered',
                    frames_num_min=10, frames_num_max=50, frames_num_per_batch=True, frames_skip_num=0)
    expected.pop('frames_num_fixed')
    assert new['trainer_cfg']['sampler'] == expected
    for c in (old, new):
        c['loss_cfg'][0].pop('type')
        c['trainer_cfg'].pop('sampler')
        c['trainer_cfg'].pop('save_name')
        c['evaluator_cfg'].pop('save_name')
    assert old == new


if __name__ == '__main__':
    tests = [v for k, v in dict(globals()).items() if k.startswith('test_')]
    for t in tests:
        t()
        print(f'OK  {t.__name__}')
    print(f'{len(tests)} tests OK')
