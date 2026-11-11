"""CPU tests for the SwinGait-3D port of Su et al., ECCV 2024: Baseline3D + SwinGaitSu backbone,
TripletNoRepeatSampler, per-batch frame sampling and the timm cosine scheduler.

Run: python tests/test_swingait_su.py   (or pytest tests/)
"""
import os
import sys
import tempfile

import numpy as np
import torch
import torch.distributed as dist
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'opengait'))
from data.collate_fn import CollateFn  # noqa: E402
from data.sampler import TripletNoRepeatSampler  # noqa: E402
from modeling.base_model import BaseModel  # noqa: E402
from modeling.losses.open_set import OpenSetLoss  # noqa: E402
from modeling.losses.triplet import TripletLoss  # noqa: E402
from modeling.losses.triplet_su import TripletLossSu  # noqa: E402
from modeling.models.baseline3d import Baseline3D  # noqa: E402

CFG_DIR = os.path.join(os.path.dirname(__file__), '..', 'configs', 'swingait')
OSL_CFG = os.path.join(CFG_DIR, 'SwinGait3D_Su2024_Gait3D_OpenSetLoss.yaml')
BASE_CFG = os.path.join(CFG_DIR, 'SwinGait3D_Su2024_Gait3D.yaml')


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg):
    """Baseline3D.build_network without BaseModel.__init__ (no data loaders or DDP)."""
    model = Baseline3D.__new__(Baseline3D)
    torch.nn.Module.__init__(model)
    model.build_network(cfg['model_cfg'])
    model.init_parameters()
    return model


def init_dist():
    if not dist.is_initialized():
        dist.init_process_group('gloo', init_method=f'file://{tempfile.mktemp()}', rank=0, world_size=1)


def test_forward_shapes_and_gradients():
    cfg = load_cfg(OSL_CFG)
    torch.manual_seed(0)
    model = build_model(cfg).train()
    n_ids, n_seq, s = 4, 4, 20
    sils = torch.rand(n_ids * n_seq, s, 64, 44)
    labs = torch.arange(n_ids).repeat_interleave(n_seq)
    out = model(([sils], labs, None, None, None))
    feat = out['training_feat']
    assert feat['triplet']['embeddings'].shape == (16, 512, 15)
    assert feat['softmax']['logits'].shape == (16, 3000, 15)
    assert 'open_set' in feat
    loss_fn = OpenSetLoss(num_folds=4, rank=1, score_scale=9.0, rank_scale=0.2, rank_diff_scale=9.0, num_parts=15)
    loss, _ = OpenSetLoss.forward.__wrapped__(loss_fn, **feat['open_set'])
    loss.mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_checkpointing_same_gradients():
    cfg = load_cfg(BASE_CFG)
    sils, labs = torch.rand(4, 12, 64, 44), torch.arange(2).repeat_interleave(2)
    grads = []
    for use_ckpt in (False, True):
        cfg['model_cfg']['backbone_cfg']['use_checkpoint'] = use_ckpt
        torch.manual_seed(0)
        model = build_model(cfg).train()
        torch.manual_seed(1)  # same DropPath masks
        model(([sils], labs, None, None, None))['training_feat']['triplet']['embeddings'].sum().backward()
        grads.append(torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]))
    assert torch.allclose(grads[0], grads[1], atol=1e-5)


def test_triplet_su_matches_ours_and_accepts_unbalanced():
    torch.manual_seed(0)
    emb = torch.randn(16, 8, 3)
    ours, su = TripletLoss(margin=0.2), TripletLossSu(margin=0.2)
    labs = torch.arange(4).repeat_interleave(4)  # P x K: same triplets as our TripletLoss
    a = TripletLoss.forward.__wrapped__(ours, emb, labs)[0]
    b = TripletLossSu.forward.__wrapped__(su, emb, labs)[0]
    assert torch.allclose(a, b, atol=1e-5)
    # TripletNoRepeatSampler batches: subjects with 1-4 sequences
    labs = torch.tensor([0, 0, 0, 0, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 6])
    emb.requires_grad_(True)
    loss = TripletLossSu.forward.__wrapped__(su, emb, labs)[0]
    loss.mean().backward()
    assert torch.isfinite(loss).all() and torch.isfinite(emb.grad).all()
    osl = OpenSetLoss(num_folds=4, rank=1, score_scale=9.0, rank_scale=0.2, rank_diff_scale=9.0, num_parts=3)
    loss = OpenSetLoss.forward.__wrapped__(osl, emb, labs)[0]
    assert torch.isfinite(loss).all()


def test_inference_variable_length():
    # evaluation: one whole sequence per GPU (all_ordered), any length
    model = build_model(load_cfg(BASE_CFG)).eval()
    assert not model.open_set_loss
    with torch.no_grad():
        for s in (7, 33):
            out = model(([torch.rand(1, s, 64, 44)], torch.zeros(1).long(), None, None, None))
            assert out['inference_feat']['embeddings'].shape == (1, 512, 15)
            assert 'open_set' not in out['training_feat']


def test_configs_differ_only_in_loss():
    a, b = load_cfg(BASE_CFG), load_cfg(OSL_CFG)
    assert [l['type'] for l in b['loss_cfg']] == ['TripletLossSu', 'CrossEntropyLoss', 'OpenSetLoss']
    b['loss_cfg'] = b['loss_cfg'][:2]
    b['model_cfg']['open_set_loss'] = False
    for c in (a, b):
        c['evaluator_cfg'].pop('save_name')
        c['trainer_cfg'].pop('save_name')
    assert a == b


def test_no_identification_configs_differ_only_in_detection():
    # [EXP] OpenSetLoss sin identificación: E4.1 + identification: false + det_weight
    for w in (1, 4, 10):
        a, b = load_cfg(OSL_CFG), load_cfg(os.path.join(CFG_DIR, f'SwinGait3D_Su2024_Gait3D_OpenSetLoss_NoId_w{w}.yaml'))
        osl = b['loss_cfg'][2]
        assert osl.pop('identification') is False and osl.pop('det_weight') == w
        for c in (a, b):
            c['evaluator_cfg'].pop('save_name')
            c['trainer_cfg'].pop('save_name')
        assert a == b


def test_calibrated_config_differs_only_in_idl():
    # [EXP] IDL calibrada: E4.1 + scale-invariant scores, alpha 20, gamma 30, beta 1, no own-template rank term
    a, b = load_cfg(OSL_CFG), load_cfg(os.path.join(CFG_DIR, 'SwinGait3D_Su2024_Gait3D_OpenSetLoss_Calib.yaml'))
    calib = dict(score_scale=20.0, rank_scale=1.0, rank_diff_scale=30.0, scale_invariant=True,
                 rank_self_term=False, rtm_softmax_scale=10.0)
    assert {k: b['loss_cfg'][2][k] for k in calib} == calib
    for k in calib:
        a['loss_cfg'][2].pop(k, None)
        b['loss_cfg'][2].pop(k)
    assert b['trainer_cfg'].pop('save_iter') == 5000
    a['trainer_cfg'].pop('save_iter')
    for c in (a, b):
        c['evaluator_cfg'].pop('save_name')
        c['trainer_cfg'].pop('save_name')
    assert a == b


def test_genuine_norm_picasso_config_differs_only_in_scale_and_gpus():
    # E4.6 (picasso, 8 GPUs): E4.5 + scale referenced to the genuine mean with gradient; DATASET_ROOT placeholder
    a = load_cfg(os.path.join(CFG_DIR, 'SwinGait3D_Su2024_Gait3D_OpenSetLoss_Calib.yaml'))
    b = load_cfg(os.path.join(CFG_DIR, 'SwinGait3D_Su2024_Gait3D_OpenSetLoss_GenNorm_picasso.yaml'))
    assert b['loss_cfg'][2].pop('scale_reference') == 'genuine'
    assert b['loss_cfg'][2].pop('scale_detach') is False
    assert b['data_cfg'].pop('dataset_root') == 'DATASET_ROOT'
    a['data_cfg'].pop('dataset_root')
    assert b['evaluator_cfg']['sampler'].pop('batch_size') == 8
    a['evaluator_cfg']['sampler'].pop('batch_size')
    assert b['model_cfg']['backbone_cfg'].pop('use_checkpoint') is False
    a['model_cfg']['backbone_cfg'].pop('use_checkpoint')
    assert b['trainer_cfg']['save_name'] == b['evaluator_cfg']['save_name'] == 'SwinGait3D_Su2024_OpenSetLoss_GenNorm'
    for c in (a, b):
        c['evaluator_cfg'].pop('save_name')
        c['trainer_cfg'].pop('save_name')
    assert a == b
    bs = b['trainer_cfg']['sampler']['batch_size']
    assert bs[0] * bs[1] % 8 == 0


class FakeDataset:
    def __init__(self, n_per_subject):
        self.label_set = [f'{i:04d}' for i in range(len(n_per_subject))]
        self.indices_dict, k = {}, 0
        for lab, n in zip(self.label_set, n_per_subject):
            self.indices_dict[lab] = list(range(k, k + n))
            k += n

    def __len__(self):
        return sum(len(v) for v in self.indices_dict.values())


def test_triplet_no_repeat_sampler():
    init_dist()
    rng = np.random.default_rng(0)
    ds = FakeDataset(rng.integers(1, 9, size=300).tolist())
    sampler = TripletNoRepeatSampler(ds, batch_size=[32, 4], batch_shuffle=True)
    owner = {i: lab for lab, idx in ds.indices_dict.items() for i in idx}
    it = iter(sampler)
    for _ in range(20):
        batch = next(it)
        assert len(batch) == 128 and len(set(batch)) == 128  # no repeated sequences
        counts = np.unique([owner[i] for i in batch], return_counts=True)[1]
        assert counts.max() <= 4 and len(counts) >= 32


def test_collate_one_length_per_batch():
    sample_cfg = dict(sample_type='unfixed_ordered', frames_num_min=10, frames_num_max=50,
                      frames_skip_num=0, frames_num_per_batch=True)
    collate = CollateFn(['a', 'b'], sample_cfg)
    lengths = set()
    for t in range(20):
        batch = [([np.zeros((L, 64, 44), np.float32)], ('a' if i % 2 else 'b', 'nm', '000'))
                 for i, L in enumerate([5, 12, 60, 31])]
        seqs, _, _, _, seqL = collate(batch)
        arr = np.asarray(seqs[0])
        assert seqL is None and arr.ndim == 4 and arr.shape[0] == 4 and 10 <= arr.shape[1] <= 50
        lengths.add(arr.shape[1])
    assert len(lengths) > 1
    # without the flag, each sequence draws its own length and the batch is concatenated
    sample_cfg['frames_num_per_batch'] = False
    seqs, _, _, _, seqL = CollateFn(['a', 'b'], sample_cfg)(batch)
    assert seqL.shape == (1, 4) and np.asarray(seqs[0][0]).shape[0] == seqL.sum()


def test_cosine_scheduler_steps_per_iteration():
    cfg = load_cfg(OSL_CFG)
    model = torch.nn.Linear(2, 2)
    holder = type('H', (), {'msg_mgr': type('M', (), {'log_info': staticmethod(lambda *a: None)})()})()
    holder.optimizer = torch.optim.AdamW(model.parameters(), **{k: v for k, v in cfg['optimizer_cfg'].items() if k != 'solver'})
    sched = BaseModel.get_scheduler(holder, cfg['scheduler_cfg'])
    lr = lambda: holder.optimizer.param_groups[0]['lr']  # noqa: E731
    assert np.isclose(lr(), 3e-4)
    sched.step(30000)
    assert np.isclose(lr(), 1.5e-4)
    sched.step(60000)
    assert lr() < 1e-8


if __name__ == '__main__':
    tests = [v for k, v in dict(globals()).items() if k.startswith('test_')]
    for t in tests:
        t()
        print(f'OK  {t.__name__}')
    print(f'{len(tests)} tests OK')
