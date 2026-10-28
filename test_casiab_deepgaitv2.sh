export CUDA_VISIBLE_DEVICES=0,1,2,3
./.venv/bin/torchrun --nproc_per_node=4 opengait/main.py --cfgs configs/deepgaitv2/DeepGaitV2_casiab.yaml --phase test
