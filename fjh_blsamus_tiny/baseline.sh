export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch \
    --use_env \
    --nproc_per_node=1 \
    --master_port=12355 \
    omni_train.py \
    --output_dir=exp_out/2025-8-9 \
    --base_lr=1e-5  \


# export CUDA_VISIBLE_DEVICES=0

# python -m torch.distributed.launch \
#     --use_env \
#     --nproc_per_node=1 \
#     --master_port=22345 \
#     omni_test.py \
#     --output_dir=exp_out/trial_cls \
#      --base_lr=2e-5  \



