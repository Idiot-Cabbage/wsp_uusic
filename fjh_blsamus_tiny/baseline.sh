export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch \
    --use_env \
    --nproc_per_node=1 \
    --master_port=12355 \
    omni_train.py \
    --output_dir=exp_out/trial_cls \
    --base_lr=0.003 \


# export CUDA_VISIBLE_DEVICES=0

# python -m torch.distributed.launch \
#     --use_env \
#     --nproc_per_node=1 \
#     --master_port=22345 \
#     omni_test.py \
#     --output_dir=exp_out/trial_cls \
#     --base_lr=0.003 \


