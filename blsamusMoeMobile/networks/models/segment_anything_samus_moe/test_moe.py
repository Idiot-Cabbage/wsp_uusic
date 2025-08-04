import torch
from .build_moesamus import build_moesamus_vit_b


def simple_moe_test():
    class Args:
        encoder_input_size = 256
        moe_expert_num = 4
        moe_top_k = 1
        router_temp = 1.0
        sam_ckpt = None

    args = Args()
    model = build_moesamus_vit_b(args, checkpoint=None)
    x = torch.randn(1, 3, args.encoder_input_size, args.encoder_input_size)
    _ = model.samus.image_encoder(x)
    if hasattr(model.samus.image_encoder.input_Adapter, "last_routing"):
        dist = model.samus.image_encoder.input_Adapter.last_routing
        print(f"Input adapter usage: {dist.tolist()}")
    for i, blk in enumerate(model.samus.image_encoder.blocks):
        if blk.window_size == 0 and hasattr(blk.MLP_Adapter, "last_routing"):
            dist = blk.MLP_Adapter.last_routing
            print(f"Block {i} adapter usage: {dist.tolist()}")
    print("aux loss", model.get_aux_loss().item())
