"""Regenerate the offline JiT fixture using the authors' implementation.

Clone https://github.com/LTH14/JiT at commit
cbc743a2ada5e9762697da2c83f8c4f8379e8c17, install its `einops` dependency,
then run `python tests/make_jit_image_golden.py /path/to/JiT`.
This script is not needed by pytest; no pretrained weights are downloaded.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.upstream))
    # Upstream constructs RoPE on CUDA and decorates RMSNorm with compile.
    # Run the same equations on CPU without requiring a CUDA PyTorch install.
    torch.compile = lambda fn, *a, **kw: fn
    torch.Tensor.cuda = lambda self, *a, **kw: self
    from model_jit import JiT

    torch.manual_seed(17)
    net = JiT(
        input_size=16, patch_size=4, hidden_size=32, depth=2, num_heads=4,
        bottleneck_dim=8, in_context_len=4, in_context_start=1,
    ).eval()
    with torch.no_grad():
        for name, parameter in net.named_parameters():
            if ("adaLN_modulation" in name
                    or name.startswith("final_layer.linear")):
                # Upstream zero initialization would skip the attention blocks
                # and yield a trivial zero output/gradient regression fixture.
                parameter.normal_(std=.08)
    x = torch.randn(2, 3, 16, 16, requires_grad=True)
    t, labels = torch.tensor([.2, .97]), torch.tensor([207, 1000])
    cotangent = torch.randn_like(x)
    y = net(x, t, labels)
    (y * cotangent).sum().backward()

    def nhwc(tensor):
        return tensor.detach().numpy().transpose(0, 2, 3, 1)

    fixture = {k: v.detach().numpy() for k, v in net.state_dict().items()}
    fixture.update(x=nhwc(x), t=t.numpy(), labels=labels.numpy(), y=nhwc(y),
                   cotangent=nhwc(cotangent), gradient=nhwc(x.grad))
    np.savez_compressed(
        Path(__file__).parent / "goldens/jit_image.npz", **fixture,
    )


if __name__ == "__main__":
    main()
