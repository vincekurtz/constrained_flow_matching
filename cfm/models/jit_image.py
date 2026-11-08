"""Native JAX inference for JiT's pretrained pixel-space image flow.

Port of LTH14/JiT, commit cbc743a2ada5e9762697da2c83f8c4f8379e8c17
(MIT, Copyright 2025 Tianhong Li; see licenses/JiT-MIT.txt).
The backbone predicts clean RGB, not velocity or latent VAE coordinates.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

from cfm.models.normalizer import Normalizer


class JiT:
    """Frozen JiT transformer; NHWC inputs and clean-image outputs."""

    def __init__(
        self,
        weights,
        *,
        size=512,
        patch=32,
        heads=16,
        depth=32,
        context_start=10,
    ):
        self.data_shape = (size, size, 3)
        self.size, self.patch, self.heads = size, patch, heads
        self.depth, self.context_start = depth, context_start
        self.p = {
            # JIT closures upload these constants. Keep the stored weights
            # on the host to avoid a second 3.8 GB copy on the GPU.
            k.removeprefix("net."): np.asarray(v, dtype=np.float32)
            for k, v in weights.items()
        }
        self.width = self.p["pos_embed"].shape[-1]
        self.context_len = self.p["in_context_posemb"].shape[1]
        if size % patch or self.width % (4 * heads):
            raise ValueError(
                "Incompatible image, patch, or attention dimensions"
            )
        if not 0 <= context_start < depth or self.context_len == 0:
            raise ValueError(
                "JiT requires in-context tokens within the network"
            )
        n = size // patch
        if self.p["pos_embed"].shape != (1, n * n, self.width):
            raise ValueError(
                "Checkpoint position embedding has the wrong shape"
            )
        if self.p["final_layer.linear.weight"].shape != (
            patch * patch * 3,
            self.width,
        ):
            raise ValueError("Checkpoint output patches have the wrong shape")
        if sum(k.endswith("attn.qkv.weight") for k in self.p) != depth:
            raise ValueError("Checkpoint transformer depth does not match")

        head = self.width // heads
        dim = head // 2
        freqs = 1 / (10000 ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        phase = np.repeat(
            np.arange(n, dtype=np.float32)[:, None] * freqs[None], 2, axis=-1
        )
        phase = np.concatenate(
            [
                np.broadcast_to(phase[:, None], (n, n, dim)),
                np.broadcast_to(phase[None, :], (n, n, dim)),
            ],
            axis=-1,
        ).reshape(n * n, head)
        self.cos, self.sin = (
            jnp.asarray(np.cos(phase)),
            jnp.asarray(np.sin(phase)),
        )
        self.cos_ctx = jnp.concatenate(
            [
                jnp.ones((self.context_len, head)),
                self.cos,
            ]
        )
        self.sin_ctx = jnp.concatenate(
            [
                jnp.zeros((self.context_len, head)),
                self.sin,
            ]
        )

    def _linear(self, x, key):
        return x @ self.p[key + ".weight"].T + self.p[key + ".bias"]

    def _norm(self, x, key):
        return (
            x
            * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6)
            * self.p[key + ".weight"]
        )

    def _rope(self, x, context):
        pairs = x.reshape(*x.shape[:-1], -1, 2)
        rotated = jnp.stack([-pairs[..., 1], pairs[..., 0]], axis=-1)
        cos = self.cos_ctx if context else self.cos
        sin = self.sin_ctx if context else self.sin
        return x * cos + rotated.reshape(x.shape) * sin

    def __call__(self, images, t, labels):
        p, b = self.p, images.shape[0]
        n, sz = self.size // self.patch, self.patch
        # Input convolution flattens C,H,W; output patches flatten H,W,C.
        patches = images.reshape(b, n, sz, n, sz, 3)
        patches = patches.transpose(0, 1, 3, 5, 2, 4).reshape(b, n * n, -1)
        h = patches @ p["x_embedder.proj1.weight"].reshape(-1, 3 * sz * sz).T
        h = (
            h @ p["x_embedder.proj2.weight"][:, :, 0, 0].T
            + p["x_embedder.proj2.bias"]
            + p["pos_embed"]
        )
        half = p["t_embedder.mlp.0.weight"].shape[1] // 2
        freq = jnp.exp(
            -math.log(10000) * jnp.arange(half, dtype=jnp.float32) / half
        )
        phase = t[:, None] * freq[None]
        time_features = jnp.concatenate(
            [jnp.cos(phase), jnp.sin(phase)], axis=-1
        )
        temb = self._linear(
            jax.nn.silu(self._linear(time_features, "t_embedder.mlp.0")),
            "t_embedder.mlp.2",
        )
        yemb = jnp.asarray(p["y_embedder.embedding_table.weight"])[labels]
        condition = jax.nn.silu(temb + yemb)
        for i in range(self.depth):
            if i == self.context_start:
                h = jnp.concatenate(
                    [
                        yemb[:, None] + p["in_context_posemb"],
                        h,
                    ],
                    axis=1,
                )
            key = f"blocks.{i}"
            sh1, sc1, g1, sh2, sc2, g2 = jnp.split(
                self._linear(condition, key + ".adaLN_modulation.1"), 6, axis=-1
            )
            a = (
                self._norm(h, key + ".norm1") * (1 + sc1[:, None])
                + sh1[:, None]
            )
            qkv = self._linear(a, key + ".attn.qkv")
            q, k, v = jnp.moveaxis(
                qkv.reshape(b, h.shape[1], 3, self.heads, -1), 2, 0
            )
            q, k, v = (x.transpose(0, 2, 1, 3) for x in (q, k, v))
            context = i >= self.context_start
            q = self._rope(self._norm(q, key + ".attn.q_norm"), context)
            k = self._rope(self._norm(k, key + ".attn.k_norm"), context)
            attention = jax.nn.softmax(
                (q @ jnp.swapaxes(k, -1, -2)) / math.sqrt(q.shape[-1]), axis=-1
            )
            a = (attention @ v).transpose(0, 2, 1, 3)
            a = a.reshape(b, h.shape[1], self.width)
            h = h + g1[:, None] * self._linear(a, key + ".attn.proj")
            a = (
                self._norm(h, key + ".norm2") * (1 + sc2[:, None])
                + sh2[:, None]
            )
            a1, a2 = jnp.split(self._linear(a, key + ".mlp.w12"), 2, axis=-1)
            h = h + g2[:, None] * self._linear(
                jax.nn.silu(a1) * a2, key + ".mlp.w3"
            )
        h = h[:, self.context_len :]
        shift, scale = jnp.split(
            self._linear(condition, "final_layer.adaLN_modulation.1"),
            2,
            axis=-1,
        )
        h = self._norm(h, "final_layer.norm_final") * (1 + scale[:, None])
        h = self._linear(h + shift[:, None], "final_layer.linear")
        return (
            h.reshape(b, n, n, sz, sz, 3)
            .transpose(0, 1, 3, 2, 4, 5)
            .reshape(b, self.size, self.size, 3)
        )


class JiTImageFlow:
    """Adapt clean prediction to the repository's standard Gaussian flow.

    Native JiT uses z = t*RGB + (1-t)*noise_scale*epsilon, RGB in [-1,1].
    Our state is x=z/noise_scale, so v=(clean-z)/(noise_scale*max(1-t,t_eps)).
    Pair with Normalizer(.5, noise_scale/2) to recover physical RGB in [0,1].
    Keep the upstream positive velocity floor (.05): it bounds the
    endpoint drift and makes the clean estimate approach the state at t=1.
    Their classifier-free guidance remains active only for .1 < t < 1.
    No reference image or inpainting mask enters this adapter.
    """

    def __init__(self, net, *, label=207, cfg=1.0, noise_scale=1.0, t_eps=.05):
        if not 0 <= label < 1000:
            raise ValueError("ImageNet label must be between 0 and 999")
        if noise_scale <= 0 or t_eps <= 0 or cfg < 0:
            raise ValueError("Invalid noise scale, time floor, or guidance")
        self.net, self.data_shape = net, net.data_shape
        self.label, self.cfg = label, cfg
        self.noise_scale, self.t_eps = noise_scale, t_eps

    def __call__(self, x, t):
        z = x * self.noise_scale
        if self.cfg == 1.0:
            # Guidance one is exactly the conditional prediction. Avoid a
            # second backbone evaluation (and its input-gradient storage).
            with jax.default_matmul_precision("highest"):
                clean = self.net(z, t, jnp.full(t.shape, self.label))
            denominator = jnp.maximum(1 - t, self.t_eps)[
                :, None, None, None
            ]
            return (clean - z) / (denominator * self.noise_scale)
        images, times = jnp.concatenate([z, z]), jnp.concatenate([t, t])
        labels = jnp.concatenate(
            [
                jnp.full(t.shape, self.label),
                jnp.full(t.shape, 1000),
            ]
        )
        with jax.default_matmul_precision("highest"):
            cond, uncond = jnp.split(self.net(images, times, labels), 2, axis=0)
        scale = jnp.where((t > 0.1) & (t < 1), self.cfg, 1.0)[
            :, None, None, None
        ]
        clean = uncond + scale * (cond - uncond)
        denominator = jnp.maximum(1 - t, self.t_eps)[:, None, None, None]
        return (clean - z) / (denominator * self.noise_scale)


def load_pretrained(path, *, label=975, cfg=1.0):
    """Load the authors' JiT-H/32 512px EMA1 checkpoint, without training."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if "model_ema1" not in state:
        raise ValueError("Expected the authors' EMA1 checkpoint")
    weights = {k: v.detach().numpy() for k, v in state["model_ema1"].items()}
    model = JiTImageFlow(JiT(weights), label=label, cfg=cfg, noise_scale=2.)
    return model, Normalizer(jnp.array(0.5), jnp.array(1.0))
