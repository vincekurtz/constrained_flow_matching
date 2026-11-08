"""Download and convert the authors' JiT-H/32 ImageNet 512px checkpoint."""

import argparse
import hashlib
from pathlib import Path
import urllib.request

from cfm.core import checkpoint
from cfm.models.jit_image import load_pretrained

CHECKPOINT_URL = (
    "https://www.dropbox.com/scl/fo/3ken1avtsd81ip67b9qpi/"
    "AK218ZNvXKSv74igVvht4PQ?rlkey=14gjrblmljewpl6ygxzlr3njm"
    "&dl=1&preview=/jit-h-32/checkpoint-last.pth"
)
CHECKPOINT_SHA256 = (
    "b647e23fd1995c41418c57d74a35f8060ae209545c390237ad59e6a498650e9b"
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(source, output, *, download=False, label=975, cfg=1.0):
    source, output = Path(source), Path(output)
    if not source.exists():
        if not download:
            raise FileNotFoundError(
                f"No checkpoint at {source}; pass --download to fetch it."
            )
        source.parent.mkdir(parents=True, exist_ok=True)
        partial = source.with_suffix(source.suffix + ".part")
        print("Downloading the authors' 11.5 GB checkpoint...", flush=True)
        request = urllib.request.Request(
            CHECKPOINT_URL, headers={"User-Agent": "cfm-pretrained-example"}
        )
        with (
            urllib.request.urlopen(request) as response,
            partial.open("wb") as f,
        ):
            while chunk := response.read(8 * 1024 * 1024):
                f.write(chunk)
        if sha256(partial) != CHECKPOINT_SHA256:
            raise ValueError(
                f"SHA256 mismatch for {partial}; download incomplete"
            )
        partial.replace(source)
    if sha256(source) != CHECKPOINT_SHA256:
        raise ValueError(f"SHA256 mismatch for {source}")
    model, normalizer = load_pretrained(source, label=label, cfg=cfg)
    checkpoint.save(output, model, normalizer)
    print(f"Saved JiT-H/32 EMA1, 512px, ImageNet class {label}: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/jit/jit-h-32.pth")
    parser.add_argument("--output", default="data/imagenet_native_model.pkl")
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--label",
        type=int,
        default=975,
        help="ImageNet class index; 975 is lakeside.",
    )
    args = parser.parse_args()
    prepare(args.source, args.output, download=args.download,
            label=args.label, cfg=args.cfg)


if __name__ == "__main__":
    main()
