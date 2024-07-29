# virtualenv --python='/usr/bin/python3.9' virtualenv-split-weights
# source virtualenv-split-weights/bin/activate
# pip install torch 'numpy<2'
# cd /mnt/ramdisk/llama3_1/Meta-Llama-3.1-405B-MP16/
# python3 split_llama3_1_weights.py

import pathlib
import torch

outdir = pathlib.Path("split_weights")
outdir.mkdir()
for f in pathlib.Path(".").glob("consolidated*.pth"):
    print(f"Splitting {f}...")
    outdir.joinpath(f).mkdir()
    vars = torch.load(
        f.as_posix(), weights_only=True, map_location=torch.device("cpu"), mmap=True
    )
    for k, v in vars.items():
        torch.save(v, outdir.joinpath(f, k))
    f.unlink()
