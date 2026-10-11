"""Run VGGT-Omega on one or more frame folders and save poses, depth, and confidence.

Usage:
  python src/run_vggt.py OUT_NAME FRAME_DIR [FRAME_DIR ...]   -> data/out/OUT_NAME.{npz,ply,png}
  python src/run_vggt.py --mem-test FRAME_DIR [FRAME_DIR ...]  -> peak VRAM per frame count
"""
import re
import sys
from pathlib import Path

import numpy as np

OUT = Path("data/out")


def src_index(name):
    m = re.search(r"_src(\d+)", name)
    return int(m.group(1)) if m else -1


def frames_of(d, vid):
    return np.flatnonzero(d["video"] == vid)


def frame_list(dirs):
    paths, video, src = [], [], []
    for vid, folder in enumerate(dirs):
        for p in sorted(Path(folder).glob("*.png")):
            paths.append(str(p))
            video.append(vid)
            src.append(src_index(p.name))
    return paths, np.array(video), np.array(src)


def bf16_weights(module):
    """Store parameters in bf16; leave buffers (fp32 RoPE periods, image mean/std) as trained."""
    for p in module.parameters():
        p.data = p.data.bfloat16()
    return module


def load_model():
    import torch
    from huggingface_hub import hf_hub_download
    from vggt_omega.models import VGGTOmega

    model = VGGTOmega().eval()
    ckpt = hf_hub_download("facebook/VGGT-Omega", "vggt_omega_1b_512.pt")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    # aggregator runs under bf16 autocast anyway (saves ~2 GB); heads stay fp32 (DenseHead requires it)
    bf16_weights(model.aggregator)
    return model.to("cuda")


def infer(model, paths):
    import torch
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    images = load_and_preprocess_images(paths, mode="max_size", image_resolution=512).to("cuda")
    with torch.inference_mode():
        pred = model(images)
    extr, intr = encoding_to_camera(pred["pose_enc"], images.shape[-2:])
    return {
        "extrinsic": extr[0].float().cpu().numpy(),
        "intrinsic": intr[0].float().cpu().numpy(),
        "depth": pred["depth"][0, ..., 0].cpu().numpy(),
        "conf": pred["depth_conf"][0].cpu().numpy(),
        "images": (images.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8),
    }


def mem_test(model, paths):
    import torch

    for n in range(10, len(paths) + 1, 10):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            infer(model, paths[:n])
        except torch.OutOfMemoryError:
            print(f"{n} frames: OOM")
            return
        print(f"{n} frames: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


def main(name, dirs):
    from geometry import confident_points, up_from_cameras
    from viz import save_ply, save_views_png

    paths, video, src = frame_list(dirs)
    d = infer(load_model(), paths)
    d.update(video=video, src=src, dirs=np.array([str(x) for x in dirs]))  # later stages find the frames here
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{name}.npz", **d)

    pts, cols = confident_points(d, np.arange(len(paths)))
    save_ply(OUT / f"{name}.ply", pts, cols)
    save_views_png(OUT / f"{name}_views.png", pts, cols, up_from_cameras(d["extrinsic"]))
    if len(dirs) > 1:  # A red, B blue: walls should merge, not double
        pa, _ = confident_points(d, frames_of(d, 0))
        pb, _ = confident_points(d, frames_of(d, 1))
        ab = np.concatenate([np.tile([255, 0, 0], (len(pa), 1)), np.tile([0, 0, 255], (len(pb), 1))]).astype(np.uint8)
        save_views_png(OUT / f"{name}_ab_views.png", np.concatenate([pa, pb]), ab, up_from_cameras(d["extrinsic"]))
        save_ply(OUT / f"{name}_ab.ply", np.concatenate([pa, pb]), ab)
    print(f"{name}: {len(paths)} frames, depth {d['depth'].shape[1:]}, {len(pts)} points")


if __name__ == "__main__":
    if sys.argv[1] == "--mem-test":
        mem_test(load_model(), frame_list(sys.argv[2:])[0])
    else:
        main(sys.argv[1], sys.argv[2:])
