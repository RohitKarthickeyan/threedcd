"""Run the whole change-detection pipeline on data/raw/before.* and data/raw/after.*.

Usage: python run.py [--frames N] [--no-llm] [--no-view]
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

RAW = Path("data/raw")
STAGES = (("alignment check", "check_alignment"), ("voxel labels", "voxel_labels"),
          ("voxel and colour diff", "diff"), ("object detection (OWLv2)", "detect_objects"),
          ("depth comparison", "depth_compare"), ("fusion", "fuse"))


def stage(name, *args):
    print(f"==> {name}", flush=True)
    t = time.time()
    if subprocess.run([sys.executable, *args]).returncode:  # one process per stage frees GPU memory between models
        sys.exit(f"Stage '{name}' failed; see the output above.")
    print(f"    done in {time.time() - t:.0f} s", flush=True)


def main():
    p = argparse.ArgumentParser(description="Detect what changed between data/raw/before.* and data/raw/after.*.")
    p.add_argument("--frames", type=int, default=40, help="frames per video (default 40)")
    p.add_argument("--no-llm", action="store_true", help="skip verification and descriptions (no LLM needed)")
    p.add_argument("--no-view", action="store_true", help="don't open the 3D viewer at the end")
    a = p.parse_args()
    videos = [next(RAW.glob(f"{name}.*"), None) for name in ("before", "after")]
    if None in videos:
        p.error("put the two videos in data/raw/ as before.<ext> and after.<ext>")

    dirs = [Path("data/frames") / v.stem for v in videos]
    for v, out in zip(videos, dirs):
        shutil.rmtree(out, ignore_errors=True)  # stale frames from another run would be mixed in
        stage(f"extract frames from {v}", "src/extract_frames.py", str(v), str(out), str(a.frames))
    stage("3D reconstruction (VGGT-Omega)", "src/run_vggt.py", "joint", *map(str, dirs))
    for name, script in STAGES:
        stage(name, f"src/{script}.py")
    changes = "data/out/changes.json"
    if not a.no_llm:
        stage("LLM verification", "src/verify_vlm.py")
        stage("LLM descriptions", "src/describe.py")
        changes = "data/out/changes_verified.json"
        print("\n" + Path("data/out/report.md").read_text(encoding="utf-8"))
    if Path("data/gt.json").exists():
        stage("score against data/gt.json", "src/score.py", changes)
    print(f"Outputs are in data/out/ (changes: {changes}).")
    if not a.no_view:
        stage("3D viewer (Ctrl+C to stop)", "src/view3d.py", changes)


if __name__ == "__main__":
    main()
