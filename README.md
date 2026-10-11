# 3D Change Detection (threedcd)

## Overview

threedcd compares two phone videos of the same room and reports what changed between them. You film a room,
move a few things, and film it again. The system works out, in 3D, which objects were moved, removed, added or
swapped, where each one was and where it went, and describes each change in a plain-English sentence.

Under the hood it:

1. Turns both videos into one shared 3D model of the room with VGGT-Omega, a neural network that recovers camera
   positions and depth from ordinary video.
2. Measures how accurately the two recordings line up, so normal reconstruction noise is not mistaken for change.
3. Looks for changes with four complementary detectors: 3D free space, per-pixel depth, colour, and named
   objects found by OWLv2.
4. Asks a vision-language model (Claude Sonnet in this repo) to check each candidate change against before and
   after photos of that exact spot, and to describe what happened.

It needs no special hardware, markers or scanning app: two handheld videos, taken from different paths, are
enough. The output is a short report, before/after images of each change, and an interactive 3D view of the room
with every change marked.

The repo includes the two recordings it was developed on (`data/raw/`, a living room where a fan, a laptop and a
pillow were moved), so you can reproduce the results below.

## Running it

### Requirements

- An NVIDIA GPU with at least 8 GB of memory, and Python 3.12.
- A Hugging Face account with access to VGGT-Omega. The weights are gated: request access at
  https://huggingface.co/facebook/VGGT-Omega (approval is automatic and usually quick).
- A vision-capable LLM for the last two steps (verifying and describing changes). Any frontier or open-source
  model that accepts images will do. As written, the code calls Claude through the Anthropic API. To use a
  different model, replace the request in `verify_vlm.ask` and `describe.describe`
  (one API call each, with the same JSON schema). Without any LLM, the pipeline still runs through `fuse.py`,
  and `data/out/changes.json` is the unverified list of changes.

### Setup

```bash
git clone https://github.com/RohitKarthickeyan/threedcd.git
cd threedcd
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
hf auth login                      # paste a Hugging Face read token
```

If you use Claude, create a file called `.env` in the repo root containing your Anthropic key. It is git-ignored.

```
ANTHROPIC_API_KEY=
```

### Run on the included recordings

From the repo root, with the virtual environment active:

```bash
python src/extract_frames.py data/raw/IMG_8289.mov data/frames/IMG_8289 40   # video A, before
python src/extract_frames.py data/raw/IMG_8290.mov data/frames/IMG_8290 40   # video B, after
python src/run_vggt.py joint data/frames/IMG_8289 data/frames/IMG_8290        # 3D model, ~1 min, 6.4 GB VRAM
python src/check_alignment.py
python src/voxel_labels.py
python src/diff.py
python src/detect_objects.py
python src/depth_compare.py
python src/fuse.py
python src/verify_vlm.py                           # Claude: confirm changes
python src/describe.py                             # Claude: describe changes
python src/score.py data/out/changes_verified.json # compare with data/gt.json
python src/view3d.py                         # 3D viewer (opens a local web page)
```

The first run downloads the models (VGGT-Omega about 4 GB, OWLv2 about 0.6 GB). Everything is written to
`data/out/`:

- `report.md`: the plain-English list of changes.
- `change_N.jpg`: a before/after image for each change.
- `changes_verified.json`: every change with its 3D location, label and description.
- `scene.glb`: the 3D scene shown by the viewer.

Results can differ slightly from the ones below: the 3D model and Claude's wording are not fully deterministic.

### Your own recordings

Record two videos of the same room, before and after, walking slowly rather than turning on the spot, and
covering the same area. Put them in `data/raw/` and run the same commands with your file names. One script
names the frame folders directly: change `FRAMES` in `src/describe.py` to match. `score.py` only makes
sense with your own ground truth in `data/gt.json`.

### Tests

```bash
python -m pytest -q                     # unit tests, synthetic data, no GPU or API needed
python tests/synthetic_e2e.py OUT_DIR   # CPU-only end-to-end check on a ray-cast room (~3 min)
```

To check how many frames fit on your GPU:
`python src/run_vggt.py --mem-test data/frames/IMG_8289 data/frames/IMG_8290`.

## What it found

Each image pair is the object at its old spot in the first video (left, before) and at its new spot in the
second video (right, after). These are the crops the system sent to Claude, located automatically from the 3D
reconstruction. Below each pair is Claude's description.

**Fan**

![Fan before and after](assets/fan.jpg)

> The gray standing fan was moved about 1.3 meters along the wall, from under the decorative wall art by the TV
> shelves to a spot beside the floor vent.

**Laptop**

![Laptop before and after](assets/laptop.jpg)

> The open silver laptop that was lying on the chaise end of the grey sofa has been moved about 1.4 metres to the
> main sofa seat, next to the brown cushions.

**Pillow**

![Pillow before and after](assets/cushion.jpg)

> The yellow-green cushion was lifted off the dark brown cushion and put on the sofa seat against the backrest,
> about 1.3 metres from where it was.

## Results

On the included recordings (two ~16 s portrait videos, three objects moved), scored against hand-marked ground
truth in `data/gt.json`: the old and new spot of each object, 6 spots in total.

| Version | Changes reported | Spots found | False alarms |
|---|---|---|---|
| 3D free-space comparison only (first prototype) | 28 | 4 / 6 | 24 |
| All four detectors + Claude verification | 13 | 6 / 6 | 5 |
| + merging fragments into moves (current) | 3 | 6 / 6 | 0 |

## How it works

```
videos -> frames -> VGGT-Omega (poses + depth, both videos in one 3D frame) -> alignment error (tau)
       -> four detectors -> fuse into one list -> Claude verifies each change -> merge fragments
       -> Claude describes each change -> report + 3D viewer
```

1. **Frames** (`extract_frames.py`). 40 frames per video, the sharpest in each time window.
2. **3D reconstruction** (`run_vggt.py`). VGGT-Omega, a feed-forward network, predicts a camera pose, depth map and
   confidence for all 80 frames in one pass, so both videos land in the same coordinate frame. No separate
   registration step.
3. **Alignment error** (`check_alignment.py`). Unchanged surfaces (walls, furniture) should coincide; how far
   they don't is tau (~3 cm here). Every later threshold is a multiple of tau, because VGGT's scale is arbitrary.
4. **Four detectors**, each with a different blind spot:
   - *Voxel diff* (`voxel_labels.py`, `diff.py`). Labels 1 cm voxels occupied / free / unknown per video, then
     compares: free before and occupied after = added, and the reverse = removed. A voxel unknown in either video
     never counts as a change, so regions only one video saw cannot produce false removals. Blind to objects
     thinner than ~3 tau.
   - *Colour change* (`diff.py`). Occupied in both videos but a different colour: one object replaced by another.
   - *Object detection* (`detect_objects.py`). OWLv2 finds named objects in every frame. They are placed in 3D,
     and A and B are matched by label and colour; far-apart matches are moves. This catches thin objects like the
     laptop, and gives labels.
   - *Depth comparison* (`depth_compare.py`). Projects B's depth into A's cameras and compares per pixel after
     subtracting the local median, which cancels regional misalignment. Sensitive to small changes.
5. **Fusion** (`fuse.py`). Groups each detector's voxels into object-sized clusters and attaches them to the
   object detector's labelled changes, giving one list.
6. **Verification** (`verify_vlm.py`). For each candidate, crops of the same 3D spot from both videos go to Claude
   Sonnet 5.5, which confirms or rejects the change. Confirmed fragments of the same move are then merged.
7. **Description** (`describe.py`). Crops sized to the whole object, from the full-resolution frames, go to Claude
   for one plain sentence per change. Geometry supplies the change type and the distance.
8. **Scoring and viewing** (`score.py`, `view3d.py`).

## Design decisions and tradeoffs

### Why VGGT-Omega instead of COLMAP

COLMAP is the classic structure-from-motion pipeline: match features across images, solve for camera poses, then
optionally run dense multi-view stereo.

- **Fragile on this kind of input.** It struggles with blurry phone video, blank walls and low-texture
  furniture, which are exactly what a living room has. When matching fails it drops frames or the whole
  reconstruction.
- **Sparse unless you run dense stereo**, which takes many minutes per scene.
- **Two videos are two reconstructions**, each in its own coordinate frame and scale, so they would need to be
  registered against each other. That registration error would show up as false changes.

VGGT-Omega predicts poses, dense depth and per-pixel confidence for all 80 frames in about 20 seconds, and puts
both videos in one coordinate frame because it sees them together. The cost is that its scale is arbitrary,
which is why every threshold here is a multiple of the measured alignment error.

### Why not 3D Gaussian splatting or NeRFs

Both are excellent at photorealistic rendering, and both are the wrong tool for deciding what changed.

- **They need camera poses first**, from COLMAP or VGGT, so they add a stage rather than replacing one.
- **They are optimised per scene**, which takes minutes to hours of training per video.
- **They fill unseen space with plausible guesses.** Floaters and hallucinated surfaces appear where no camera
  looked. Change detection depends on telling *empty* (seen through) apart from *unknown* (never seen). A splat
  or NeRF blurs that distinction, and its guesses would show up as false changes.
- **Comparing renders mixes in view-dependent effects** such as reflections and lighting, which are not real
  changes.

The voxel diff instead uses explicit evidence: a camera saw a surface here (occupied), saw through here (free),
or never looked (unknown). Splats are still a good fit for a nicer demo: train them on the VGGT poses and paint the
detected changes onto them.

### Why not panorama stitching or a 2D image diff

- **Stitching assumes the camera only rotates.** Our videos are walked, and the parallax that walking creates is
  exactly what stitching cannot model. Objects at different depths misalign, and the misalignment looks like
  change.
- **2D has no "where" or "unknown".** A 2D difference cannot say where an object went in the room, cannot tell
  "moved" apart from "seen from a different angle", and cannot tell "removed" apart from "hidden behind
  something".

The pipeline does use 2D images where they are strong: recognising objects (OWLv2) and judging before/after crops
(Claude). But it ties everything to 3D positions.

### Why several detectors and a VLM

No single signal caught all three changes:

| Detector | Catches | Misses |
|---|---|---|
| Voxel diff | large objects (fan, pillow) | the laptop: its reclined lid sits within ~3 cm of the cushion |
| Depth comparison | the laptop | objects that fill a whole comparison block |
| Object detection | named objects, any thickness; labels | unnamed objects; same-looking swaps (without the colour check) |
| Colour change | one object replaced by another | lighting changes (false alarms) |

The detectors are deliberately tuned to catch everything, so on their own they produce many false alarms. A vision-
language model looking at the actual before/after crops removes most of them. That combination
gives high recall from geometry and high precision from the VLM.

### Things tried and dropped

- **DINOv2 features instead of colour.** Per-voxel patch features varied more between views of the same video
  than between the two videos, and missed the laptop entirely. Plain Lab colour worked better.
- **Matching objects by label only.** The pillow's move was a swap: each of its spots held *a* pillow in both
  videos. Matching by label and colour fixed it.
- **Dilating the unknown region**, as originally planned. Every surface borders unseen space behind it, so
  dilation erased every change.
