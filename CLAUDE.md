# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Fork of Endo-4DGS (endoscopic monocular 4D Gaussian Splatting) adapted to support the iMED 2026 challenge (EndoVis 2026 / MICCAI): a static two-camera novel-view-synthesis protocol. Gaussians are trained on one endoscope camera stream and evaluated on a held-out second camera, using given relative pose rather than SfM.

Base method: 3D Gaussian Splatting (via `submodules/diff-gaussian-rasterization-depth` and `submodules/simple-knn`) with a deformation field over time built from a K-Planes/HexPlane representation (`scene/hexplane.py`, `scene/deformation.py`) — training runs a "coarse" stage (static Gaussians) followed by a "fine" stage (time-conditioned deformation).

## Setup

```bash
git submodule update --init --recursive
conda create -n ED4DGS python=3.8 && conda activate ED4DGS
pip install -r requirements.txt
pip install -e submodules/diff-gaussian-rasterization-depth
pip install -e submodules/simple-knn
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install torchmetrics
```

There is no test suite, linter, or CI in this repo — validation is done by running training/rendering and inspecting PSNR/SSIM/LPIPS output.

## Common commands

Train on the iMED dataset (config-driven, per-scene):
```bash
sh train_imed.sh
# equivalent to:
PYTHONPATH='.' python train.py -s data/imed/<session_name> --port 6017 --expname "imed/<session_name>" --configs arguments/imed.py
```

Render + evaluate a trained model:
```bash
python render.py --model_path <OUTPUT_PATH> --pc --skip_video --skip_train --configs arguments/imed.py
python metrics.py --model_paths <OUTPUT_PATH>
```

Other dataset entry points follow the same pattern with a different `--configs` file (e.g. `arguments/stereomis.py`, `arguments/endonerf.py`, `arguments/scared.py`) — see `train.sh`/`render.sh`/`eval.sh` for the exact invocations used on other datasets.

StereoMIS preprocessing (depth/pose prep before training):
```bash
sh prepare_stereomis.sh   # cd stereomis && python process_dataset.py && python pre_pose_stereomis.py
sh prepare_depth.sh       # scripts/pre_dam_dep.py — Depth-Anything depth prediction
```

## Dataset dispatch and the iMED protocol

`scene/__init__.py` (`Scene.__init__`) sniffs `args.source_path` to decide which loader to use, in this priority order: `sparse/` → Colmap, `transforms_train.json` → Blender, iMED structure (`pose.txt` + `K.txt` + `endoscope1/`/`endoscope2/` dirs) → iMED, `poses_bounds.npy` (+ "endo"/"stereomis" in path) → EndoNeRF/StereoMIS, `poses_bounds.npy` alone → DyNeRF, `dataset.json` → Nerfies, `point_cloud.obj` → SCARED. Loaders are registered in `scene/dataset_readers.py:sceneLoadTypeCallbacks`.

iMED-specific loader is `scene/imed_loader.py` (`IMED_Dataset`). Key facts baked into it and worth knowing before touching it:
- Train view is always `endoscope2/L`, test/held-out view is always `endoscope1/L` (see `format_infos`).
- `pose.txt` must have exactly 2 rows (`cam_id tx ty tz qx qy qz qw`), cam id `0` = endoscope2, cam id `1` = endoscope1 — this is a static two-camera rig, not per-frame poses.
- `K.txt` holds named intrinsics blocks (`K1_L`, `K2_L`) parsed by scanning for `#K...` header lines.
- Depth (`depthL/*.npy`) is at half the RGB resolution (`toolL` masks are full RGB res); the loader asserts a fixed 2x downscale factor between RGB and depth/mask grids.
- `toolL` masks are inverted on load: `mask = 1 - raw/255`, so mask `True` means "valid, non-tool" region.
- Time values for the deformation field are assigned by frame index / (n-1) per camera stream, independent of any real timestamp.

## iMED evaluation is reprojection-masked, not just tool-masked

Because training and test cameras are physically different cameras, plain PSNR/SSIM over the full frame includes regions in `endoscope1` that `endoscope2` never observed. Both `train.py` (validation during training) and `metrics.py` (final eval) build a **global overlap mask**: depth from `endoscope2` is reprojected into the `endoscope1` camera frame using the two-camera pose/intrinsics relation, then dilated/closed/filled into one static per-sequence mask. This mask is combined with the per-frame tool mask and applied to both PSNR and SSIM before averaging.

- Training-time mask logic lives inline in `train.py:training_report` (`use_imed_overlap_eval`, derived from a foreground-pixel heuristic — cheaper than reprojection).
- Eval-time mask is the accurate reprojection version: `metrics.py:_build_global_imed_overlap_mask`, cached to `overlap_mask.png` / `imed_reprojection_mask.png` in the results dir on first run.
- `metrics.py` reimplements masked PSNR/SSIM locally (`masked_psnr`, `masked_ssim`) rather than using `utils/image_utils.py`, specifically to support this mask; keep both in sync if changing masking semantics.
- Detecting "is this an iMED run" is done by string-matching `"imed"` in the source path (`train.py`) or in the `source_path` recovered from a run's `cfg_args` file (`metrics.py:_extract_source_path_from_cfg`). Any change to iMED directory naming needs to update both detection sites plus `scene/__init__.py`'s structural check.

## Training loop structure (`train.py`)

- `training()` runs `scene_reconstruction()` twice: once for `stage="coarse"` (`opt.coarse_iterations`, static Gaussians, no deformation applied) and once for `stage="fine"` (`opt.iterations`, deformation network active). Checkpoints/config strings note which stage they belong to.
- Per-iteration losses are toggled by `PipelineParams` flags (`use_depth`, `use_smooth`, `use_normal`, `use_confidence`) set per-dataset in `arguments/<dataset>.py`; all raw image/depth/normal/confidence tensors are masked (`* mask_tensor`) before loss computation, so a missing or zero-size mask silently zeroes the loss.
- NaN loss triggers `os.execv` to relaunch the same process from scratch (a poor-man's OOM/divergence retry) — not a crash you should "fix" by removing.
- Gaussian densify/prune/grow are gated by both an iteration schedule (`opt.densify_from_iter`, etc.) and hard point-count bounds (`<360000` to grow, `>200000` to prune).

## Config resolution

Per-dataset hyperparameters live under `arguments/<name>.py` as plain dicts (`ModelParams`, `OptimizationParams`, `ModelHiddenParams`, `PipelineParams`) and are merged over the `arguments/__init__.py` argparse defaults via `--configs <path>` (loaded with `mmcv.Config.fromfile` + `utils/params_utils.merge_hparams` in `train.py`). When adding a new dataset variant, add a new `arguments/<name>.py` rather than editing defaults in `arguments/__init__.py`.

`render.py`'s `get_combined_args` (in `arguments/__init__.py`) reloads the `cfg_args` file saved at training time in `<model_path>/cfg_args`, so rendering/eval reuses the exact training-time model params unless overridden on the command line.
