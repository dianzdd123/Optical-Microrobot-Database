# Reusing Optical-Microrobot-Database

Optical microrobot depth and pose baselines.

**Current scope:** Dataset project page and baseline scripts; the full image dataset must be obtained separately.

This repository is a fork of [LannWei/Optical-Microrobot-Database](https://github.com/LannWei/Optical-Microrobot-Database). Original authorship, citations and upstream notices are retained. Changes here apply to this fork.

## Environment and inspection

The research scripts import PyTorch/torchvision, pandas, Pillow, scikit-learn, THOP, transformers and, for optimisation, Optuna. No original pinned environment is included. Install a compatible PyTorch/torchvision pair for your hardware and record your working versions rather than treating an untested package list as a lockfile.

```bash
python scripts/check_repository.py
python -m pip install pandas scikit-learn
python -m unittest discover -s tests -v
```

The split regression tests require pandas and scikit-learn only; they do not train models or require the published dataset.

## File map

| Path | Purpose |
|---|---|
| `index.html`, `resource/`, `static/` | Dataset project page and example figures |
| `pose_data.py` | Deterministic pose-file discovery and stratified train/validation/test split |
| `pose_dataset.py` | Image transforms and PyTorch dataset wrapper |
| `pose_baseline.py`, `pose_model_zoo.py` | Pitch/roll classification experiment and backbones |
| `depth_baseline.py`, `depth_model_zoo.py` | Depth regression experiment and backbones |
| `search_pose_*.py`, `search_depth_*.py` | Search and inference experiments |

## Data contracts

Pose data live under `Image_Pose/<robot>/P<pitch>_R<roll>/` with PNG/JPG/JPEG images. Pose-folder names must match the complete `P<number>_R<number>` pattern; signed integer angles are supported. `load_dataframes()` returns **train, validation, test, number of pitch classes, number of roll classes**. The default image-level split is 60/20/20 with seed 42. Each joint pose class needs enough examples for both stratified stages; a small or empty dataset produces an explanatory error.

Label mappings are derived from the complete collection and carried in the returned DataFrame metadata so a sampled subset cannot silently renumber pose classes. When constructing an independent DataFrame, explicitly pass common `p_map` and `r_map` to `PoseDataset` if you need alignment across datasets.

Depth experiments expect `Image_Depth/<robot>/`, image files and a text file containing one depth value per image (last whitespace-separated field on each line). The script pairs labels with lexicographically sorted image filenames. Verify this ordering and image/label counts before training.

## Running an experiment

Set `robot_name`, model list, data path, sample budget and output locations at the top of the selected script. Run it from the repository root only after placing the real data. The baseline scripts execute training at top level; importing them is not a safe inspection method. Use the shared dataset modules for reuse instead.

Pose outputs are written under `checkpoint/`; depth outputs under `checkpoint_Depth/`. Preserve the exact split and class mappings with your checkpoints. Image-level splitting is a baseline protocol: correlated frames can cross splits, so use sequence-/robot-level held-out evaluation when that is the research question.

The project page describes dataset access. Example images in `resource/` are illustrations, not the full training data. Neither syntax checks nor synthetic split tests establish paper-level reproduction.

## Attribution and reuse permissions

No repository-wide licence file is included in this checkout. Confirm reuse and redistribution permissions with the relevant rights holders; this maintenance update does not add a licence or relicense upstream code, data or assets.

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the local workflow and what to include with a proposed change.
