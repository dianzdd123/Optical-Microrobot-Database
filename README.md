# Optical Microrobot Database

**MSEI Lab:** [Website](https://zhangrobotics.com/msei_website/) · [Website on GitHub](https://github.com/dianzdd123/msei_website)

## Start here

**Repository:** [dianzdd123/Optical-Microrobot-Database](https://github.com/dianzdd123/Optical-Microrobot-Database)

This repository is a fork of [LannWei/Optical-Microrobot-Database](https://github.com/LannWei/Optical-Microrobot-Database). Original authorship, citations and upstream notices are retained. Changes here apply to this fork.

**Scope:** Dataset project page and baseline scripts; the full image dataset must be obtained separately.

| Need | Where to start |
|---|---|
| Setup, file map, inputs/outputs and troubleshooting | [Reuse guide](docs/REUSE.md) |
| Local structural validation | `python scripts/check_repository.py` (Python 3.10+) |
| Proposed fixes and reproducibility details | [Contributing](CONTRIBUTING.md) |

The structural check does not run experiments or certify hardware, model accuracy or dependency compatibility. Reuse permissions need clarification: this checkout has no repository-wide licence file.


Project page, example imagery and Python baselines for optical microrobot pose classification and depth regression.

The full dataset is external to this Git checkout. See the [project page source](index.html) for the research description and data-access information.

## Main components

- [Pose data preparation](pose_data.py) and [PyTorch dataset](pose_dataset.py).
- [Pose baseline](pose_baseline.py) and [depth baseline](depth_baseline.py).
- Model definitions in `pose_model_zoo.py` and `depth_model_zoo.py`.
- Hyperparameter-search and inference scripts prefixed `search_`.
- Project-page illustrations under `resource/` and `static/`.
