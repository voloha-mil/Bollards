# Bollards

Country classification from Street View bollards with a detector + classifier pipeline.

Bollards mines Street View imagery, crops detected bollards, and predicts country labels.
It covers dataset mining, local dataset preparation, training, analysis, and a live
screen demo for fast iteration.

## Highlights

- Uses pretrained YOLO-based bollard detection with configurable filters ([YOLOv12_traffic-delineator](https://huggingface.co/maco018/YOLOv12_traffic-delineator)).
- Trains timm backbone classifier with bbox-geometry metadata conditioning (BollardNet).
- End-to-end workflows: OSV-5M mining, dataset prep, training, analysis.
- Live screen inference with hotkey trigger and grid viewer.
- JSON configs with include/override support for reproducible runs.

## Contents

- [Visuals (placeholders)](#visuals-placeholders)
- [What is a bollard?](#what-is-a-bollard)
- [Quickstart](#quickstart)
- [Pipeline](#pipeline)
- [Workflows](#workflows)
- [Training overview](#training-overview)
- [Configuration](#configuration)
- [Data and Storage](#data-and-storage)
- [Results (placeholders)](#results-placeholders)
- [Project Layout](#project-layout)
- [Development](#development)
- [Roadmap](#roadmap)
- [License](#license)

## Visuals (placeholders)

Add assets under `docs/assets/` (or update paths to your preferred location).

![Hero: live screen grid](docs/assets/hero_live_screen.png)
![Pipeline diagram](docs/assets/pipeline_diagram.png)
![Mining montage](docs/assets/mining_montage.png)
![Training curves](docs/assets/training_curves.png)
![Confusion matrix](docs/assets/confusion_matrix.png)
![Live screen demo](docs/assets/live_screen_demo.gif)

## What is a bollard?

Bollards are short, vertical posts placed to guide traffic or protect pedestrians.
They vary by shape, material, reflectors, and spacing across regions, making them
a useful visual cue for country classification.

![Bollard examples](docs/assets/bollard_examples.png)

## Quickstart

Install core dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional extras:

```bash
# Live screen capture dependencies
pip install -r requirements.live.txt

# Dev tooling
pip install -r requirements.dev.txt
```

On Ubuntu, `init.sh` documents required system packages and a one-shot setup flow.

## Pipeline

High-level flow:

```text
OSV-5M frames -> YOLO detector -> bollard crops + bbox meta -> BollardNet classifier -> country prediction
```

The classifier fuses image features (timm backbone) with bbox geometry/confidence
metadata for improved discrimination.

## Workflows

All entrypoints accept `--config` and `--set` overrides. Defaults live in `configs/`.

- Mine OSV-5M: run the detector and build a filtered dataset index (optional S3 artifacts).
- Prepare local dataset: download/filter crops and write `local_data/images/` + `local_data/meta/`.
- Train: fit the classifier and save checkpoints + TensorBoard logs to `runs/`.
- Analyze: generate reports and galleries for a run.
- Live screen: capture the desktop, detect bollards, and render a prediction grid.

## Training overview

The training loop builds a classifier on bollard crops plus bbox metadata:

- **Data inputs**: `local_data/meta/train.csv` and `local_data/meta/val.csv` include
  image paths, labels, bbox coordinates, and detection confidence.
- **Preprocessing**: crops are expanded (`data.expand`) and augmented when enabled
  (resize/pad, random crop, flips, color jitter, blur, affine).
- **Model**: a timm backbone for image features plus a small MLP for bbox meta
  (`x_center`, `y_center`, `w`, `h`, `conf`), then a fused classification head.
- **Loss**: cross-entropy with label smoothing by default, optionally focal loss;
  per-sample weights are scaled by bbox confidence and clamped by `optim.conf_weight_min`.
- **Imbalance handling**: choose **one** of balanced sampling, class weighting, or focal loss.
- **Optimization**: AdamW with separate learning rates for backbone/head and cosine
  annealing; optional backbone freeze for the first `schedule.freeze_epochs`.
- **Metrics & checkpoints**: top-1, top-5, and mAP; best checkpoint is selected by
  `logging.best_metric`. Optional golden dataset eval adds extra metrics and grids.
- **Hugging Face upload**: enable `hub.enabled` to push `best.pt` plus configs to a
  model repo after training completes.

## Configuration

Configs are JSON with an `include` key for layered defaults (see `configs/common_*.json`).
You can override any field at runtime with `--set`:

```bash
python scripts/train.py \
  --config configs/train.json \
  --set data.batch_size=64 \
  --set optim.lr=0.0001 \
  --set augment.enabled=false
```

If you omit `--config`, each command loads its default file from `configs/`.

To auto-push the best checkpoint, set `hub.enabled=true` and `hub.repo_id` in
`configs/train.json`. You can scope uploads with `hub.path_in_repo` (supports
`{run_name}`), control artifacts with `hub.upload_include`, and provide a token via
`hub.token_env` or your cached HF login.

## Data and Storage

Local folders used by default:

- `local_data/`: prepared dataset with `images/` and `meta/`.
- `golden_dataset/`: curated labels and images for analysis/validation.
- `filtered_dataset_osv5m/`: mined OSV-5M detections.
- `runs/`: training outputs, live runs, and analysis artifacts.

Optional S3 layout when mining with `s3_bucket` enabled:

```text
runs/
  osv5m_cpu/
    w00/
      test/
        images/
        annotated/
        state/
          cursor.json
          processed_ids.txt
          filtered.csv
```

## Results (placeholders)

Add benchmark numbers and visuals as they become available.

| Model | Top-1 | Top-5 | Notes |
| --- | --- | --- | --- |
| baseline | -- | -- | replace with metrics |

![Top errors gallery](docs/assets/top_errors_gallery.png)

## Project Layout

- `bollards/`: core library (data, models, pipelines, utils)
- `bollards/pipelines/`: per-pipeline entrypoints (`analyze`, `train`, `osv5m`, `local_dataset`, `live_screen`)
- `bollards/utils/`: shared helpers (config loader, io, visuals, runtime)
- `scripts/`: CLI entrypoints
- `configs/`: default configs and shared includes
- `tests/`: unit tests
- `runs/`: outputs (checkpoints, logs, analysis, live runs)

## Development

- Install dev deps: `pip install -r requirements.dev.txt`
- Lint: `ruff check .`
- Tests: `python -m unittest discover -s tests -p "test_*.py"`

## Roadmap

- Class activation maps for interpretability.
- Transformers in live mode.
- Add poles and road signs as additional categories.

## License

See `LICENSE`.
