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
- [Quickstart](#quickstart)
- [Pipeline](#pipeline)
- [Workflows](#workflows)
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

### Mine OSV-5M for bollards

```bash
python scripts/mine_osv5m.py --config configs/mine_osv5m.json
```

Outputs a filtered dataset index under `filtered_dataset_osv5m/` and, when enabled,
optional S3 artifacts (images, annotations, and state).

### Prepare a local training dataset

```bash
python scripts/prepare_local_dataset.py --config configs/prepare_local_dataset.json
```

Writes `local_data/images/` and `local_data/meta/` (train/val CSVs, country maps).

### Train the classifier

```bash
python scripts/train.py --config configs/train.json
```

Checkpoints and TensorBoard logs are stored under `runs/bollard_country/`.

### Analyze a run

```bash
python scripts/analyze_run.py --config configs/analyze_run.json
```

Produces reports and galleries in `runs/analyze_run/`.

### Live screen classification

```bash
python scripts/live_screen.py --config configs/live_screen.json
```

Captures a screen region, detects bollards, and displays a prediction grid.
Outputs are written under `runs/live_screen/`.

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

- `bollards/`: core library (data, models, training, pipelines, io)
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
