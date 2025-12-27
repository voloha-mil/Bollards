# Bollards

Country classification from Google Street View bollards.

Bollards is a project for country classification based on bollard appearance in
Street View imagery. It uses a detector to find bollards and a classifier on
top of the detected crops to predict the country.

## Analysis

`scripts/analyze_run.py` generates a report directory under `runs/analyze_run/`
with summary tables, plots, and galleries for a given dataset and checkpoint.

## Live screen classification

The live screen pipeline captures a region from your display and runs the
classifier on detected bollards. Configure triggers, capture region, and
outputs in `configs/live_screen.json`, and disable the viewer for headless runs.

## Training

`scripts/train.py` trains the country classifier from prepared crops and writes
checkpoints and TensorBoard logs under `runs/bollard_country/`.

## Data mining

`scripts/mine_osv5m.py` runs a YOLO detector over OSV-5M shards and writes
filtered detections plus optional image/annotation artifacts for downstream
dataset prep.

## Development

- Install dev deps: `pip install -r requirements.dev.txt`
- Lint: `ruff check .`
- Tests: `python -m unittest discover -s tests -p "test_*.py"`

## Layout

- `bollards/`: core library (data, models, training, pipelines, io)
- `scripts/`: CLI entrypoints
- `configs/`: default configs
- `tests/`: unit tests

## S3 layout

- `runs/` (root for data-mining outputs)
  - `osv5m_cpu/` (OSV-5M mining outputs, split by worker and split)
    - `w00/`, `w01/`, ... (one folder per worker)
      - `test/` (dataset split name)
        - `images/` (original images)
        - `annotated/` (visualizations)
        - `state/`
          - `cursor.json` (resume pointer)
          - `processed_ids.txt` (dedupe + resumability)
          - `filtered.csv` (mined dataset index)

## Roadmap

- Class activation maps for interpretability.
- Transformers in live mode.
- Add poles and road signs as additional categories.
