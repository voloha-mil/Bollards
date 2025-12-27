export PYTORCH_ENABLE_MPS_FALLBACK=1

http://huggingface.co/maco018/YOLOv12_traffic-delineator

### Quickstart

- Configs live in `configs/` (JSON). Override with `--set section.key=value`.
- Configs can share blocks via `"include": "other.json"` (path relative to the config).
- Mining: `python scripts/mine_osv5m.py --config configs/mine_osv5m.json --set target=500`
- Prepare local dataset: `python scripts/prepare_local_dataset.py --config configs/prepare_local_dataset.json --set num_boxes=2000`
- Training: `python scripts/train.py --config configs/train.json --set data.batch_size=128`
- Live screen: `python scripts/live_screen.py --config configs/live_screen.json --set classifier.checkpoint_path=runs/bollard_country/<run>/best.pt`

### Environment and secrets

- Copy `.env.example` to `.env` and fill in tokens/keys. `.env` is gitignored.
- CLI auto-loads `.env` if `python-dotenv` is installed (included in `requirements.txt`).
- Common variables: `HF_TOKEN`.

### Development (lint/test)

- Install minimal dev deps: `pip install -r requirements.dev.txt`
- Lint: `ruff check .`
- Tests: `python -m unittest discover -s tests -p "test_*.py"`

### Live screen classification

- Install extra deps: `pip install -r requirements.live.txt`
- Capture is driven by `trigger.mode`:
  - `hotkey` uses `trigger.hotkey` (default `<ctrl>+<shift>+b`).
  - `stdin` waits for Enter; type `quit` to exit.
- Screen region: set `capture.monitor_index` and optional `capture.region` (`left`, `top`, `width`, `height`).
- Bounding-box filters match `configs/prepare_local_dataset.json`: `filters.min_conf`, `filters.min_box_w_px`, `filters.min_box_h_px`, `filters.cls_allow`, `filters.max_boxes_per_image`.
- Outputs go to `runs/live_screen/run_*/` with `crops/`, `grid.jpg`, `session.jsonl`, and `summary.json`.
- `output.viewer_enabled=true` shows a live grid window; disable for headless runs.

### Layout

- `bollards/`: core library (data, models, training, pipelines, io)
- `scripts/`: CLI entrypoints
- `configs/`: default configs


### S3 layout (`s3://geo-bollard-ml/`)

* **`runs/`** *(root for data-mining outputs / artifacts; no model checkpoints here yet)*

  * **`osv5m_cpu/`** *(OSV-5M mining pipeline outputs, split by worker and split name)*

    * **`w00/`**, **`w01/`**, … *(one folder per worker / shard)*

      * **`test/`** *(dataset split name; re-annotated subset from the dataset’s `test` split — not “test-only” tooling)*

        * **`images/`**
          Original images (as downloaded/used by the worker) for manual browsing.
        * **`annotated/`**
          Same images but with bounding boxes rendered on top (visualization only; not the source of truth).
        * **`state/`** *(mining bookkeeping)*

          * `cursor.json` — resume pointer / progress cursor for the mining loop
          * `processed_ids.txt` — dedupe + resumability (IDs already processed)
          * `filtered.csv` — **main output table**: the mined dataset index (rows link to S3 detection files + include bbox coordinates, GPS/country labels, etc.)
