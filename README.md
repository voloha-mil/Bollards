export PYTORCH_ENABLE_MPS_FALLBACK=1

http://huggingface.co/maco018/YOLOv12_traffic-delineator


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
