import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List, Sequence

from PIL import Image
from ultralytics import YOLO


@dataclass(frozen=True)
class DetectionPayload:
    n_boxes: int
    boxes_xyxy_json: str
    boxes_conf_json: str
    boxes_cls_json: str


def load_yolo(weights_path: Path) -> YOLO:
    return YOLO(str(weights_path))


def run_inference_batch(
    model: YOLO,
    image_paths: Sequence[Path],
    imgsz: int,
    conf: float,
    device: str,
    batch: int,
) -> List[Any]:
    """
    Run a single batched predict call.
    Returns a list of Results (same order as image_paths).
    """
    results = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=imgsz,
        conf=conf,
        device=device,
        batch=batch,
        verbose=False,
    )
    return results


def run_inference(
    model: YOLO,
    image_path: Path,
    imgsz: int,
    conf: float,
    device: str,
) -> Any:
    # Returns Ultralytics Results list; caller can take [0]
    return model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        device=device,
        verbose=False,
    )


def extract_detection_payload(result0: Any) -> DetectionPayload:
    boxes = result0.boxes
    n_boxes = 0 if boxes is None else int(len(boxes))

    if n_boxes <= 0:
        return DetectionPayload(
            n_boxes=0,
            boxes_xyxy_json="[]",
            boxes_conf_json="[]",
            boxes_cls_json="[]",
        )

    xyxy = boxes.xyxy.cpu().numpy().tolist()
    confs = boxes.conf.cpu().numpy().tolist()
    clss = boxes.cls.cpu().numpy().tolist()

    return DetectionPayload(
        n_boxes=n_boxes,
        boxes_xyxy_json=json.dumps(xyxy),
        boxes_conf_json=json.dumps(confs),
        boxes_cls_json=json.dumps(clss),
    )


def save_annotated(result0: Any, out_path: Path) -> None:
    """
    Save an annotated visualization (boxes drawn) using Ultralytics `plot()`.
    """
    arr = result0.plot()  # numpy array, usually BGR
    if getattr(arr, "ndim", 0) == 3 and arr.shape[2] == 3:
        arr = arr[:, :, ::-1]  # BGR -> RGB

    img = Image.fromarray(arr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
