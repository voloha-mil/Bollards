"""Shared constants for dataset schema and pipeline outputs."""

# Training dataset schema
PATH_COL = "image_path"
LABEL_COL = "country_id"
COUNTRY_STR_COL = "country"
META_COLS = ["x_center", "y_center", "w", "h", "conf"]
BBOX_COLS = ["x1", "y1", "x2", "y2"]

# ImageNet normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Mining pipeline (filtered.csv) fields
OSV5M_FILTERED_FIELDS_PREFIX = ["id", "split", "shard", "image_path", "annotated_path"]
OSV5M_FILTERED_FIELDS_SUFFIX = [
    "n_boxes",
    "boxes_xyxy",
    "boxes_conf",
    "boxes_cls",
    "model_repo",
    "model_file",
    "created_at_unix",
    "s3_bucket",
    "s3_prefix",
    "s3_image_key",
    "s3_annotated_key",
]

# Local dataset preparation schema
LOCAL_DATASET_REQUIRED_COLS = [
    "id",
    "split",
    "country",
    "n_boxes",
    "boxes_xyxy",
    "boxes_conf",
    "boxes_cls",
    "s3_bucket",
    "s3_image_key",
]
LOCAL_DATASET_OPTIONAL_COLS = ["s3_annotated_key"]
LOCAL_DATASET_OUT_COLS = [
    "sample_id",
    "image_path",
    "country_id",
    "country",
    "x1",
    "y1",
    "x2",
    "y2",
    "x_center",
    "y_center",
    "w",
    "h",
    "conf",
    "cls",
    "image_id",
]
