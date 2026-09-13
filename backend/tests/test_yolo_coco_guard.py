"""COCO-checkpoint guard: pure-function tests, no model loading required."""
from backend.services.yolo_service import COCO_OVERLAP_THRESHOLD, coco_overlap, is_coco_like

FULL_COCO_80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
]


def test_full_coco_checkpoint_is_refused():
    assert is_coco_like(set(FULL_COCO_80)) is True


def test_sonar_class_names_are_not_coco():
    sonar_names = {"wreck", "debris", "anchor_chain", "unknown_object", "pipeline"}
    assert is_coco_like(sonar_names) is False
    assert coco_overlap(sonar_names) == set()


def test_mixed_but_mostly_sonar_names_pass():
    # A handful of incidental overlaps (e.g. a class legitimately named "boat")
    # shouldn't trip the guard if the checkpoint is mostly sonar-specific.
    names = {"wreck", "debris", "anchor_chain", "unknown_object", "pipeline", "boat"}
    overlap_fraction = len(coco_overlap(names)) / len(names)
    assert overlap_fraction < COCO_OVERLAP_THRESHOLD
    assert is_coco_like(names) is False


def test_empty_names_not_flagged():
    assert is_coco_like(set()) is False
