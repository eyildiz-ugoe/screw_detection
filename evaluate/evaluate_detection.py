"""Precision/recall evaluation helpers for the screw detection benchmark.

The original repository shipped a script that depended on :mod:`numpy` and
Matplotlib.  The execution environment used for the automated tests does not
provide these libraries, therefore the implementation below uses nothing but the
Python standard library.  The behaviour matches the original script which was
based on the Pascal VOC devkit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:  # pragma: no cover - optional dependency used only for plotting
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None


BBox = Tuple[float, float, float, float]


def _linspace(start: float, stop: float, step: float) -> Iterable[float]:
    """Return an iterator similar to ``numpy.arange`` for small ranges."""

    current = start
    while current <= stop + 1e-12:  # be forgiving with floating point noise
        yield current
        current += step


@dataclass
class VocEvalResult:
    """Container returned by :func:`voc_eval`."""

    recall: List[float]
    precision: List[float]
    ap: float


def voc_ap(rec: Sequence[float], prec: Sequence[float], use_07_metric: bool = False) -> float:
    """Compute Average Precision following the Pascal VOC protocol."""

    if use_07_metric:
        ap = 0.0
        for threshold in _linspace(0.0, 1.0, 0.1):
            candidates = [p for r, p in zip(rec, prec) if r >= threshold - 1e-12]
            ap += (max(candidates) if candidates else 0.0) / 11.0
        return ap

    mrec = [0.0] + list(rec) + [1.0]
    mpre = [0.0] + list(prec) + [0.0]

    for index in range(len(mpre) - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])

    ap = 0.0
    for i in range(len(mrec) - 1):
        if mrec[i + 1] != mrec[i]:
            ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
    return ap


def _load_ground_truth(groundtruth_path: Path) -> Tuple[int, Dict[str, Dict[str, List]]]:
    npos = 0
    class_recs: Dict[str, Dict[str, List]] = {}
    with groundtruth_path.open("r", encoding="utf-8") as groundtruth_file:
        for line in groundtruth_file:
            if ".png " not in line:
                continue
            line_split = line.strip().split(".png ")
            image_id = line_split[0].split("/")[-1]
            boxes = line_split[1].split(" ")
            bbox: List[BBox] = []
            for box in boxes:
                if not box:
                    continue
                xmin, ymin, xmax, ymax, _ = box.split(",")
                bbox.append((float(xmin), float(ymin), float(xmax), float(ymax)))
            det = [False] * len(bbox)
            npos += len(det)
            class_recs[image_id] = {
                "bbox": bbox,
                "det": det,
            }
    return npos, class_recs


def _iou(det_box: BBox, gt_box: BBox) -> float:
    xmin_det, ymin_det, xmax_det, ymax_det = det_box
    xmin_gt, ymin_gt, xmax_gt, ymax_gt = gt_box

    ixmin = max(xmin_gt, xmin_det)
    iymin = max(ymin_gt, ymin_det)
    ixmax = min(xmax_gt, xmax_det)
    iymax = min(ymax_gt, ymax_det)

    iw = max(ixmax - ixmin + 1.0, 0.0)
    ih = max(iymax - iymin + 1.0, 0.0)
    inter = iw * ih

    det_area = (xmax_det - xmin_det + 1.0) * (ymax_det - ymin_det + 1.0)
    gt_area = (xmax_gt - xmin_gt + 1.0) * (ymax_gt - ymin_gt + 1.0)
    union = det_area + gt_area - inter

    if union <= 0:
        return 0.0
    return inter / union


def voc_eval(
    det_path: Path,
    groundtruth_path: Path,
    ovthresh: float = 0.5,
    use_07_metric: bool = False,
) -> VocEvalResult:
    det_path = Path(det_path)
    groundtruth_path = Path(groundtruth_path)

    npos, class_recs = _load_ground_truth(groundtruth_path)

    with det_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    detections = [x.strip().split(" ") for x in lines if x.strip()]
    image_ids = [fields[0] for fields in detections]
    confidence = [float(fields[1]) for fields in detections]
    boxes = [tuple(float(v) for v in fields[2:6]) for fields in detections]

    sorted_indices = sorted(range(len(confidence)), key=lambda idx: confidence[idx], reverse=True)
    boxes = [boxes[idx] for idx in sorted_indices]
    image_ids = [image_ids[idx] for idx in sorted_indices]

    tp = [0.0] * len(boxes)
    fp = [0.0] * len(boxes)

    for idx, box in enumerate(boxes):
        record = class_recs.get(image_ids[idx])
        if not record:
            fp[idx] = 1.0
            continue

        bbox_list: List[BBox] = record["bbox"]
        det_flags: List[bool] = record["det"]

        overlaps = [_iou(box, gt_box) for gt_box in bbox_list]
        ovmax = max(overlaps) if overlaps else 0.0
        jmax = overlaps.index(ovmax) if overlaps else -1

        if ovmax > ovthresh and jmax >= 0 and not det_flags[jmax]:
            tp[idx] = 1.0
            det_flags[jmax] = True
        else:
            fp[idx] = 1.0

    fp_cum = list(accumulate(fp))
    tp_cum = list(accumulate(tp))
    rec = [t / npos if npos else 0.0 for t in tp_cum]
    prec = [t / max(t + f, 1e-12) for t, f in zip(tp_cum, fp_cum)]
    ap = voc_ap(rec, prec, use_07_metric)

    return VocEvalResult(rec, prec, ap)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="mAP Calculation")
    parser.add_argument("--det_path", dest="det_path", default="det_2keras.txt", help="Detection file path")
    parser.add_argument("--gt_path", dest="gt_path", default="gt_test.txt", help="Ground-truth file path")
    parser.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="Disable the precision/recall plot",
    )
    parser.set_defaults(plot=True)
    return parser.parse_args()


def _maybe_plot(rec: Sequence[float], prec: Sequence[float]) -> None:  # pragma: no cover - optional plotting
    if plt is None:
        raise RuntimeError("matplotlib is not available in this environment")
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall curve")
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    print("VOC07 metric? " + ("Yes" if False else "No"))
    result = voc_eval(Path(args.det_path), Path(args.gt_path), ovthresh=0.5, use_07_metric=False)
    print(f"AP = {result.ap:.4f}")
    if args.plot:
        _maybe_plot(result.recall, result.precision)
