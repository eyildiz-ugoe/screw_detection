"""Evaluate a trained DenseNet201 classifier on screw vs. non-screw folders."""
from __future__ import annotations

import argparse
from pathlib import Path

from pipelines import MissingDependencyError, evaluate_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Path to the trained weights file")
    parser.add_argument("--screw-dir", type=Path, required=True, help="Directory with screw images")
    parser.add_argument("--non-screw-dir", type=Path, required=True, help="Directory with non-screw images")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for the screw class")
    return parser


def main(args: list[str] | None = None) -> None:
    parser = build_parser()
    namespace = parser.parse_args(args=args)

    try:
        metrics = evaluate_directory(
            "densenet201",
            weights_path=namespace.weights,
            screw_dir=namespace.screw_dir,
            non_screw_dir=namespace.non_screw_dir,
            threshold=namespace.threshold,
        )
    except MissingDependencyError as exc:
        parser.error(str(exc))
    else:
        print("==============================")
        print("TP: {tp:.0f} TN: {tn:.0f} FP: {fp:.0f} FN: {fn:.0f}".format(**metrics))
        print(f"Accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":  # pragma: no cover
    main()
