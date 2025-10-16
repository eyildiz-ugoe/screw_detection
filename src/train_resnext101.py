"""Command-line interface for training the ResNeXt101 screw classifier."""
from __future__ import annotations

import argparse
from pathlib import Path

from pipelines import MissingDependencyError, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True, help="Directory with training images arranged by class")
    parser.add_argument("--val-dir", type=Path, required=True, help="Directory with validation images arranged by class")
    parser.add_argument("--output-weights", type=Path, default=None, help="Where to store the best model weights")
    parser.add_argument("--log-dir", type=Path, default=None, help="Optional TensorBoard log directory")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--freeze-layers", type=int, default=0, help="Freeze this many layers from the backbone")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate for Adam")
    parser.add_argument("--initial-weights", type=Path, default=None, help="Optional checkpoint to start from")
    return parser


def main(args: list[str] | None = None) -> None:
    parser = build_parser()
    namespace = parser.parse_args(args=args)

    try:
        train_model(
            "resnext101",
            train_dir=namespace.train_dir,
            val_dir=namespace.val_dir,
            output_weights=namespace.output_weights,
            log_dir=namespace.log_dir,
            batch_size=namespace.batch_size,
            epochs=namespace.epochs,
            freeze_layers=namespace.freeze_layers,
            learning_rate=namespace.learning_rate,
            initial_weights=namespace.initial_weights,
        )
    except MissingDependencyError as exc:
        parser.error(str(exc))


if __name__ == "__main__":  # pragma: no cover
    main()
