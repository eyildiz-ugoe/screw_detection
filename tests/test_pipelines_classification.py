import unittest
from pathlib import Path

from pipelines.classification import MissingDependencyError, build_classifier, ensure_tensorflow, train_model


class ClassificationPipelineDependencyTests(unittest.TestCase):
    def test_ensure_tensorflow_raises_missing_dependency(self) -> None:
        with self.assertRaises(MissingDependencyError):
            ensure_tensorflow()

    def test_build_classifier_requires_tensorflow(self) -> None:
        with self.assertRaises(MissingDependencyError):
            build_classifier("xception")

    def test_train_model_requires_tensorflow(self) -> None:
        with self.assertRaises(MissingDependencyError):
            train_model(
                "xception",
                train_dir=Path("train"),
                val_dir=Path("val"),
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
