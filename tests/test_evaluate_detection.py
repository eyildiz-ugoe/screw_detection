import unittest
from pathlib import Path

from evaluate.evaluate_detection import VocEvalResult, voc_eval


class EvaluateDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.det_path = Path("evaluate/det_2keras.txt")
        self.det_tf_path = Path("evaluate/det_2tf.txt")
        self.gt_path = Path("evaluate/gt_test.txt")

    def test_voc_eval_returns_expected_ap_for_keras_detector(self) -> None:
        result = voc_eval(self.det_path, self.gt_path, ovthresh=0.5, use_07_metric=False)
        self.assertIsInstance(result, VocEvalResult)
        self.assertAlmostEqual(result.ap, 0.1572, places=4)
        self.assertEqual(len(result.precision), len(result.recall))

    def test_voc_eval_returns_expected_ap_for_tensorflow_detector(self) -> None:
        result = voc_eval(self.det_tf_path, self.gt_path, ovthresh=0.5, use_07_metric=False)
        self.assertAlmostEqual(result.ap, 0.1564, places=4)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
