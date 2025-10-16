
# **DCNN-Based Screw Detection for Automated Disassembly Processes**

This repository provides the source code and resources associated with the paper *[DCNN-Based Screw Detection for Automated Disassembly Processes](https://ieeexplore.ieee.org/abstract/document/9067965)*. The implementation uses Python 3, Keras and TensorFlow, integrating advanced deep learning models for screw detection in automated disassembly workflows. The detection pipeline leverages Hough Transform to identify screw candidates, followed by classification into screws or artifacts using an integrated model based on Xception and InceptionV3.

![Screw Detection Sample](assets/sample.png)

## **Repository Contents**
- Source code for the Screw Detector using Xception and InceptionV3.
- Pre-trained models and weights.
- Dataset for training and evaluation.

If this work is helpful in your research, please consider citing this repository. You can access the dataset [here](https://zenodo.org/records/4727706) and the model weights [here](https://zenodo.org/records/10474868).

---

## **Citation**
To cite this work, please use the following BibTeX entry:

```bibtex
@inproceedings{yildiz2019dcnn,
  title={DCNN-Based Screw Detection for Automated Disassembly Processes},
  author={Yildiz, Erenus and W{"o}rg{"o}tter, Florentin},
  booktitle={2019 15th International Conference on Signal-Image Technology \& Internet-Based Systems (SITIS)},
  pages={187--192},
  year={2019},
  organization={IEEE}
}
```

## **Installation and Usage**

For standalone usage, such as result generation and evaluation:
1. Clone this repository.
2. (Optional) Run `python setup.py` to download the dataset and pretrained Xception weights. The helper expects an
   available RAR extractor (`unrar` or the `rarfile` Python package). If no extractor is available the script
   will abort with a descriptive error – in that case download and extract the archives manually.
3. Run the precision/recall evaluation on the provided sample detections:
   ```bash
   python evaluate/evaluate_detection.py --det_path evaluate/det_2keras.txt --gt_path evaluate/gt_test.txt --no-plot
   ```
   The evaluation script is implemented using only the Python standard library so it works in minimal environments.
4. Run the unit tests to verify the installation:
   ```bash
   python -m unittest discover -s tests -p 'test_*.py'
   ```
5. The scripts in `src/` now expose command-line interfaces so the dataset and weight locations can be
   provided at runtime. Use ``python src/train_xception.py --help`` (or the equivalent for another architecture)
   to view the supported options.

### Training and classifier evaluation

The model training utilities depend on TensorFlow 2.x and OpenCV. Install them with::

```bash
pip install tensorflow opencv-python
```

Each architecture has a dedicated helper in ``src/``. For example, to train an Xception backbone::

```bash
python src/train_xception.py --train-dir /path/to/train --val-dir /path/to/val --output-weights models/xception.h5
```

After training you can evaluate a weights file on two folders of images (positive and negative examples)::

```bash
python src/prediction_xception.py --weights models/xception.h5 --screw-dir data/screw --non-screw-dir data/none
```

The commands print the confusion matrix and accuracy using a configurable decision threshold. All scripts share
the same options so swapping to another architecture only requires changing the module name.

---

## **Model Evaluation**
The repository allows comparison of the integrated model against two baseline approaches:
1. A combination of one TensorFlow model and one Keras model.
2. A combination of two Keras models.

The repository ships the detection outputs (`det_2tf.txt` and `det_2keras.txt`) used in the paper together with the
ground-truth annotations (`gt_test.txt`). These files allow reproducing the precision/recall curves without requiring the
pre-trained neural network weights. Use the command shown in the installation section to run the evaluation.  The scripts in
`evaluate/evaluate_classifiers_*.py` still implement the original Hough-circle + classifier pipeline; however they require a
TensorFlow 1.x environment and paths to the dataset that are not part of this repository. Consider them historical references.

---

## **Contact**
For questions or support, you can:
- Open an issue on the GitHub repository.
---
