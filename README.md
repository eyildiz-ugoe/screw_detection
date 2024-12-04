
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
2. Go to sd_venv folder and run `bash setup.py` which will install the dependencies.
3. Get back to the main project folder, and run `python setup.py` to download model and weights.
4. Update paths in the Python files as necessary to avoid path errors during execution.

---

## **Model Evaluation**
The repository allows comparison of the integrated model against two baseline approaches:
1. A combination of one TensorFlow model and one Keras model.
2. A combination of two Keras models.

Follow these steps for evaluation:

1. Create a directory for storing detection results:
   ```bash
   cd evaluate
   mkdir result_images
   ```
2. Run the two baseline models sequentially to generate detection results:
   ```bash
   python3 evaluate_classifiers_2tf.py
   python3 evaluate_classifiers_2keras.py
   ```
   These scripts produce the detection files `det_2tf.txt` and `det_2keras.txt`.

3. Evaluate the baseline models against ground truth annotations, and generate precision-recall curves:
   ```bash
   python3 evaluate_detection.py --det_path det_2tf.txt --gt_path gt_test.txt
   python3 evaluate_detection.py --det_path det_2keras.txt --gt_path gt_test.txt
   ```

4. Evaluate the integrated model:
   ```bash
   python3 evaluate_classifiers_integrated.py
   ```
   Sample results:
   ```
   Maximum accuracy: 0.9897316219369895
   TP: 990  
   TN: 3247  
   FP: 10  
   FN: 38  
   Accuracy: 0.9887981330221703
   ```

---

## **Contact**
For questions or support, you can:
- Open an issue on the GitHub repository.
---
