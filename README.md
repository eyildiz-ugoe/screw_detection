# "DNN-Based Screw Detection for Automated Disassembly Processes"

This is an implementation of Screw Detectoron Python 3, Keras, TensorFlow and ROS. The scheme uses Hough Transform to get the candidates and then runs the integrated model to classify the candidates into screws and artifacts. The integrated model is based on Xception and InceptionV3.

Publicized code for the paper [DCNN-Based Screw Detection for Automated Disassembly Processes](https://ieeexplore.ieee.org/abstract/document/9067965)

![Screw Detection Sample](assets/sample.png)

The repository includes:

    - Source code of Screw Detector built on Xception and InceptionV3.
    - Models
    - Dataset

The code is documented and designed to be easy to extend. If you use it in your research, please consider citing this repository (bibtex will be below later on). The dataset was created by using the Screw Detector in the offline mode, which can be triggered once the ROS-Node is up and running. You can use this mode to collect images for your own dataset.

# Citation
Use this bibtex to cite this repository:

@inproceedings{yildiz2019dcnn,
  title={DCNN-Based Screw Detection for Automated Disassembly Processes},
  author={Yildiz, Erenus and W{\"o}rg{\"o}tter, Florentin},
  booktitle={2019 15th International Conference on Signal-Image Technology \& Internet-Based Systems (SITIS)},
  pages={187--192},
  year={2019},
  organization={IEEE}
}

# Contributing

Contributions to this repository are welcome. Examples of things you can contribute:

    - Training on other datasets.
    - Accuracy Improvements.
    - Visualizations and examples.

# R0S-Based Installation

If you want to use the node directly on your system via ROS, follow the below steps:

1. Clone this repository
2. Install ROS (tested only on Melodic) from its official website.
3. Install python3, tensorflow-gpu==1.9.0, opencv-python==3.4.3.18 preferably via pip.
3. Download the weights: https://owncloud.gwdg.de/index.php/s/PJIPYTvBteXlqhv (Extract it and change the path in `src/candidate_generator.py` accordingly.
4. Download the dataset (optional): https://owncloud.gwdg.de/index.php/s/AJ7W6t1nrIxYnol
5. Since the code is ROS-based, you'll need an RGB Camera and its ROS node which publishes RGB images. If you have these two, then you need to modify the following files to get the code working with your own camera:
   - `launch/screw_detection.launch`
   - `src/candidate_generator.py`
6. The command to run: `roslaunch screw_detection screw_detection.launch`

# Standalone Installation

If you are only interested in the results and evaluation, follow the below steps:

1. Clone this repository.
2. Install python3, tensorflow-gpu==1.9.0, opencv-python==3.4.3.18 preferably via pip, as well as other required packages if they are asked for.
3. Download the weights: https://owncloud.gwdg.de/index.php/s/PJIPYTvBteXlqhv
4. Download the dataset: https://owncloud.gwdg.de/index.php/s/AJ7W6t1nrIxYnol
5. Change the paths in each `.py`  file as you run since you'll probably face a path error, which you can then fix by entering your path of extraction.

# Evaluating the models

We compare our integrated model against two best performing models which are using 1 Tensorflow model + 1 Keras model and 2 Keras models, respectively. In order to evaluate all, we need to first run these two networks and save the detection results to a folder, so let's do the following:

`cd evaluate & mkdir result_images`

Alright. Let's run the networks one by one and wait for them to save the results, creating `det_2keras.txt` and `det_2tf.txt`:

`python3 evaluate_classifiers_2tf.py`

`python3 evaluate_classifiers_2keras.py`

2. Following the creation of `det_2keras.txt` and `det_2tf.txt`, which are the detection results of the above executions, now the actual evaluation could be executed, which would compare these two with the ground truth annotations and pop up the precision-recall curve:

`python3 evaluate_detection.py --det_path det_2tf.txt --gt_path gt_test.txt`

`python3 evaluate_detection.py --det_path det_2keras.txt --gt_path gt_test.txt`

3. Finally, our integrated model can be evaluated with the following:

`python3 evaluate_classifiers_integrated.py`

which would yield:

`maximum accuracy:  0.9897316219369895
TP:  990  
TN:  3247  
FP:  10  
FN:  38
accuracy:  0.9887981330221703`

# Contact
For questions, you can create an issue on Github or simply contact eyildiz@gwdg.de
