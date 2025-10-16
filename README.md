
# 🔩 **DCNN-Based Screw Detection for Automated Disassembly Processes**

This repository provides the complete source code and resources for the paper *[DCNN-Based Screw Detection for Automated Disassembly Processes](https://ieeexplore.ieee.org/abstract/document/9067965)*. 

**Key Features:**
- 🎯 **6 Pre-trained CNN Models**: Xception, InceptionV3, InceptionResNetV2, DenseNet201, ResNet101V2, ResNeXt101
- 🔬 **Three Usage Modes**: Classification only, Standalone detection, ROS integration
- 📊 **98.7% Accuracy**: State-of-the-art screw classification
- 🚀 **Modern TensorFlow 2.x**: Updated from legacy TF 1.x codebase
- 🐍 **Pure Python**: No ROS required for standalone usage

![Screw Detection Sample](assets/sample.png)

---

## 📋 **Table of Contents**

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage Modes](#usage-modes)
  - [Mode 1: Classification Testing](#mode-1-classification-testing)
  - [Mode 2: Standalone Detection](#mode-2-standalone-detection)
  - [Mode 3: ROS Integration](#mode-3-ros-integration)
- [Training Models](#training-models)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Available Models](#available-models)
- [Citation](#citation)
- [Contact](#contact)

---

## ⚡ **Quick Start**

### Test Pre-trained Model (No Dataset Required)
```bash
git clone https://github.com/eyildiz-ugoe/screw_detection.git
cd screw_detection
pip install -r requirements.txt

# Run detection evaluation on sample data
python evaluate/evaluate_detection.py \
  --det_path evaluate/det_2keras.txt \
  --gt_path evaluate/gt_test.txt \
  --no-plot
```
**Expected output**: `AP = 0.1572`

### Full Detection on Your Own Image
```bash
# Download models first (see Installation section)
python standalone_detection.py \
  --image your_image.jpg \
  --model1 models/xception.h5 \
  --model2 models/inceptionv3.h5 \
  --output result.jpg
```

---

## 🔧 **Installation**

### 1. Clone Repository
```bash
git clone https://github.com/eyildiz-ugoe/screw_detection.git
cd screw_detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.7+
- TensorFlow 2.x (GPU support included)
- OpenCV
- NumPy
- Matplotlib (optional, for plotting)

> **Note**: TensorFlow 2.20+ requires cuDNN 9.3.0+. If you have an older cuDNN (e.g., 8.9.7), either upgrade cuDNN or use CPU mode by setting `CUDA_VISIBLE_DEVICES=""`.

### 3. Download Dataset and Pre-trained Models

**Option A: Automated Download** (requires `unrar`):
```bash
python setup.py
```

**Option B: Manual Download**:
- Dataset: [ScrewDTF dataset](https://zenodo.org/records/4727706) (~1.8GB)
- Pre-trained weights: [Model weights](https://zenodo.org/records/10474868) (~2.1GB)

> **Note**: The dataset from Zenodo comes in TFRecord format. After extraction, run `python ScrewDTF/tfreader.py` to convert TFRecords to JPG images organized by label.

**Expected folder structure after extraction and conversion**:
```
screw_detection/
├── ScrewDTF/              # Dataset
│   ├── Train/
│   │   ├── label_0/       # Non-screw images
│   │   └── label_1/       # Screw images
│   ├── Eval/              # Validation set
│   └── Test/              # Test set
└── models/                # Pre-trained weights
    ├── xception.h5
    ├── inceptionv3.h5
    ├── inceptionResNetv2.h5
    ├── densenet201.h5
    ├── resnet101v2.h5
    └── resnext101.h5
```

---

## 🎯 **Usage Modes**

This repository supports three different usage modes depending on your needs:

### **Mode 1: Classification Testing**
Test pre-trained classifiers on pre-cropped screw images.

**Use when**: Testing model accuracy, comparing architectures, benchmarking.

**Example**:
```bash
# Test Xception model
python src/prediction_xception.py \
  --weights models/xception.h5 \
  --screw-dir ScrewDTF/Test/label_1 \
  --non-screw-dir ScrewDTF/Test/label_0

# Output:
# ==============================
# TP: 1843 TN: 1974 FP: 8 FN: 42
# Accuracy: 0.9871
```

**Test all 6 models at once**:
```bash
python test_all_classifiers.py
```

**Available architectures**: `xception`, `inceptionv3`, `inceptionresnetv2`, `densenet201`, `resnet101v2`, `resnext101`

---

### **Mode 2: Standalone Detection** 🆕
End-to-end screw detection on any image (no ROS required).

**Use when**: Processing static images, batch detection, development/debugging.

**Pipeline**:
1. Find circular candidates (Hough Transform)
2. Extract patches around candidates
3. Classify with CNN ensemble (Xception + InceptionV3)
4. Output bounding boxes with confidence scores

**Basic usage**:
```bash
python standalone_detection.py \
  --image your_image.jpg \
  --model1 models/xception.h5 \
  --model2 models/inceptionv3.h5 \
  --output result.jpg
```

**Example output**:
```
Finding circular candidates...
Found 3 circular candidates
Classifying candidates...
Detected 1 screws

Detected 1 screws:
  1. Position: (832, 82), Radius: 20, Confidence: 0.887
```

**Advanced options**:
```bash
# Adjust detection threshold
python standalone_detection.py --image input.jpg --threshold 0.3 --output result.jpg

# Tune Hough parameters for different screw sizes
python standalone_detection.py --image input.jpg \
  --min-radius 10 \
  --max-radius 50 \
  --hough-upper 150 \
  --hough-lower 30 \
  --output result.jpg

# Use different model combination
python standalone_detection.py --image input.jpg \
  --model1 models/resnet101v2.h5 \
  --model2 models/densenet201.h5 \
  --output result.jpg

# Display result window
python standalone_detection.py --image input.jpg --show

# Use GPU
python standalone_detection.py --image input.jpg --use-gpu --output result.jpg
```

**Batch processing**:
```bash
#!/bin/bash
for img in input_images/*.jpg; do
    output="results/$(basename $img)"
    python standalone_detection.py --image "$img" --output "$output"
done
```

---

### **Mode 3: ROS Integration** 🤖
Full detection pipeline integrated with ROS (Robot Operating System).

**Use when**: Integrating with robotics hardware, real-time camera streams, ROS-based systems.

**File**: `ros_detection.py` (formerly `candidate_generator.py`)

**Requirements**:
- ROS (Kinetic/Melodic/Noetic)
- TensorFlow 1.x (for compatibility with original system)
- ROS cv_bridge, sensor_msgs

**Features**:
- Subscribes to ROS Image topics
- Real-time Hough circle detection
- CNN classification
- Publishes detection results to ROS topics
- Dynamic reconfigure for parameter tuning

> **Note**: This mode requires a full ROS workspace setup and is intended for robotics integration. For standalone testing, use **Mode 2** instead.

---

## 🎓 **Training Models**

Train your own classifiers on custom datasets:

```bash
# Train Xception (71×71 images)
python src/train_xception.py \
  --train-dir ScrewDTF/Train \
  --val-dir ScrewDTF/Eval \
  --output-weights models/my_xception.h5 \
  --epochs 16 \
  --batch-size 8

# Train InceptionV3 (139×139 images)
python src/train_inceptionv3.py \
  --train-dir ScrewDTF/Train \
  --val-dir ScrewDTF/Eval \
  --output-weights models/my_inceptionv3.h5

# View all training options
python src/train_xception.py --help
```

**Training options**:
- `--train-dir`: Training images organized by class (label_0/, label_1/)
- `--val-dir`: Validation images organized by class
- `--output-weights`: Where to save the best model
- `--epochs`: Number of training epochs (default: 16)
- `--batch-size`: Mini-batch size (default: 8)
- `--learning-rate`: Adam learning rate (default: 1e-4)
- `--freeze-layers`: Freeze N layers from backbone
- `--initial-weights`: Start from existing checkpoint

---

## 📊 **Evaluation**

### Classifier Evaluation

Evaluate a trained model on test images:

```bash
python src/prediction_xception.py \
  --weights models/xception.h5 \
  --screw-dir ScrewDTF/Test/label_1 \
  --non-screw-dir ScrewDTF/Test/label_0 \
  --threshold 0.5
```

**Output**:
```
==============================
TP: 1843 TN: 1974 FP: 8 FN: 42
Accuracy: 0.9871
```

### Detection Pipeline Evaluation

Reproduce the paper's precision/recall metrics:

```bash
# Evaluate Keras-based detector
python evaluate/evaluate_detection.py \
  --det_path evaluate/det_2keras.txt \
  --gt_path evaluate/gt_test.txt

# Evaluate TensorFlow-based detector  
python evaluate/evaluate_detection.py \
  --det_path evaluate/det_2tf.txt \
  --gt_path evaluate/gt_test.txt
```

**What these files are**:
- `det_2keras.txt` / `det_2tf.txt`: Pre-computed detection results from the paper
- `gt_test.txt`: Ground truth bounding box annotations
- Allows reproducing precision/recall curves without running full detection

---

## 🗂️ **Project Structure**

```
screw_detection/
│
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git exclusions
├── conftest.py                         # Import path setup
├── setup.py                            # Dataset/weights downloader
│
├── standalone_detection.py             # 🆕 End-to-end detection (no ROS)
├── ros_detection.py                    # ROS-integrated detection
├── test_all_classifiers.py             # Test all 6 models
│
├── src/                                # Source code
│   ├── pipelines/
│   │   ├── __init__.py
│   │   └── classification.py          # Core training/inference logic
│   │
│   ├── resnet.py                       # ResNeXt implementation
│   │
│   ├── train_*.py                      # Training CLIs (6 architectures)
│   └── prediction_*.py                 # Inference CLIs (6 architectures)
│
├── evaluate/                           # Evaluation scripts & data
│   ├── evaluate_detection.py          # Precision/recall computation
│   ├── det_2keras.txt                 # Sample detection results
│   ├── det_2tf.txt                    # Sample detection results
│   └── gt_test.txt                    # Ground truth annotations
│
├── tests/                              # Unit tests
│   ├── test_evaluate_detection.py
│   └── test_pipelines_classification.py
│
├── assets/                             # Images for documentation
│   └── sample.png
│
# Data folders (not in git, download separately):
├── models/                             # Pre-trained weights (.h5)
└── ScrewDTF/                           # Training/test dataset
    ├── Train/
    ├── Eval/
    └── Test/
```

---

## 🏆 **Available Models**

All models are trained on the ScrewDTF dataset and achieve high accuracy:

| Model | Input Size | Weights Size | Test Accuracy | Use Case |
|-------|-----------|--------------|---------------|----------|
| **Xception** ⭐ | 71×71 | 80 MB | **98.71%** | Best overall, fast inference |
| **ResNet101V2** ⭐ | 64×64 | 489 MB | **97.60%** | High accuracy, smallest input |
| ResNeXt101 | 64×64 | 485 MB | 92.29% | Good accuracy |
| InceptionResNetV2 | 139×139 | 625 MB | 91.18% | Larger input, more detail |
| InceptionV3 | 139×139 | 251 MB | 91.08% | Good for ensemble |
| DenseNet201 | 221×221 | 212 MB | 89.60% | Largest input size |

**Model weights**: Download from [Zenodo](https://zenodo.org/records/10474868)

---

## 🧪 **Testing**

Run unit tests to verify installation:

```bash
# All tests
python -m pytest tests/ -v

# Specific test
python -m pytest tests/test_evaluate_detection.py -v

# Test all classifiers (requires dataset and models)
python test_all_classifiers.py
```

---

## 📚 **Citation**

If this work is helpful in your research, please cite:

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

**Paper**: [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/9067965)

**Dataset**: [ScrewDTF on Zenodo](https://zenodo.org/records/4727706)

**Model Weights**: [Pre-trained models on Zenodo](https://zenodo.org/records/10474868)

---

## 💡 **GPU vs CPU Usage**

**GPU Mode** (default, requires CUDA/cuDNN):
```bash
python standalone_detection.py --image input.jpg --use-gpu --output result.jpg
```

**CPU Mode** (recommended if CUDA version mismatch):
```bash
# Method 1: Environment variable
CUDA_VISIBLE_DEVICES="" python standalone_detection.py --image input.jpg --output result.jpg

# Method 2: No GPU flag (default in most scripts)
python standalone_detection.py --image input.jpg --output result.jpg
```

> **Important**: TensorFlow 2.20+ requires cuDNN 9.3.0+. If you have cuDNN 8.9.7 or encounter GPU errors, use CPU mode or downgrade TensorFlow: `pip install tensorflow==2.15`

---

## 📞 **Contact & Support**

- **Issues**: [GitHub Issues](https://github.com/eyildiz-ugoe/screw_detection/issues)
- **Author**: Erenus Yildiz
- **Institution**: University of Göttingen

---

## 📝 **License**

See repository for license details.

---

## 🎉 **What's New in This Version**

This is a refactored version with significant improvements:

✅ **TensorFlow 2.x Support**: Modernized from legacy TF 1.x  
✅ **Standalone Detection**: No ROS required for basic usage  
✅ **Unified API**: Consistent interfaces across all architectures  
✅ **Better Documentation**: Single comprehensive README  
✅ **Cleaner Structure**: Organized files and clear naming  
✅ **Unit Tests**: Automated testing for reliability  
✅ **ResNeXt101 Fix**: Fixed import issues  
✅ **GPU/CPU Flexibility**: Easy switching between modes  

**Legacy files** (TF 1.x, kept for reference):
- `evaluate/evaluate_classifiers_*.py` - Original evaluation scripts (require TF 1.x)
- Kept for historical reference; not needed for normal usage

---
