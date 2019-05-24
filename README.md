# "DNN-Based Screw Detection for Automated Disassembly Processes"
Publicized code for the paper "DNN-Based Screw Detection for Automated Disassembly Processes"

# Requirements

- ROS Melodic
- tensorflow-gpu==1.9.0
- opencv-python==3.4.3.18
- Since the code is ROS-based, you'll need an RGB Camera and its ROS node which publishes RGB images. If you have these two, then you need to modify the following files to get the code working with your own camera:
   - `launch/screw_detection.launch`
   - `src/candidate_generator.py`

- Weights: https://owncloud.gwdg.de/index.php/s/PJIPYTvBteXlqhv (Extract it and change the path in `src/candidate_generator.py` accordingly.
