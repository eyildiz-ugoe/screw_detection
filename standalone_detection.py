#!/usr/bin/env python
"""Simple end-to-end screw detection without ROS dependencies.

This script implements the complete detection pipeline:
1. Load an image
2. Find circular candidates using Hough Circle Transform
3. Extract patches around candidates
4. Classify patches using trained CNNs
5. Output detection results with bounding boxes

Usage:
    python detect_screws.py --image path/to/image.jpg --model1 models/xception.h5 --model2 models/inceptionv3.h5
    python detect_screws.py --image path/to/image.jpg --model1 models/xception.h5 --model2 models/inceptionv3.h5 --output results.jpg
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# Add src to path for imports
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipelines import build_classifier, MissingDependencyError


class ScrewDetector:
    """End-to-end screw detector combining Hough circles and CNN classification."""
    
    def __init__(
        self,
        model1_path: Path,
        model2_path: Path,
        hough_upper_threshold: int = 100,
        hough_lower_threshold: int = 50,
        hough_min_radius: int = 5,
        hough_max_radius: int = 30,
        detection_threshold: float = 0.5,
        use_gpu: bool = False,
    ):
        """Initialize the detector with two models.
        
        Args:
            model1_path: Path to Xception model weights (71x71 input)
            model2_path: Path to InceptionV3 model weights (139x139 input)
            hough_upper_threshold: Upper threshold for Canny edge detector
            hough_lower_threshold: Lower threshold for circle accumulator
            hough_min_radius: Minimum circle radius to detect
            hough_max_radius: Maximum circle radius to detect
            detection_threshold: Confidence threshold for classifying as screw
            use_gpu: Whether to use GPU (default: CPU only)
        """
        import os
        if not use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        print(f"Loading Xception model from {model1_path}...")
        self.model1 = build_classifier(
            "xception",
            num_classes=2,
            base_weights=None,
            weights_path=model1_path,
        )
        
        print(f"Loading InceptionV3 model from {model2_path}...")
        self.model2 = build_classifier(
            "inceptionv3",
            num_classes=2,
            base_weights=None,
            weights_path=model2_path,
        )
        
        self.image_size1 = (71, 71)
        self.image_size2 = (139, 139)
        
        # Hough circle parameters
        self.hough_upper_threshold = hough_upper_threshold
        self.hough_lower_threshold = hough_lower_threshold
        self.hough_min_radius = hough_min_radius
        self.hough_max_radius = hough_max_radius
        
        self.detection_threshold = detection_threshold
        
        print("Detector initialized successfully!")
    
    def find_candidates(self, image: np.ndarray) -> List[Tuple[int, int, int]]:
        """Find circular candidates using Hough Circle Transform.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            List of (x, y, radius) tuples for detected circles
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Hough Circle Transform
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=self.hough_upper_threshold,
            param2=self.hough_lower_threshold,
            minRadius=self.hough_min_radius,
            maxRadius=self.hough_max_radius,
        )
        
        if circles is None:
            return []
        
        # Convert to integer coordinates
        circles = np.round(circles[0, :]).astype(int)
        return [(x, y, r) for x, y, r in circles]
    
    def extract_patch(
        self, image: np.ndarray, x: int, y: int, r: int, target_size: Tuple[int, int]
    ) -> np.ndarray:
        """Extract and resize a square patch around a circle.
        
        Args:
            image: Source image
            x, y: Center coordinates
            r: Radius
            target_size: Desired output size (width, height)
        
        Returns:
            Resized patch normalized to [0, 1]
        """
        # Extract square region
        y1, y2 = max(0, y - r), min(image.shape[0], y + r)
        x1, x2 = max(0, x - r), min(image.shape[1], x + r)
        patch = image[y1:y2, x1:x2]
        
        if patch.size == 0:
            # Return black patch if extraction failed
            return np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)
        
        # Resize to target size
        resized = cv2.resize(patch, target_size)
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def classify_candidates(
        self, image: np.ndarray, candidates: List[Tuple[int, int, int]]
    ) -> List[Tuple[int, int, int, float]]:
        """Classify each candidate as screw or non-screw.
        
        Args:
            image: Source image
            candidates: List of (x, y, radius) tuples
        
        Returns:
            List of (x, y, radius, confidence) for detected screws
        """
        if not candidates:
            return []
        
        detections = []
        
        for x, y, r in candidates:
            # Extract patches for both models
            patch1 = self.extract_patch(image, x, y, r, self.image_size1)
            patch2 = self.extract_patch(image, x, y, r, self.image_size2)
            
            # Batch dimension
            batch1 = np.expand_dims(patch1, axis=0)
            batch2 = np.expand_dims(patch2, axis=0)
            
            # Get predictions from both models
            pred1 = self.model1.predict(batch1, verbose=0)[0]
            pred2 = self.model2.predict(batch2, verbose=0)[0]
            
            # Average the predictions (ensemble)
            screw_confidence = (pred1[1] + pred2[1]) / 2.0
            
            # If confidence exceeds threshold, it's a screw
            if screw_confidence >= self.detection_threshold:
                detections.append((x, y, r, float(screw_confidence)))
        
        return detections
    
    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, float]]:
        """Run complete detection pipeline.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            List of (x, y, radius, confidence) for detected screws
        """
        print(f"Finding circular candidates...")
        candidates = self.find_candidates(image)
        print(f"Found {len(candidates)} circular candidates")
        
        if not candidates:
            return []
        
        print(f"Classifying candidates...")
        detections = self.classify_candidates(image, candidates)
        print(f"Detected {len(detections)} screws")
        
        return detections
    
    def visualize(
        self,
        image: np.ndarray,
        detections: List[Tuple[int, int, int, float]],
        output_path: Path = None,
    ) -> np.ndarray:
        """Draw detection results on image.
        
        Args:
            image: Source image
            detections: List of (x, y, radius, confidence) tuples
            output_path: Optional path to save the result
        
        Returns:
            Image with drawn detections
        """
        result = image.copy()
        
        for x, y, r, conf in detections:
            # Draw bounding box
            cv2.rectangle(result, (x - r, y - r), (x + r, y + r), (0, 255, 0), 2)
            
            # Draw circle
            cv2.circle(result, (x, y), r, (0, 255, 0), 2)
            
            # Draw confidence
            label = f"{conf:.2f}"
            cv2.putText(
                result,
                label,
                (x - r, y - r - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        
        if output_path:
            cv2.imwrite(str(output_path), result)
            print(f"Result saved to {output_path}")
        
        return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument("--model1", type=Path, default=Path("models/xception.h5"), help="Xception model weights")
    parser.add_argument("--model2", type=Path, default=Path("models/inceptionv3.h5"), help="InceptionV3 model weights")
    parser.add_argument("--output", type=Path, help="Output image path (optional)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--show", action="store_true", help="Display result window")
    
    # Hough parameters
    parser.add_argument("--hough-upper", type=int, default=100, help="Canny upper threshold")
    parser.add_argument("--hough-lower", type=int, default=50, help="Circle accumulator threshold")
    parser.add_argument("--min-radius", type=int, default=5, help="Minimum circle radius")
    parser.add_argument("--max-radius", type=int, default=30, help="Maximum circle radius")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.image.exists():
        parser.error(f"Image not found: {args.image}")
    if not args.model1.exists():
        parser.error(f"Model 1 not found: {args.model1}")
    if not args.model2.exists():
        parser.error(f"Model 2 not found: {args.model2}")
    
    print("=" * 80)
    print("SCREW DETECTION PIPELINE".center(80))
    print("=" * 80)
    print(f"\nInput: {args.image}")
    print(f"Models: {args.model1.name} + {args.model2.name}")
    print(f"Threshold: {args.threshold}")
    print()
    
    # Load image
    print("Loading image...")
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f"Failed to load image: {args.image}")
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    print()
    
    try:
        # Initialize detector
        detector = ScrewDetector(
            model1_path=args.model1,
            model2_path=args.model2,
            hough_upper_threshold=args.hough_upper,
            hough_lower_threshold=args.hough_lower,
            hough_min_radius=args.min_radius,
            hough_max_radius=args.max_radius,
            detection_threshold=args.threshold,
            use_gpu=args.use_gpu,
        )
        print()
        
        # Run detection
        detections = detector.detect(image)
        print()
        
        # Display results
        print("=" * 80)
        print("RESULTS".center(80))
        print("=" * 80)
        print(f"\nDetected {len(detections)} screws:\n")
        
        for i, (x, y, r, conf) in enumerate(detections, 1):
            print(f"  {i}. Position: ({x}, {y}), Radius: {r}, Confidence: {conf:.3f}")
        
        # Visualize
        if args.output or args.show:
            result = detector.visualize(image, detections, args.output)
            
            if args.show:
                cv2.imshow("Screw Detection", result)
                print("\nPress any key to close the window...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        print("\n" + "=" * 80)
        
    except MissingDependencyError as e:
        parser.error(str(e))
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
