#!/usr/bin/env python
"""Test inference on all 6 pre-trained models to verify they work correctly.

This script tests all available model architectures on the test dataset and
reports their accuracy. It's useful for validating that all models are properly
configured and can load their weights successfully.

Usage:
    python test_all_models.py [--use-gpu]

Options:
    --use-gpu    Use GPU if available (default: CPU only to avoid CUDA issues)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path for imports
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipelines import AVAILABLE_ARCHITECTURES, MissingDependencyError, evaluate_directory


def test_model(
    architecture: str,
    weights_path: Path,
    screw_dir: Path,
    non_screw_dir: Path,
    use_gpu: bool = False,
) -> Optional[Dict[str, float]]:
    """Test a single model and return its metrics.
    
    Args:
        architecture: Name of the architecture (e.g., 'xception')
        weights_path: Path to the .h5 weights file
        screw_dir: Directory containing screw images
        non_screw_dir: Directory containing non-screw images
        use_gpu: Whether to use GPU (default: CPU only)
    
    Returns:
        Dictionary with metrics or None if test failed
    """
    if not weights_path.exists():
        print(f"  ⚠️  Weights not found: {weights_path}")
        return None
    
    try:
        # Disable GPU if requested
        if not use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        metrics = evaluate_directory(
            architecture,
            weights_path=weights_path,
            screw_dir=screw_dir,
            non_screw_dir=non_screw_dir,
            threshold=0.5,
        )
        return metrics
    except MissingDependencyError as e:
        print(f"  ❌ Missing dependency: {e}")
        return None
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics for display."""
    return (
        f"TP: {metrics['tp']:4.0f} | "
        f"TN: {metrics['tn']:4.0f} | "
        f"FP: {metrics['fp']:2.0f} | "
        f"FN: {metrics['fn']:3.0f} | "
        f"Accuracy: {metrics['accuracy']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU if available (default: CPU only)")
    parser.add_argument("--weights-dir", type=Path, default=Path("models"), help="Directory containing model weights")
    parser.add_argument("--screw-dir", type=Path, default=Path("ScrewDTF/Test/label_1"), help="Directory with screw images")
    parser.add_argument("--non-screw-dir", type=Path, default=Path("ScrewDTF/Test/label_0"), help="Directory with non-screw images")
    args = parser.parse_args()
    
    # Validate directories
    if not args.weights_dir.exists():
        parser.error(f"Weights directory not found: {args.weights_dir}")
    if not args.screw_dir.exists():
        parser.error(f"Screw directory not found: {args.screw_dir}")
    if not args.non_screw_dir.exists():
        parser.error(f"Non-screw directory not found: {args.non_screw_dir}")
    
    print("=" * 100)
    print("TESTING ALL PRE-TRAINED MODELS".center(100))
    print("=" * 100)
    print(f"\nMode: {'GPU' if args.use_gpu else 'CPU (safe mode)'}")
    print(f"Weights: {args.weights_dir}")
    print(f"Test data: {args.screw_dir.parent}")
    print()
    
    # Model weight filenames (some have different casing)
    weight_files = {
        "xception": "xception.h5",
        "inceptionv3": "inceptionv3.h5",
        "inceptionresnetv2": "inceptionResNetv2.h5",  # Note the casing
        "densenet201": "densenet201.h5",
        "resnet101v2": "resnet101v2.h5",
        "resnext101": "resnext101.h5",
    }
    
    results: List[tuple] = []
    
    for i, (arch_key, config) in enumerate(AVAILABLE_ARCHITECTURES.items(), 1):
        weight_file = weight_files[arch_key]
        weights_path = args.weights_dir / weight_file
        
        print(f"[{i}/6] Testing {config.name.upper():20s} ({config.image_size[0]}×{config.image_size[1]})...")
        
        metrics = test_model(
            arch_key,
            weights_path,
            args.screw_dir,
            args.non_screw_dir,
            use_gpu=args.use_gpu,
        )
        
        if metrics:
            print(f"      ✅ {format_metrics(metrics)}")
            results.append((config.name, config.image_size, metrics))
        else:
            print(f"      ❌ Test failed")
            results.append((config.name, config.image_size, None))
        
        print()
    
    # Print summary
    print("=" * 100)
    print("SUMMARY".center(100))
    print("=" * 100)
    print()
    print(f"{'Model':<20s} | {'Size':<10s} | {'TP':>4s} | {'TN':>4s} | {'FP':>2s} | {'FN':>3s} | {'Accuracy':>8s}")
    print("-" * 100)
    
    successful = 0
    failed = 0
    best_accuracy = 0.0
    best_model = ""
    
    for name, size, metrics in results:
        if metrics:
            acc = metrics['accuracy']
            status = "✅" if acc > 0.95 else ""
            print(
                f"{name:<20s} | {size[0]:>3d}×{size[1]:<3d}    | "
                f"{metrics['tp']:>4.0f} | {metrics['tn']:>4.0f} | "
                f"{metrics['fp']:>2.0f} | {metrics['fn']:>3.0f} | "
                f"{acc:>7.2%} {status}"
            )
            successful += 1
            if acc > best_accuracy:
                best_accuracy = acc
                best_model = name
        else:
            print(f"{name:<20s} | {size[0]:>3d}×{size[1]:<3d}    | FAILED ❌")
            failed += 1
    
    print("-" * 100)
    print(f"\nResults: {successful}/6 models passed, {failed} failed")
    
    if best_model:
        print(f"🏆 Best performer: {best_model} ({best_accuracy:.2%})")
    
    print("\n" + "=" * 100)
    
    # Exit with error code if any tests failed
    if failed > 0:
        sys.exit(1)
    else:
        print("\n✅ All models tested successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
