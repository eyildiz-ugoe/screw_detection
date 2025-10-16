"""High level helpers for training and evaluating the classification models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

try:  # OpenCV is optional when only training with tf.keras generators.
    import cv2
except ImportError:  # pragma: no cover - handled dynamically in evaluate_directory
    cv2 = None  # type: ignore


class MissingDependencyError(RuntimeError):
    """Raised when an optional dependency is required but not installed."""


@dataclass(frozen=True)
class ArchitectureConfig:
    """Configuration for a classifier architecture."""

    name: str
    image_size: Tuple[int, int]
    builder: Callable[[Any, Tuple[int, int], Optional[str]], "tf.keras.Model"]


def ensure_tensorflow():  # type: ignore[override]
    """Import TensorFlow lazily and provide a helpful error message if missing."""

    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised when TF is unavailable
        raise MissingDependencyError(
            "TensorFlow is required for the training and inference scripts. "
            "Install it with `pip install tensorflow` before running these commands."
        ) from exc
    return tf


def _ensure_cv2():
    if cv2 is None:  # pragma: no cover - evaluated when OpenCV is missing
        raise MissingDependencyError(
            "OpenCV is required for image based evaluation. Install it with "
            "`pip install opencv-python`."
        )
    return cv2


def _ensure_numpy():
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised when NumPy is unavailable
        raise MissingDependencyError(
            "NumPy is required for batching images during evaluation. "
            "Install it with `pip install numpy`."
        ) from exc
    return np


def _build_xception(tf, image_size, base_weights):
    return tf.keras.applications.Xception(
        include_top=False,
        weights=base_weights,
        input_shape=(image_size[0], image_size[1], 3),
    )


def _build_inception_v3(tf, image_size, base_weights):
    return tf.keras.applications.InceptionV3(
        include_top=False,
        weights=base_weights,
        input_shape=(image_size[0], image_size[1], 3),
    )


def _build_inception_resnet_v2(tf, image_size, base_weights):
    return tf.keras.applications.InceptionResNetV2(
        include_top=False,
        weights=base_weights,
        input_shape=(image_size[0], image_size[1], 3),
    )


def _build_densenet201(tf, image_size, base_weights):
    return tf.keras.applications.DenseNet201(
        include_top=False,
        weights=base_weights,
        input_shape=(image_size[0], image_size[1], 3),
    )


def _build_resnet101_v2(tf, image_size, base_weights):
    return tf.keras.applications.ResNet101V2(
        include_top=False,
        weights=base_weights,
        input_shape=(image_size[0], image_size[1], 3),
    )


def _build_resnext101(tf, image_size, base_weights):
    import sys
    from pathlib import Path
    
    # Add src directory to path if not already present
    src_dir = Path(__file__).resolve().parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    import resnet

    return resnet.ResNeXt101(
        include_top=False,
        weights=base_weights,
        input_shape=(image_size[0], image_size[1], 3),
    )


AVAILABLE_ARCHITECTURES: Dict[str, ArchitectureConfig] = {
    "xception": ArchitectureConfig("xception", (71, 71), _build_xception),
    "inceptionv3": ArchitectureConfig("inceptionv3", (139, 139), _build_inception_v3),
    "inceptionresnetv2": ArchitectureConfig(
        "inceptionresnetv2", (139, 139), _build_inception_resnet_v2
    ),
    "densenet201": ArchitectureConfig("densenet201", (221, 221), _build_densenet201),
    "resnet101v2": ArchitectureConfig("resnet101v2", (64, 64), _build_resnet101_v2),
    "resnext101": ArchitectureConfig("resnext101", (64, 64), _build_resnext101),
}


def _normalise_architecture_name(name: str) -> str:
    key = name.lower().replace("_", "")
    if key not in AVAILABLE_ARCHITECTURES:
        raise KeyError(
            f"Unknown architecture '{name}'. Available options: {', '.join(sorted(AVAILABLE_ARCHITECTURES))}."
        )
    return key


def build_classifier(
    architecture: str,
    *,
    num_classes: int = 2,
    base_weights: Optional[str] = "imagenet",
    image_size: Optional[Tuple[int, int]] = None,
    dropout_rate: float = 0.5,
    weights_path: Optional[Path] = None,
):
    """Create a classifier model with a global average pooling head.

    Parameters
    ----------
    architecture:
        Identifier of the base network. See :data:`AVAILABLE_ARCHITECTURES`.
    num_classes:
        Number of output classes. Defaults to binary classification.
    base_weights:
        Initial weights for the backbone (``None`` disables pretrained weights).
    image_size:
        Optionally override the default input size for the network.
    dropout_rate:
        Dropout applied before the classification layer.
    weights_path:
        Optional path to a set of weights to load after building the model.
    """

    tf = ensure_tensorflow()
    key = _normalise_architecture_name(architecture)
    config = AVAILABLE_ARCHITECTURES[key]
    target_size = image_size or config.image_size

    base_model = config.builder(tf, target_size, base_weights)
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    if dropout_rate:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs, name=f"{key}_classifier")

    if weights_path is not None:
        model.load_weights(str(weights_path))

    return model


def _create_data_generator(tf, train: bool = True):
    if train:
        return tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255.0,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.01,
            zoom_range=[0.9, 1.25],
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.4, 1.5],
            fill_mode="reflect",
        )
    return tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)


def _build_callbacks(
    tf,
    *,
    checkpoint_path: Optional[Path],
    log_dir: Optional[Path],
) -> List["tf.keras.callbacks.Callback"]:
    callbacks: List["tf.keras.callbacks.Callback"] = []
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                save_best_only=True,
                save_weights_only=True,
                monitor="val_accuracy",
                mode="max",
                verbose=1,
            )
        )
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=str(log_dir)))
    return callbacks


def train_model(
    architecture: str,
    train_dir: Path,
    val_dir: Path,
    *,
    output_weights: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    batch_size: int = 8,
    epochs: int = 15,
    freeze_layers: int = 0,
    learning_rate: float = 1e-4,
    initial_weights: Optional[Path] = None,
) -> "tf.keras.callbacks.History":
    """Train one of the supported classifier models on an image directory."""

    tf = ensure_tensorflow()
    key = _normalise_architecture_name(architecture)
    config = AVAILABLE_ARCHITECTURES[key]

    train_dir = Path(train_dir)
    val_dir = Path(val_dir)
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    model = build_classifier(
        architecture,
        num_classes=2,
        base_weights="imagenet",
        image_size=config.image_size,
        weights_path=initial_weights,
    )

    if freeze_layers:
        for layer in model.layers[:freeze_layers]:
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    train_gen = _create_data_generator(tf, train=True)
    val_gen = _create_data_generator(tf, train=False)

    train_flow = train_gen.flow_from_directory(
        str(train_dir),
        target_size=config.image_size,
        class_mode="categorical",
        shuffle=True,
        batch_size=batch_size,
    )
    val_flow = val_gen.flow_from_directory(
        str(val_dir),
        target_size=config.image_size,
        class_mode="categorical",
        shuffle=False,
        batch_size=batch_size,
    )

    callbacks = _build_callbacks(tf, checkpoint_path=output_weights, log_dir=log_dir)

    history = model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=epochs,
        callbacks=callbacks,
    )

    return history


def _load_images(image_paths: Iterable[Path], image_size: Tuple[int, int]) -> "np.ndarray":
    cv2_module = _ensure_cv2()
    np = _ensure_numpy()
    processed: List[np.ndarray] = []
    for path in image_paths:
        image = cv2_module.imread(str(path))
        if image is None:
            continue
        resized = cv2_module.resize(image, image_size)
        data = resized.astype(np.float32) / 255.0
        processed.append(data)
    if not processed:
        return np.empty((0, image_size[0], image_size[1], 3), dtype=np.float32)
    return np.stack(processed, axis=0)


def evaluate_directory(
    architecture: str,
    weights_path: Path,
    *,
    screw_dir: Path,
    non_screw_dir: Path,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Evaluate a trained classifier on two folders of images."""

    tf = ensure_tensorflow()
    key = _normalise_architecture_name(architecture)
    config = AVAILABLE_ARCHITECTURES[key]

    screw_dir = Path(screw_dir)
    non_screw_dir = Path(non_screw_dir)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    if not screw_dir.exists():
        raise FileNotFoundError(f"Screw directory not found: {screw_dir}")
    if not non_screw_dir.exists():
        raise FileNotFoundError(f"Non screw directory not found: {non_screw_dir}")

    model = build_classifier(
        architecture,
        num_classes=2,
        base_weights=None,
        image_size=config.image_size,
        weights_path=weights_path,
    )

    screw_images = sorted(p for p in screw_dir.iterdir() if p.suffix.lower() not in {".txt"})
    non_screw_images = sorted(p for p in non_screw_dir.iterdir() if p.suffix.lower() not in {".txt"})

    screw_batch = _load_images(screw_images, config.image_size)
    non_screw_batch = _load_images(non_screw_images, config.image_size)

    tp = tn = fp = fn = 0

    if screw_batch.size:
        preds = model.predict(screw_batch, verbose=0)
        labels = preds[:, 1] >= threshold
        tp = int(labels.sum())
        fn = int(labels.shape[0] - tp)

    if non_screw_batch.size:
        preds = model.predict(non_screw_batch, verbose=0)
        labels = preds[:, 1] >= threshold
        fp = int(labels.sum())
        tn = int(labels.shape[0] - fp)

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    return {
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "accuracy": accuracy,
    }
