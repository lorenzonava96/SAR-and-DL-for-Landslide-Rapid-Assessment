from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any
from app.config import ModelLoadConfig


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def focal_loss(alpha: float = 0.25, gamma: float = 2.0):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("Model inference requires the 'inference' extra") from exc
    def loss(y_true: Any, y_pred: Any) -> Any:
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)
        p_t = tf.exp(-bce)
        return alpha * (1 - p_t) ** gamma * bce
    return loss


def build_model(config: ModelLoadConfig) -> Any:
    """Construct the orbit-specific CNN. Does not load or download weights."""
    try:
        from tensorflow.keras import Model, Input
        from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Concatenate
        from tensorflow.keras.optimizers import Adam
    except ImportError as exc:
        raise RuntimeError("Model inference requires the 'inference' extra") from exc
    inputs = Input(shape=(config.patch_size, config.patch_size, config.channels))
    x = inputs
    for _ in range(2):
        x = Conv2D(config.filters_first_layer, 3, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D()(x)
    x = Conv2D(config.filters_first_layer, 3, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Concatenate(axis=-1)([x, x, x])
    x = MaxPooling2D()(x)
    x = Dropout(config.dropout)(x)
    x = Flatten()(x)
    x = Dense(config.filters_first_layer * 8, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs, outputs)
    model.compile(loss=focal_loss(), optimizer=Adam(config.learning_rate), metrics=["accuracy"])
    return model


def load_model(config: ModelLoadConfig) -> tuple[Any, str]:
    """Build and load a local weight file. Network downloads are intentionally excluded."""
    if not config.weights_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {config.weights_path}")
    model = build_model(config)
    model.load_weights(config.weights_path)
    return model, sha256_file(config.weights_path)
