"""
3-class: normal / smoke / fire — same label order as the web UI (0 normal, 1 smoke, 2 fire).
Preprocessing: x = x_float / 127.0 - 1.0, matches Classify.js processImage.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

CLASS_NAMES = ["normal", "smoke", "fire"]


def preprocess_tfjs_style(image: np.ndarray) -> np.ndarray:
    """ImageDataGenerator passes uint8 [0,255] HWC."""
    return image.astype(np.float32) / 127.0 - 1.0


def build_model() -> keras.Model:
    base = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling="avg",
    )
    base.trainable = False
    inputs = base.input
    out = layers.Dense(len(CLASS_NAMES), activation="softmax", name="probs")(base.output)
    model = keras.Model(inputs, out, name="fire_smoke_mobilenet")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def val_dir_ready(val_dir: Path) -> bool:
    if not val_dir.is_dir():
        return False
    for name in CLASS_NAMES:
        d = val_dir / name
        if not d.is_dir():
            return False
        if not any(d.iterdir()):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    train_dir = root / "dataset" / "train"
    val_dir = root / "dataset" / "val"

    if not train_dir.is_dir():
        raise SystemExit(f"Missing training directory: {train_dir}")

    gpus = tf.config.list_physical_devices("GPU")
    print("TensorFlow:", tf.__version__)
    print("Visible GPU devices:", gpus)
    if not gpus:
        print(
            "Warning: no GPU detected; training on CPU will be slow. "
            "Install NVIDIA drivers/CUDA matching your TensorFlow build, or use WSL2+GPU; "
            "see scripts/README.txt."
        )
    else:
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    use_split = not val_dir_ready(val_dir)
    if use_split:
        print("No complete val/{normal,smoke,fire}; using 20% of train for validation.")

    train_aug = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_tfjs_style,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=(0.85, 1.15),
        validation_split=0.2 if use_split else 0.0,
    )
    train_flow = train_aug.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=args.batch_size,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=True,
        subset="training" if use_split else None,
    )
    val_flow = None
    if use_split:
        val_flow = train_aug.flow_from_directory(
            train_dir,
            target_size=(224, 224),
            batch_size=args.batch_size,
            class_mode="categorical",
            classes=CLASS_NAMES,
            shuffle=False,
            subset="validation",
        )
    else:
        val_plain = keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocess_tfjs_style,
        )
        val_flow = val_plain.flow_from_directory(
            val_dir,
            target_size=(224, 224),
            batch_size=args.batch_size,
            class_mode="categorical",
            classes=CLASS_NAMES,
            shuffle=False,
        )

    model = build_model()
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=4, restore_best_weights=True, monitor="val_accuracy"
        ),
    ]
    model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    keras_path = models_dir / "fire_smoke.keras"
    model.save(keras_path)
    print(f"Saved Keras model: {keras_path}")

    out_tfjs = root / "tfjs-web-app" / "public" / "model"
    out_tfjs.mkdir(parents=True, exist_ok=True)
    try:
        import tensorflowjs as tfjs

        tfjs.converters.save_keras_model(model, str(out_tfjs))
        print(f"Exported TF.js to: {out_tfjs}")
        print(
            "Frontend: set REACT_APP_MODEL_URL=/model/model.json and "
            "REACT_APP_NUM_CLASSES=3 in tfjs-web-app/.env, then restart npm run start-dev."
        )
    except Exception as e:
        print(f"TF.js export skipped (dependency conflict possible): {e}")
        print(
            "Run later: python scripts/export_keras_to_tfjs.py "
            "models/fire_smoke.keras tfjs-web-app/public/model"
        )


if __name__ == "__main__":
    main()
