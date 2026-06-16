"""
disease_detection/train_cnn.py
================================
Trains a MobileNetV2-based classifier on the PlantVillage dataset
(38 classes: 14 crops × healthy/diseased variants).

Download dataset:
    kaggle datasets download -d abdallahalidev/plantvillage-dataset
    unzip to  data/plantvillage/  with sub-folders per class.

Usage:
    python disease_detection/train_cnn.py

Output: core/ml_models/disease_model.h5
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data", "plantvillage")
MODEL_DIR  = os.path.join(BASE_DIR, "core", "ml_models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "disease_model.h5")

# ── Hyper-parameters ─────────────────────────────────────────
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS_HEAD = 10          # train only the new classification head
EPOCHS_FINE = 20          # fine-tune last N layers of MobileNetV2
FINE_TUNE_AT = 100        # unfreeze from this layer index onwards
NUM_CLASSES = 38          # PlantVillage standard split


# ─────────────────────────────────────────────────────────────
#  1.  DATA PIPELINE
# ─────────────────────────────────────────────────────────────

def build_datasets():
    """Return (train_ds, val_ds, class_names) using Keras image_dataset_from_directory."""
    train_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )
    val_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )
    class_names = train_ds.class_names
    print(f"Classes ({len(class_names)}): {class_names}")

    # ── Augmentation (training only) ──────────────────────────
    augment = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.1),
    ], name="augmentation")

    # Normalise to [0, 1]; MobileNetV2 uses preprocess_input internally
    rescale = layers.Rescaling(1.0 / 255)

    train_ds = (
        train_ds
        .map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        .map(lambda x, y: (rescale(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        val_ds
        .map(lambda x, y: (rescale(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, val_ds, class_names


# ─────────────────────────────────────────────────────────────
#  2.  MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────

def build_model(num_classes: int) -> keras.Model:
    """
    MobileNetV2 backbone + custom classification head.
    Phase 1: backbone frozen (fast head training)
    Phase 2: last FINE_TUNE_AT+ layers unfrozen (fine-tuning)
    """
    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False      # Phase 1: freeze backbone

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="AgriBot_Disease_CNN")
    model.summary()
    return model, base


# ─────────────────────────────────────────────────────────────
#  3.  TRAINING
# ─────────────────────────────────────────────────────────────

def train():
    train_ds, val_ds, class_names = build_datasets()
    model, base = build_model(len(class_names))

    # ── Phase 1: train head only ──────────────────────────────
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")],
    )
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]
    print("\n─── Phase 1: Training classification head ───")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, callbacks=callbacks)

    # ── Phase 2: fine-tune backbone ───────────────────────────
    base.trainable = True
    for layer in base.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),   # lower LR for fine-tuning
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")],
    )
    print(f"\n─── Phase 2: Fine-tuning from layer {FINE_TUNE_AT} ───")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE, callbacks=callbacks)

    # ── Save class names alongside model ─────────────────────
    cn_path = os.path.join(os.path.dirname(__file__), "class_names.py")
    with open(cn_path, "w") as f:
        f.write(f"# Auto-generated by train_cnn.py\nDISEASE_CLASS_NAMES = {class_names!r}\n")
    print(f"✅  Class names saved → {cn_path}")

    # ── Evaluate on validation set ────────────────────────────
    loss, acc, top3 = model.evaluate(val_ds, verbose=1)
    print(f"\nFinal val accuracy : {acc:.4f}")
    print(f"Final val top-3    : {top3:.4f}")
    print(f"✅  Model saved     → {MODEL_PATH}")


if __name__ == "__main__":
    train()
