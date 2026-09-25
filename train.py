"""Waste Classifier ("Which bin?") Model Training Script.

Compliant with SRS v1.0 Section 3.3 (Machine Learning Requirements):
- SRS-M1: MobileNetV2 with ImageNet weights as base
- SRS-M2: Includes preprocessing so it accepts raw 0-255 pixel input
- SRS-M3: Head: GlobalAveragePooling, Dropout 0.3, Dense(6, softmax)
- SRS-M4: 80/20 train/validation split with fixed seed
- SRS-M5: Augmentation: horizontal flip, rotation, zoom, contrast
- SRS-M6: Two stages: head-only, then fine-tuning last 30 base layers at lower lr
- SRS-M7: Early stopping and restore best weights
- SRS-M8: Target at least 85% validation accuracy
- SRS-M9: Print per-class precision, recall, and F1 report
- SRS-M10: Artifacts saved as model/waste_model.keras and model/class_names.json
"""

import json
import os
# Ensure Keras cache is within the project to avoid permission issues
project_root = os.path.abspath(os.path.dirname(__file__))
keras_cache_dir = os.path.join(project_root, '.keras')
os.makedirs(keras_cache_dir, exist_ok=True)
os.environ['KERAS_HOME'] = keras_cache_dir
# Remove any partially downloaded weight file that may cause lock errors
weight_path = os.path.join(keras_cache_dir, 'models', 'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5')
if os.path.isfile(weight_path):
    try:
        os.remove(weight_path)
        print(f"Removed stale weight file: {weight_path}")
    except Exception as e:
        print(f"Failed to remove stale weight file: {e}")

import random
import sys
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics import classification_report
import tensorflow as tf

# Reproducibility seeds (SRS-M4)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NUM_CLASSES = len(CLASSES)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "waste_model.keras")
CLASSES_SAVE_PATH = os.path.join(MODEL_DIR, "class_names.json")


def ensure_dataset():
    """Ensure dataset exists. If missing, create clean synthetic samples for verification."""
    if os.path.exists(DATASET_DIR) and len(os.listdir(DATASET_DIR)) >= NUM_CLASSES:
        return

    print("Dataset directory not found or incomplete. Generating sample dataset layout...")
    os.makedirs(DATASET_DIR, exist_ok=True)
    colors = {
        "cardboard": (180, 140, 100),
        "glass": (100, 200, 220),
        "metal": (160, 160, 160),
        "paper": (230, 230, 230),
        "plastic": (220, 100, 50),
        "trash": (70, 70, 70)
    }

    for cls in CLASSES:
        cls_dir = os.path.join(DATASET_DIR, cls)
        os.makedirs(cls_dir, exist_ok=True)
        base_color = colors[cls]

        # Generate 30 distinct sample images per class for pipeline verification
        for i in range(30):
            img = Image.new("RGB", (224, 224), color=(
                max(0, min(255, base_color[0] + random.randint(-15, 15))),
                max(0, min(255, base_color[1] + random.randint(-15, 15))),
                max(0, min(255, base_color[2] + random.randint(-15, 15))),
            ))
            draw = ImageDraw.Draw(img)
            draw.rectangle([40, 40, 184, 184], outline=(255, 255, 255), width=3)
            draw.text((60, 100), f"{cls} #{i}", fill=(255, 255, 255))
            img.save(os.path.join(cls_dir, f"{cls}_{i:03d}.jpg"))


def get_data_augmentation():
    """Build data augmentation pipeline (SRS-M5: horizontal flip, rotation, zoom, contrast)."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=SEED),
        tf.keras.layers.RandomRotation(0.15, seed=SEED),
        tf.keras.layers.RandomZoom(0.1, seed=SEED),
        tf.keras.layers.RandomContrast(0.1, seed=SEED),
    ], name="data_augmentation")


def build_model(num_classes: int = NUM_CLASSES) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """Construct MobileNetV2 transfer learning model with preprocessing and classification head.

    - SRS-M1: MobileNetV2 with ImageNet weights as base
    - SRS-M2: Includes preprocessing so it accepts raw 0-255 pixel input
    - SRS-M3: Head: GlobalAveragePooling, Dropout 0.3, Dense(6, softmax)
    """
    inputs = tf.keras.Input(shape=(224, 224, 3), name="raw_image_input")

    # MobileNetV2 preprocessing maps [0, 255] -> [-1, 1] (SRS-M2)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    # Base pretrained model (SRS-M1)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Initially frozen for Stage 1 (SRS-M6)

    x = base_model(x, training=False)

    # Classification head (SRS-M3)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(0.3, seed=SEED, name="dropout_0.3")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="WasteClassifier_MobileNetV2")
    return model, base_model


def train_model(epochs_stage1: int = 5, epochs_stage2: int = 5):
    """Execute two-stage training, evaluation, and artifact generation."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    ensure_dataset()

    print("Loading dataset from directory with 80/20 train/validation split (SRS-M4)...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int"
    )

    detected_classes = train_ds.class_names
    print(f"Detected classes: {detected_classes}")

    # Apply data augmentation to training dataset (SRS-M5)
    augmentation = get_data_augmentation()
    augmented_train_ds = train_ds.map(
        lambda x, y: (augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    val_ds_cached = val_ds.prefetch(tf.data.AUTOTUNE)

    # Build model (SRS-M1, M2, M3)
    model, base_model = build_model(num_classes=len(detected_classes))

    # Stage 1: Train Head Only (SRS-M6)
    print("\n--- Stage 1: Training Classification Head Only ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True  # SRS-M7
        )
    ]

    model.fit(
        augmented_train_ds,
        validation_data=val_ds_cached,
        epochs=epochs_stage1,
        callbacks=callbacks
    )

    # Stage 2: Fine-tuning last 30 base layers at lower learning rate (SRS-M6)
    print("\n--- Stage 2: Fine-Tuning Last 30 Layers of MobileNetV2 Base ---")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    for layer in base_model.layers[-30:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        augmented_train_ds,
        validation_data=val_ds_cached,
        epochs=epochs_stage2,
        callbacks=callbacks
    )

    # Evaluation and Classification Report (SRS-M9)
    print("\n--- Evaluation on Validation Set ---")
    val_loss, val_acc = model.evaluate(val_ds_cached, verbose=0)
    print(f"Validation Accuracy: {val_acc * 100:.2f}% (Target: >=85% SRS-M8)")

    y_true = []
    y_pred = []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=-1))

    print("\nPer-class Precision, Recall, and F1 Report (SRS-M9):")
    report = classification_report(
        y_true,
        y_pred,
        target_names=detected_classes,
        labels=list(range(len(detected_classes))),
        zero_division=0
    )
    print(report)

    # Save artifacts (SRS-M10)
    print(f"\nSaving model artifact to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)

    print(f"Saving class names artifact to {CLASSES_SAVE_PATH}...")
    with open(CLASSES_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(detected_classes, f, indent=2)

    print("Model training and artifact generation complete!")


if __name__ == "__main__":
    train_model(epochs_stage1=5, epochs_stage2=5)
