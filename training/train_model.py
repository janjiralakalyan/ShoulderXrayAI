"""
Training script for Shoulder X-ray Classification Model.

Uses the MURA (Musculoskeletal Radiograph) dataset to train an
EfficientNetB0-based binary classifier for shoulder X-rays.

This script can be run:
  1. Locally:  python training/train_model.py
  2. On Google Colab: Copy-paste into a notebook cell

DATASET SETUP:
  1. Download the MURA dataset from:
     https://stanfordmlgroup.github.io/competitions/mura/
  2. Extract it so the structure looks like:
     data/
       MURA-v1.1/
         train/
           XR_SHOULDER/
             patient00001/
               study1_positive/
                 image1.png
               study1_negative/
                 image1.png
         valid/
           XR_SHOULDER/
             ...
"""

import os
import sys
import glob

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10
LEARNING_RATE = 1e-3
FINE_TUNE_LR = 1e-5

# Paths — adjust these if your dataset location differs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "MURA-v1.1")
TRAIN_DIR = os.path.join(DATA_DIR, "train", "XR_SHOULDER")
VALID_DIR = os.path.join(DATA_DIR, "valid", "XR_SHOULDER")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "backend", "model.h5")


# ─────────────────────────────────────────────
# Dataset Preparation
# ─────────────────────────────────────────────

def prepare_dataframe(root_dir: str) -> tuple[list[str], list[int]]:
    """
    Walk the MURA shoulder directory and create lists of
    (file_path, label) where label is 0=NORMAL, 1=ABNORMAL.

    MURA convention:
      - Folder name contains "positive" → ABNORMAL (1)
      - Folder name contains "negative" → NORMAL (0)
    """
    file_paths = []
    labels = []

    for patient_dir in sorted(glob.glob(os.path.join(root_dir, "patient*"))):
        for study_dir in sorted(glob.glob(os.path.join(patient_dir, "study*"))):
            label = 1 if "positive" in study_dir.lower() else 0
            for img_path in glob.glob(os.path.join(study_dir, "*.png")):
                file_paths.append(img_path)
                labels.append(label)

    return file_paths, labels


def create_dataset(
    file_paths: list[str],
    labels: list[int],
    augment: bool = False,
) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset from file paths and labels.
    Uses EfficientNet's native preprocess_input (expects [0, 255] range).
    Applies augmentation if augment=True.
    """

    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        # EfficientNet preprocess_input expects float [0, 255]
        img = tf.cast(img, tf.float32)
        img = preprocess_input(img)
        return img, label

    def augment_image(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.15)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        # Random rotation via crop & resize
        img = tf.image.resize_with_crop_or_pad(img, 248, 248)
        img = tf.image.random_crop(img, size=[224, 224, 3])
        return img, label

    dataset = tf.data.Dataset.from_tensor_slices(
        (file_paths, labels)
    )
    dataset = dataset.shuffle(buffer_size=len(file_paths), seed=42)
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset


# ─────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────

def build_model() -> tf.keras.Model:
    """
    Build the EfficientNetB0-based classification model.

    Architecture:
      - EfficientNetB0 (frozen, pretrained on ImageNet)
      - GlobalAveragePooling2D
      - Dense(128, relu)
      - Dropout(0.5)
      - Dense(1, sigmoid)
    """
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )
    base_model.trainable = False  # Freeze base layers initially

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train():
    """Main training function with two-phase approach."""

    print("=" * 60)
    print("  Shoulder X-ray Classification — Model Training")
    print("=" * 60)

    # ── Verify dataset exists ──
    if not os.path.exists(TRAIN_DIR):
        print(f"\n❌ Training data not found at: {TRAIN_DIR}")
        print(
            "\nPlease download the MURA dataset from:\n"
            "  https://stanfordmlgroup.github.io/competitions/mura/\n"
            "\nAnd extract it to:\n"
            f"  {DATA_DIR}\n"
        )
        sys.exit(1)

    # ── Prepare data ──
    print("\n📂 Loading dataset...")
    train_paths, train_labels = prepare_dataframe(TRAIN_DIR)
    valid_paths, valid_labels = prepare_dataframe(VALID_DIR)

    print(f"   Training   : {len(train_paths)} images")
    print(f"     Normal   : {train_labels.count(0)}")
    print(f"     Abnormal : {train_labels.count(1)}")
    print(f"   Validation : {len(valid_paths)} images")
    print(f"     Normal   : {valid_labels.count(0)}")
    print(f"     Abnormal : {valid_labels.count(1)}")

    # ── Create datasets ──
    print("\n⚙️  Creating data pipelines (with augmentation for training)...")
    train_dataset = create_dataset(train_paths, train_labels, augment=True)
    valid_dataset = create_dataset(valid_paths, valid_labels, augment=False)

    # ── Build model ──
    print("\n🧠 Building EfficientNetB0 model...")
    model = build_model()
    model.summary()

    # ── Callbacks ──
    callbacks = [
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    # ══════════════════════════════════════════
    # Phase 1: Train with frozen base
    # ══════════════════════════════════════════
    print(f"\n🚀 Phase 1: Training head layers ({INITIAL_EPOCHS} epochs, base frozen)...")
    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=INITIAL_EPOCHS,
        callbacks=callbacks,
    )

    phase1_acc = max(history.history["val_accuracy"])
    print(f"\n   Phase 1 best val_accuracy: {phase1_acc:.2%}")

    # ══════════════════════════════════════════
    # Phase 2: Fine-tune top layers of EfficientNet
    # ══════════════════════════════════════════
    print(f"\n🔧 Phase 2: Fine-tuning top layers ({FINE_TUNE_EPOCHS} epochs)...")

    # Unfreeze the base model
    base_model = model.layers[0]
    base_model.trainable = True

    # Freeze all layers except the last 20
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"   Unfrozen {trainable_count} of {len(base_model.layers)} base layers")

    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=FINE_TUNE_LR),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # Update callbacks for phase 2
    callbacks_phase2 = [
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-8,
            verbose=1,
        ),
    ]

    history_fine = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=callbacks_phase2,
    )

    # ── Results ──
    best_val_acc = max(history_fine.history["val_accuracy"])
    print("\n" + "=" * 60)
    print(f"  ✅ Training complete!")
    print(f"  📊 Phase 1 best val_accuracy: {phase1_acc:.2%}")
    print(f"  📊 Phase 2 best val_accuracy: {best_val_acc:.2%}")
    print(f"  💾 Model saved to: {MODEL_SAVE_PATH}")
    print("=" * 60)

    return model, history_fine


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    train()
