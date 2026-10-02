import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from asl_utils import save_label_mapping

# --------------- CONFIGURATION --------------- #
IMG_SIZE = 300
BATCH_SIZE = 32
BASE_EPOCHS = 6       # Frozen base initial training
FINETUNE_EPOCHS = 12  # Fine-tuning unfrozen top layers
DATA_DIR = "Data"
MODEL_PATH = "sign_model.h5"
LABELS_PATH = "labels.json"
# --------------------------------------------- #

def train():
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Dataset directory '{DATA_DIR}' not found. Run datacollection.py first.")

    class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    num_classes = len(class_names)
    print(f"[train_model] Found {num_classes} classes: {class_names}")

    if num_classes == 0:
        raise ValueError(f"No class folders found in '{DATA_DIR}'.")

    # Data Generator with robust augmentation
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=False,  # Sign language orientation matters
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # Save class indices mapping for zero-overhead prediction
    idx_to_label = {v: k for k, v in train_gen.class_indices.items()}
    save_label_mapping(idx_to_label, LABELS_PATH)

    # Build Lightweight Transfer Learning Model with MobileNetV2
    print("[train_model] Building MobileNetV2 backbone...")
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False  # Freeze pre-trained weights

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),  # Drastically reduces param count from 26M -> ~250K vs Flatten()
        BatchNormalization(),
        Dense(256, activation='relu'),
        Dropout(0.35),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    ]

    print("\n--- STAGE 1: Training Classification Head (Base Frozen) ---\n")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=BASE_EPOCHS,
        callbacks=callbacks
    )

    print("\n--- STAGE 2: Fine-Tuning Backbone Layers ---\n")
    base_model.trainable = True
    fine_tune_at = len(base_model.layers) - 40  # Unfreeze last 40 layers

    for i, layer in enumerate(base_model.layers):
        layer.trainable = (i >= fine_tune_at)

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=FINETUNE_EPOCHS,
        callbacks=callbacks
    )

    print(f"\n[train_model] Training complete! Best model saved to {MODEL_PATH} & labels to {LABELS_PATH}")

if __name__ == "__main__":
    train()
